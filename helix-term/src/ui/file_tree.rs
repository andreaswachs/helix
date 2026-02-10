use crate::{
    compositor::{Component, Compositor, Context, Event, EventResult},
    ctrl, key,
    ui::{
        document::render_document,
        text_decorations::DecorationManager,
        EditorView,
    },
};
use helix_core::{text_annotations::TextAnnotations, Position};
use helix_view::{
    editor::{Action, FileTreeConfig},
    graphics::{CursorKind, Margin, Modifier, Rect},
    keyboard::{KeyCode, KeyModifiers},
    view::ViewPosition,
    Document, Editor,
};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
};
use tui::{
    buffer::Buffer as Surface,
    text::Span,
    widgets::{Block, Borders, Widget},
};

/// Minimum width to show preview pane
pub const MIN_AREA_WIDTH_FOR_PREVIEW: u16 = 72;

/// Global storage for the last file tree state (for resume functionality)
static LAST_FILE_TREE_STATE: Mutex<Option<FileTreeState>> = Mutex::new(None);

/// Save the file tree state for later resumption
pub fn save_last_state(state: FileTreeState) {
    if let Ok(mut guard) = LAST_FILE_TREE_STATE.lock() {
        *guard = Some(state);
    }
}

/// Get the last saved file tree state
pub fn get_last_state() -> Option<FileTreeState> {
    LAST_FILE_TREE_STATE.lock().ok().and_then(|guard| guard.clone())
}

/// Check if there's a saved state available
pub fn has_saved_state() -> bool {
    LAST_FILE_TREE_STATE.lock().ok().map_or(false, |guard| guard.is_some())
}
/// Maximum file size to preview (10MB)
pub const MAX_FILE_SIZE_FOR_PREVIEW: u64 = 10 * 1024 * 1024;

/// Cached preview content
pub enum CachedPreview {
    Document(Box<Document>),
    Directory(Vec<(String, bool)>),
    Binary,
    LargeFile,
    NotFound,
}

impl CachedPreview {
    fn placeholder(&self) -> &str {
        match self {
            CachedPreview::Document(_) => "<Invalid file>",
            CachedPreview::Directory(_) => "<Directory>",
            CachedPreview::Binary => "<Binary file>",
            CachedPreview::LargeFile => "<File too large to preview>",
            CachedPreview::NotFound => "<File not found>",
        }
    }
}

pub const ID: &str = "file-tree";

/// Saved state of the file tree for resumption
#[derive(Debug, Clone)]
pub struct FileTreeState {
    pub root: PathBuf,
    pub expanded: HashSet<PathBuf>,
    pub cursor: usize,
    pub scroll_offset: usize,
    pub input_mode: SavedInputMode,
    pub input_buffer: String,
    pub show_preview: bool,
    pub preview_scroll: usize,
}

/// Simplified input mode for saving (excludes transient states)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SavedInputMode {
    Normal,
    Search,
    GoToFolder,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InputMode {
    Normal,
    Search,
    GoToFolder,
    Add,
    Rename,
    ConfirmDelete,
    Help,
}

/// Git status for a file
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitStatus {
    Modified,
    Added,
    Deleted,
    Renamed,
    Untracked,
    Ignored,
}

impl GitStatus {
    fn indicator(&self) -> &'static str {
        match self {
            GitStatus::Modified => "M",
            GitStatus::Added => "A",
            GitStatus::Deleted => "D",
            GitStatus::Renamed => "R",
            GitStatus::Untracked => "?",
            GitStatus::Ignored => "!",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TreeEntry {
    pub path: PathBuf,
    pub is_dir: bool,
    pub depth: usize,
}

/// Search result entry with relative path for display
#[derive(Debug, Clone)]
struct SearchResult {
    path: PathBuf,
    is_dir: bool,
    relative_path: String,
    score: i64,
}

/// Fuzzy match result with score
struct FuzzyMatch {
    score: i64,
}

/// Check if an event is a paste command (Ctrl+V or Cmd+V on macOS)
fn is_paste_event(event: &helix_view::input::KeyEvent) -> bool {
    if let KeyCode::Char('v') = event.code {
        event.modifiers.contains(KeyModifiers::CONTROL)
            || event.modifiers.contains(KeyModifiers::SUPER)
    } else {
        false
    }
}

/// Check if an event is a delete word command (Ctrl+W or Cmd+Backspace on macOS)
fn is_delete_word_event(event: &helix_view::input::KeyEvent) -> bool {
    // Ctrl+W
    if let KeyCode::Char('w') = event.code {
        if event.modifiers.contains(KeyModifiers::CONTROL) {
            return true;
        }
    }
    // Cmd+Backspace (macOS style)
    if event.code == KeyCode::Backspace && event.modifiers.contains(KeyModifiers::SUPER) {
        return true;
    }
    false
}

/// Check if an event is a clear line command (Ctrl+U or Cmd+U)
fn is_clear_line_event(event: &helix_view::input::KeyEvent) -> bool {
    if let KeyCode::Char('u') = event.code {
        event.modifiers.contains(KeyModifiers::CONTROL)
            || event.modifiers.contains(KeyModifiers::SUPER)
    } else {
        false
    }
}

/// Perform fzf-style fuzzy matching on a string
/// Returns None if no match, Some(FuzzyMatch) with score if matched
fn fuzzy_match(pattern: &str, text: &str) -> Option<FuzzyMatch> {
    if pattern.is_empty() {
        return Some(FuzzyMatch { score: 0 });
    }

    let pattern_lower: Vec<char> = pattern.to_lowercase().chars().collect();
    let text_lower: Vec<char> = text.to_lowercase().chars().collect();
    let text_chars: Vec<char> = text.chars().collect();

    if pattern_lower.is_empty() {
        return Some(FuzzyMatch { score: 0 });
    }

    // First pass: check if all pattern chars exist in text in order
    let mut pi = 0;
    for &c in text_lower.iter() {
        if pi < pattern_lower.len() && c == pattern_lower[pi] {
            pi += 1;
        }
    }

    if pi != pattern_lower.len() {
        return None; // Not all pattern chars found
    }

    // Second pass: find optimal match positions
    let positions = find_best_match_positions(&pattern_lower, &text_lower, &text_chars);

    // Calculate score based on match quality
    let score = calculate_match_score(&positions, &text_chars, text.len());

    Some(FuzzyMatch { score })
}

/// Find the best match positions that maximize the score
fn find_best_match_positions(pattern: &[char], text_lower: &[char], text_chars: &[char]) -> Vec<usize> {
    let n = pattern.len();

    if n == 0 {
        return Vec::new();
    }

    // We use a greedy approach with lookahead for performance
    let mut positions = Vec::with_capacity(n);
    let mut last_match: Option<usize> = None;

    // Find all possible positions for each pattern char
    let mut char_positions: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, &c) in text_lower.iter().enumerate() {
        for (pi, &pc) in pattern.iter().enumerate() {
            if c == pc {
                char_positions[pi].push(i);
            }
        }
    }

    // Greedy selection with preference for:
    // 1. Consecutive matches
    // 2. Word boundary matches
    // 3. Earlier positions (shorter path prefix)
    for pi in 0..n {
        let valid_positions: Vec<usize> = char_positions[pi]
            .iter()
            .copied()
            .filter(|&pos| last_match.map_or(true, |last| pos > last))
            .collect();

        if valid_positions.is_empty() {
            // Fallback: use first available position after last match
            break;
        }

        // Score each valid position
        let best_pos = valid_positions
            .iter()
            .copied()
            .max_by_key(|&pos| {
                let mut pos_score: i64 = 0;

                // Bonus for consecutive match
                if let Some(last) = last_match {
                    if pos == last + 1 {
                        pos_score += 100;
                    }
                }

                // Bonus for word boundary
                if is_word_boundary(pos, text_chars) {
                    pos_score += 80;
                }

                // Bonus for being at start
                if pos == 0 {
                    pos_score += 70;
                }

                // Penalty for distance from last match (prefer closer)
                if let Some(last) = last_match {
                    pos_score -= (pos - last - 1) as i64;
                }

                // Small penalty for later positions
                pos_score -= (pos / 10) as i64;

                pos_score
            })
            .unwrap_or(valid_positions[0]);

        positions.push(best_pos);
        last_match = Some(best_pos);
    }

    positions
}

/// Check if position is at a word boundary
fn is_word_boundary(pos: usize, text: &[char]) -> bool {
    if pos == 0 {
        return true;
    }
    let prev = text[pos - 1];
    matches!(prev, '/' | '\\' | '.' | '_' | '-' | ' ')
}

/// Calculate the match score based on positions
fn calculate_match_score(positions: &[usize], text: &[char], text_len: usize) -> i64 {
    if positions.is_empty() {
        return 0;
    }

    let mut score: i64 = 100; // Base score for a match

    // Bonus for consecutive matches
    let mut consecutive_count = 0;
    for i in 1..positions.len() {
        if positions[i] == positions[i - 1] + 1 {
            consecutive_count += 1;
            score += 20 + consecutive_count * 5; // Increasing bonus for longer consecutive runs
        } else {
            consecutive_count = 0;
        }
    }

    // Bonus for word boundary matches
    for &pos in positions {
        if is_word_boundary(pos, text) {
            score += 15;
        }
    }

    // Bonus for match at start
    if positions[0] == 0 {
        score += 25;
    }

    // Bonus for matching at the end (filename usually)
    let last_slash = text.iter().rposition(|&c| c == '/' || c == '\\').unwrap_or(0);
    let matches_after_slash = positions.iter().filter(|&&p| p > last_slash).count();
    score += (matches_after_slash as i64) * 10;

    // Penalty for total gap size (unmatched chars between matches)
    if positions.len() > 1 {
        let total_span = positions[positions.len() - 1] - positions[0] + 1;
        let gap_penalty = (total_span - positions.len()) as i64;
        score -= gap_penalty;
    }

    // Slight preference for shorter paths
    score -= (text_len / 20) as i64;

    score
}

pub struct FileTree {
    /// Flattened list of visible entries (normal mode)
    entries: Vec<TreeEntry>,
    /// Currently selected entry index
    cursor: usize,
    /// Scroll offset for viewport
    scroll_offset: usize,
    /// Root directory
    root: PathBuf,
    /// Set of expanded directories
    expanded: HashSet<PathBuf>,
    /// Current input mode
    input_mode: InputMode,
    /// Input buffer for search/add/rename
    input_buffer: String,
    /// Cursor position in input buffer
    input_cursor: usize,
    /// Search results
    search_results: Vec<SearchResult>,
    /// Path being renamed (stored when entering rename mode)
    rename_target: Option<PathBuf>,
    /// Message to display (errors, confirmations)
    message: Option<String>,
    /// Preview cache for files/directories
    preview_cache: HashMap<Arc<Path>, CachedPreview>,
    /// Buffer for reading file content type detection
    read_buffer: Vec<u8>,
    /// Scroll offset for preview pane
    preview_scroll: usize,
    /// Git status for files
    git_statuses: HashMap<PathBuf, GitStatus>,
    /// Whether to show the preview pane
    show_preview: bool,
}

impl FileTree {
    pub fn new(root: PathBuf, editor: &Editor) -> Self {
        let git_statuses = Self::load_git_statuses(&root);
        let mut tree = Self {
            entries: Vec::new(),
            cursor: 0,
            scroll_offset: 0,
            root: root.clone(),
            expanded: HashSet::new(),
            input_mode: InputMode::Normal,
            input_buffer: String::new(),
            input_cursor: 0,
            search_results: Vec::new(),
            rename_target: None,
            message: None,
            preview_cache: HashMap::new(),
            read_buffer: Vec::with_capacity(1024),
            preview_scroll: 0,
            git_statuses,
            show_preview: true,
        };
        // Expand the root by default
        tree.expanded.insert(root);

        // Try to jump to the current file automatically
        tree.auto_reveal_current_file(editor);

        tree
    }

    /// Automatically reveal and select the current file in the tree (silent, no messages)
    fn auto_reveal_current_file(&mut self, editor: &Editor) {
        // Get the current view's document
        let view = editor.tree.get(editor.tree.focus);
        let doc = &editor.documents[&view.doc];

        if let Some(path) = doc.path() {
            let path = path.to_path_buf();

            // Check if the file is within our root directory
            if !path.starts_with(&self.root) {
                // File is outside the tree root, just build normally
                self.rebuild_entries(editor);
                return;
            }

            // Expand all parent directories to reveal the file
            let mut current = path.clone();
            while current != self.root {
                if let Some(parent) = current.parent() {
                    self.expanded.insert(parent.to_path_buf());
                    current = parent.to_path_buf();
                } else {
                    break;
                }
            }

            self.rebuild_entries(editor);

            // Find and select the file
            if let Some(pos) = self.entries.iter().position(|e| e.path == path) {
                self.cursor = pos;
            }
        } else {
            // No file path, just build the tree normally
            self.rebuild_entries(editor);
        }
    }

    /// Create a FileTree from saved state (for resume functionality)
    pub fn from_state(state: FileTreeState, editor: &Editor) -> Self {
        let git_statuses = Self::load_git_statuses(&state.root);
        let config = editor.config();
        let mut tree = Self {
            entries: Vec::new(),
            cursor: state.cursor,
            scroll_offset: state.scroll_offset,
            root: state.root.clone(),
            expanded: state.expanded,
            input_mode: match state.input_mode {
                SavedInputMode::Normal => InputMode::Normal,
                SavedInputMode::Search => InputMode::Search,
                SavedInputMode::GoToFolder => InputMode::GoToFolder,
            },
            input_buffer: state.input_buffer.clone(),
            input_cursor: state.input_buffer.len(),
            search_results: Vec::new(),
            rename_target: None,
            message: Some("Resumed".to_string()),
            preview_cache: HashMap::new(),
            read_buffer: Vec::with_capacity(1024),
            preview_scroll: state.preview_scroll,
            git_statuses,
            show_preview: state.show_preview,
        };

        // Rebuild entries
        tree.rebuild_entries(editor);

        // If we were in search mode, re-perform the search
        match tree.input_mode {
            InputMode::Search => tree.perform_search(&config.file_tree),
            InputMode::GoToFolder => tree.perform_folder_search(&config.file_tree),
            _ => {}
        }

        // Ensure cursor is still valid
        let len = match tree.input_mode {
            InputMode::Search | InputMode::GoToFolder => tree.search_results.len(),
            _ => tree.entries.len(),
        };
        if len > 0 && tree.cursor >= len {
            tree.cursor = len.saturating_sub(1);
        }

        tree
    }

    /// Save the current state for later resumption
    pub fn save_state(&self) -> FileTreeState {
        let saved_mode = match self.input_mode {
            InputMode::Search => SavedInputMode::Search,
            InputMode::GoToFolder => SavedInputMode::GoToFolder,
            // All other modes save as Normal since they're transient
            _ => SavedInputMode::Normal,
        };

        FileTreeState {
            root: self.root.clone(),
            expanded: self.expanded.clone(),
            cursor: self.cursor,
            scroll_offset: self.scroll_offset,
            input_mode: saved_mode,
            input_buffer: self.input_buffer.clone(),
            show_preview: self.show_preview,
            preview_scroll: self.preview_scroll,
        }
    }

    /// Load git status for all files in the repository
    fn load_git_statuses(root: &Path) -> HashMap<PathBuf, GitStatus> {
        let mut statuses = HashMap::new();

        // Run git status --porcelain to get file statuses
        let output = Command::new("git")
            .args(["status", "--porcelain", "-uall"])
            .current_dir(root)
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if line.len() < 4 {
                        continue;
                    }
                    let status_chars: Vec<char> = line.chars().take(2).collect();
                    let file_path = line[3..].trim();

                    // Handle renamed files (show as renamed)
                    let file_path = if let Some(pos) = file_path.find(" -> ") {
                        &file_path[pos + 4..]
                    } else {
                        file_path
                    };

                    let full_path = root.join(file_path);

                    let status = match (status_chars[0], status_chars[1]) {
                        ('?', '?') => GitStatus::Untracked,
                        ('!', '!') => GitStatus::Ignored,
                        ('A', _) | (_, 'A') => GitStatus::Added,
                        ('D', _) | (_, 'D') => GitStatus::Deleted,
                        ('R', _) => GitStatus::Renamed,
                        ('M', _) | (_, 'M') | ('U', _) | (_, 'U') => GitStatus::Modified,
                        _ => continue,
                    };

                    statuses.insert(full_path, status);
                }
            }
        }

        statuses
    }

    /// Refresh the tree and git statuses
    fn refresh(&mut self, editor: &Editor) {
        self.git_statuses = Self::load_git_statuses(&self.root);
        self.preview_cache.clear();
        self.rebuild_entries(editor);
        self.message = Some("Refreshed".to_string());
    }

    /// Jump to the currently open file in the editor
    fn jump_to_current_file(&mut self, editor: &Editor) {
        // Get the current view's document
        let view = editor.tree.get(editor.tree.focus);
        let doc = &editor.documents[&view.doc];
        if let Some(path) = doc.path() {
            let path = path.to_path_buf();
            // Expand all parent directories
            let mut current = path.clone();
            while current != self.root {
                if let Some(parent) = current.parent() {
                    self.expanded.insert(parent.to_path_buf());
                    current = parent.to_path_buf();
                } else {
                    break;
                }
            }

            self.rebuild_entries(editor);

            // Find and select the file
            if let Some(pos) = self.entries.iter().position(|e| e.path == path) {
                self.cursor = pos;
                self.preview_scroll = 0;
                self.message = Some(format!(
                    "Jumped to {}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                ));
            } else {
                self.message = Some("File not found in tree".to_string());
            }
        } else {
            self.message = Some("Current buffer has no file path".to_string());
        }
    }

    /// Copy the selected file's path to clipboard
    fn copy_path_to_clipboard(&self, cx: &mut Context) {
        let path = if self.input_mode == InputMode::Search
            || self.input_mode == InputMode::GoToFolder
        {
            self.selected_search_result().map(|r| r.path.clone())
        } else {
            self.selected_entry().map(|e| e.path.clone())
        };

        if let Some(path) = path {
            let path_str = path.to_string_lossy().to_string();
            // Use helix's clipboard register '+'
            match cx.editor.registers.write('+', vec![path_str.clone()]) {
                Ok(_) => cx.editor.set_status(format!("Copied: {}", path_str)),
                Err(e) => cx.editor.set_error(format!("Failed to copy: {}", e)),
            }
        }
    }

    /// Format file size for display
    fn format_file_size(size: u64) -> String {
        if size < 1024 {
            format!("{}B", size)
        } else if size < 1024 * 1024 {
            format!("{:.1}K", size as f64 / 1024.0)
        } else if size < 1024 * 1024 * 1024 {
            format!("{:.1}M", size as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.1}G", size as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }

    /// Rebuild the flattened entry list from the expanded state
    fn rebuild_entries(&mut self, editor: &Editor) {
        self.entries.clear();
        let config = editor.config();
        self.collect_entries(&self.root.clone(), 0, &config.file_tree);

        // Ensure cursor is still valid
        if self.cursor >= self.entries.len() {
            self.cursor = self.entries.len().saturating_sub(1);
        }
    }

    /// Recursively collect entries from a directory
    fn collect_entries(&mut self, dir: &Path, depth: usize, config: &FileTreeConfig) {
        use ignore::WalkBuilder;

        let mut walk_builder = WalkBuilder::new(dir);
        let entries: Vec<_> = walk_builder
            .hidden(config.hidden)
            .follow_links(config.follow_symlinks)
            .git_ignore(config.git_ignore)
            .ignore(config.ignore)
            .max_depth(Some(1))
            .sort_by_file_name(|a, b| a.cmp(b))
            .build()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path() != dir)
            .collect();

        // Separate directories and files, sort each group
        let mut dirs: Vec<_> = entries.iter().filter(|e| e.path().is_dir()).collect();
        let mut files: Vec<_> = entries.iter().filter(|e| !e.path().is_dir()).collect();

        dirs.sort_by(|a, b| a.path().cmp(b.path()));
        files.sort_by(|a, b| a.path().cmp(b.path()));

        // Add directories first, then files
        for entry in dirs {
            let path = entry.path().to_path_buf();
            let is_expanded = self.expanded.contains(&path);

            self.entries.push(TreeEntry {
                path: path.clone(),
                is_dir: true,
                depth,
            });

            if is_expanded {
                self.collect_entries(&path, depth + 1, config);
            }
        }

        for entry in files {
            self.entries.push(TreeEntry {
                path: entry.path().to_path_buf(),
                is_dir: false,
                depth,
            });
        }
    }

    /// Perform recursive fuzzy search (fzf-style)
    fn perform_search(&mut self, config: &FileTreeConfig) {
        use ignore::WalkBuilder;

        self.search_results.clear();

        if self.input_buffer.is_empty() {
            return;
        }

        let query = &self.input_buffer;

        let mut walk_builder = WalkBuilder::new(&self.root);
        let mut results: Vec<_> = walk_builder
            .hidden(config.hidden)
            .follow_links(config.follow_symlinks)
            .git_ignore(config.git_ignore)
            .ignore(config.ignore)
            .build()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path() != self.root)
            .filter_map(|entry| {
                let path = entry.path().to_path_buf();
                let is_dir = path.is_dir();
                let relative_path = path
                    .strip_prefix(&self.root)
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| path.to_string_lossy().to_string());

                // Fuzzy match against the relative path
                fuzzy_match(query, &relative_path).map(|m| SearchResult {
                    path,
                    is_dir,
                    relative_path,
                    score: m.score,
                })
            })
            .collect();

        // Sort by score (highest first), then by path length (shorter first)
        results.sort_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then_with(|| a.relative_path.len().cmp(&b.relative_path.len()))
        });

        // Limit results
        results.truncate(100);

        self.search_results = results;
        self.cursor = 0;
        self.scroll_offset = 0;
    }

    /// Perform recursive fuzzy search for folders only (fzf-style)
    fn perform_folder_search(&mut self, config: &FileTreeConfig) {
        use ignore::WalkBuilder;

        self.search_results.clear();

        if self.input_buffer.is_empty() {
            return;
        }

        let query = &self.input_buffer;

        let mut walk_builder = WalkBuilder::new(&self.root);
        let mut results: Vec<_> = walk_builder
            .hidden(config.hidden)
            .follow_links(config.follow_symlinks)
            .git_ignore(config.git_ignore)
            .ignore(config.ignore)
            .build()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path() != self.root)
            .filter(|entry| entry.path().is_dir()) // Only directories
            .filter_map(|entry| {
                let path = entry.path().to_path_buf();
                let relative_path = path
                    .strip_prefix(&self.root)
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| path.to_string_lossy().to_string());

                // Fuzzy match against the relative path
                fuzzy_match(query, &relative_path).map(|m| SearchResult {
                    path,
                    is_dir: true,
                    relative_path,
                    score: m.score,
                })
            })
            .collect();

        // Sort by score (highest first), then by path length (shorter first)
        results.sort_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then_with(|| a.relative_path.len().cmp(&b.relative_path.len()))
        });

        // Limit results
        results.truncate(100);

        self.search_results = results;
        self.cursor = 0;
        self.scroll_offset = 0;
    }

    fn move_cursor(&mut self, delta: isize) {
        let len = if self.input_mode == InputMode::Search
            || self.input_mode == InputMode::GoToFolder
        {
            self.search_results.len()
        } else {
            self.entries.len()
        };

        if len == 0 {
            return;
        }

        let new_cursor = if delta > 0 {
            (self.cursor + delta as usize).min(len - 1)
        } else {
            self.cursor.saturating_sub((-delta) as usize)
        };

        // Reset preview scroll when cursor moves
        if new_cursor != self.cursor {
            self.preview_scroll = 0;
        }

        self.cursor = new_cursor;
    }

    fn selected_entry(&self) -> Option<&TreeEntry> {
        self.entries.get(self.cursor)
    }

    fn selected_search_result(&self) -> Option<&SearchResult> {
        self.search_results.get(self.cursor)
    }

    fn toggle_expand(&mut self, editor: &Editor) {
        if let Some(entry) = self.entries.get(self.cursor).cloned() {
            if entry.is_dir {
                if self.expanded.contains(&entry.path) {
                    self.expanded.remove(&entry.path);
                } else {
                    self.expanded.insert(entry.path);
                }
                self.rebuild_entries(editor);
            }
        }
    }

    fn collapse_or_go_parent(&mut self, editor: &Editor) {
        if let Some(entry) = self.entries.get(self.cursor).cloned() {
            if entry.is_dir && self.expanded.contains(&entry.path) {
                // Collapse the directory
                self.expanded.remove(&entry.path);
                self.rebuild_entries(editor);
            } else if let Some(parent) = entry.path.parent() {
                // Go to parent directory
                if parent != self.root {
                    if let Some(pos) = self.entries.iter().position(|e| e.path == parent) {
                        self.cursor = pos;
                    }
                }
            }
        }
    }

    fn get_icon<'a>(&self, entry: &TreeEntry, config: &'a FileTreeConfig) -> &'a str {
        if !config.show_icons {
            return "";
        }

        if entry.is_dir {
            if self.expanded.contains(&entry.path) {
                return &config.icons.directory_open;
            } else {
                return &config.icons.directory;
            }
        }

        // Check filename first
        if let Some(filename) = entry.path.file_name().and_then(|n| n.to_str()) {
            if let Some(icon) = config.icons.filenames.get(filename) {
                return icon;
            }
        }

        // Check extension
        if let Some(ext) = entry.path.extension().and_then(|e| e.to_str()) {
            if let Some(icon) = config.icons.extensions.get(ext) {
                return icon;
            }
        }

        &config.icons.file
    }

    fn get_search_icon<'a>(&self, result: &SearchResult, config: &'a FileTreeConfig) -> &'a str {
        if !config.show_icons {
            return "";
        }

        if result.is_dir {
            return &config.icons.directory;
        }

        // Check filename first
        if let Some(filename) = result.path.file_name().and_then(|n| n.to_str()) {
            if let Some(icon) = config.icons.filenames.get(filename) {
                return icon;
            }
        }

        // Check extension
        if let Some(ext) = result.path.extension().and_then(|e| e.to_str()) {
            if let Some(icon) = config.icons.extensions.get(ext) {
                return icon;
            }
        }

        &config.icons.file
    }

    fn adjust_scroll(&mut self, viewport_height: usize) {
        if viewport_height == 0 {
            return;
        }

        // Ensure cursor is visible
        if self.cursor < self.scroll_offset {
            self.scroll_offset = self.cursor;
        } else if self.cursor >= self.scroll_offset + viewport_height {
            self.scroll_offset = self.cursor - viewport_height + 1;
        }
    }

    fn enter_search_mode(&mut self) {
        self.input_mode = InputMode::Search;
        self.input_buffer.clear();
        self.search_results.clear();
        self.input_cursor = 0;
        self.cursor = 0;
        self.scroll_offset = 0;
        self.message = None;
    }

    fn enter_goto_folder_mode(&mut self) {
        self.input_mode = InputMode::GoToFolder;
        self.input_buffer.clear();
        self.search_results.clear();
        self.input_cursor = 0;
        self.cursor = 0;
        self.scroll_offset = 0;
        self.message = None;
    }

    fn enter_add_mode(&mut self) {
        self.input_mode = InputMode::Add;
        self.input_buffer.clear();
        self.input_cursor = 0;
        self.message = None;
    }

    fn enter_rename_mode(&mut self) {
        let entry_info = self.selected_entry().map(|entry| {
            (
                entry.path.clone(),
                entry
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default(),
            )
        });
        if let Some((path, name)) = entry_info {
            self.rename_target = Some(path);
            self.input_buffer = name;
            self.input_cursor = self.input_buffer.len();
            self.input_mode = InputMode::Rename;
            self.message = None;
        }
    }

    fn enter_delete_mode(&mut self) {
        if self.selected_entry().is_some() {
            self.input_mode = InputMode::ConfirmDelete;
            self.message = None;
        }
    }

    fn exit_input_mode(&mut self) {
        self.input_mode = InputMode::Normal;
        self.input_buffer.clear();
        self.search_results.clear();
        self.input_cursor = 0;
        self.rename_target = None;
        self.cursor = 0;
        self.scroll_offset = 0;
    }

    fn handle_input_char(&mut self, c: char, config: &FileTreeConfig) {
        self.input_buffer.insert(self.input_cursor, c);
        self.input_cursor += 1;
        match self.input_mode {
            InputMode::Search => self.perform_search(config),
            InputMode::GoToFolder => self.perform_folder_search(config),
            _ => {}
        }
    }

    fn handle_input_backspace(&mut self, config: &FileTreeConfig) {
        if self.input_cursor > 0 {
            self.input_cursor -= 1;
            self.input_buffer.remove(self.input_cursor);
            match self.input_mode {
                InputMode::Search => self.perform_search(config),
                InputMode::GoToFolder => self.perform_folder_search(config),
                _ => {}
            }
        }
    }

    /// Delete word backwards (Ctrl+W behavior)
    fn handle_input_delete_word(&mut self, config: &FileTreeConfig) {
        if self.input_cursor == 0 {
            return;
        }

        // Find the start of the word to delete
        let chars: Vec<char> = self.input_buffer.chars().collect();
        let mut new_cursor = self.input_cursor;

        // Skip any trailing whitespace/separators
        while new_cursor > 0 {
            let c = chars[new_cursor - 1];
            if c.is_alphanumeric() || c == '_' {
                break;
            }
            new_cursor -= 1;
        }

        // Delete the word characters
        while new_cursor > 0 {
            let c = chars[new_cursor - 1];
            if !c.is_alphanumeric() && c != '_' {
                break;
            }
            new_cursor -= 1;
        }

        // Remove characters from new_cursor to input_cursor
        if new_cursor < self.input_cursor {
            self.input_buffer = chars[..new_cursor]
                .iter()
                .chain(chars[self.input_cursor..].iter())
                .collect();
            self.input_cursor = new_cursor;

            match self.input_mode {
                InputMode::Search => self.perform_search(config),
                InputMode::GoToFolder => self.perform_folder_search(config),
                _ => {}
            }
        }
    }

    /// Paste text from clipboard
    fn handle_input_paste(&mut self, cx: &mut Context, config: &FileTreeConfig) {
        // Read from the '+' register (system clipboard)
        if let Some(contents) = cx.editor.registers.read('+', cx.editor) {
            let text: String = contents
                .into_iter()
                .flat_map(|s| s.chars().filter(|c| *c != '\n' && *c != '\r').collect::<Vec<_>>())
                .collect();

            if !text.is_empty() {
                // Insert at cursor position
                self.input_buffer.insert_str(self.input_cursor, &text);
                self.input_cursor += text.len();

                match self.input_mode {
                    InputMode::Search => self.perform_search(config),
                    InputMode::GoToFolder => self.perform_folder_search(config),
                    _ => {}
                }
            }
        }
    }

    /// Clear the entire input buffer (Ctrl+U behavior)
    fn handle_input_clear(&mut self, config: &FileTreeConfig) {
        self.input_buffer.clear();
        self.input_cursor = 0;

        match self.input_mode {
            InputMode::Search => self.perform_search(config),
            InputMode::GoToFolder => self.perform_folder_search(config),
            _ => {}
        }
    }

    /// Get the directory where new files should be created
    fn get_target_directory(&self) -> PathBuf {
        if let Some(entry) = self.selected_entry() {
            if entry.is_dir {
                entry.path.clone()
            } else {
                entry
                    .path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| self.root.clone())
            }
        } else {
            self.root.clone()
        }
    }

    /// Create a new file or directory
    fn create_entry(&mut self, editor: &Editor) -> Result<PathBuf, String> {
        let name = self.input_buffer.trim();
        if name.is_empty() {
            return Err("Name cannot be empty".to_string());
        }

        let target_dir = self.get_target_directory();
        let new_path = target_dir.join(name);

        // Check if already exists
        if new_path.exists() {
            return Err(format!("'{}' already exists", name));
        }

        // Create parent directories if needed
        if let Some(parent) = new_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create parent directories: {}", e))?;
            }
        }

        // If name ends with /, create directory; otherwise create file
        if name.ends_with('/') || name.ends_with(std::path::MAIN_SEPARATOR) {
            fs::create_dir_all(&new_path)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        } else {
            fs::File::create(&new_path).map_err(|e| format!("Failed to create file: {}", e))?;
        }

        // Expand the target directory to show the new entry
        self.expanded.insert(target_dir);
        self.rebuild_entries(editor);

        // Try to select the new entry
        if let Some(pos) = self.entries.iter().position(|e| e.path == new_path) {
            self.cursor = pos;
        }

        Ok(new_path)
    }

    /// Rename a file or directory
    fn rename_entry(&mut self, editor: &Editor) -> Result<PathBuf, String> {
        let new_name = self.input_buffer.trim();
        if new_name.is_empty() {
            return Err("Name cannot be empty".to_string());
        }

        let old_path = self
            .rename_target
            .as_ref()
            .ok_or_else(|| "No file selected for rename".to_string())?;

        let parent = old_path
            .parent()
            .ok_or_else(|| "Cannot rename root".to_string())?;

        let new_path = parent.join(new_name);

        // Check if target already exists (and is not the same file)
        if new_path.exists() && new_path != *old_path {
            return Err(format!("'{}' already exists", new_name));
        }

        fs::rename(old_path, &new_path).map_err(|e| format!("Failed to rename: {}", e))?;

        // Update expanded set if we renamed a directory
        if self.expanded.contains(old_path) {
            self.expanded.remove(old_path);
            self.expanded.insert(new_path.clone());
        }

        self.rebuild_entries(editor);

        // Try to select the renamed entry
        if let Some(pos) = self.entries.iter().position(|e| e.path == new_path) {
            self.cursor = pos;
        }

        Ok(new_path)
    }

    /// Delete a file or directory
    fn delete_entry(&mut self, editor: &Editor) -> Result<(), String> {
        let entry = self
            .selected_entry()
            .ok_or_else(|| "No file selected".to_string())?
            .clone();

        if entry.is_dir {
            fs::remove_dir_all(&entry.path)
                .map_err(|e| format!("Failed to delete directory: {}", e))?;
            self.expanded.remove(&entry.path);
        } else {
            fs::remove_file(&entry.path).map_err(|e| format!("Failed to delete file: {}", e))?;
        }

        self.rebuild_entries(editor);

        Ok(())
    }

    /// Get the preview for the currently selected entry
    fn get_preview<'a>(&'a mut self, editor: &'a Editor) -> Option<&'a CachedPreview> {
        let path = if self.input_mode == InputMode::Search
            || self.input_mode == InputMode::GoToFolder
        {
            self.selected_search_result().map(|r| r.path.clone())?
        } else {
            self.selected_entry().map(|e| e.path.clone())?
        };

        // Check if already open in editor
        if let Some(_doc) = editor.document_by_path(&path) {
            // For open documents, we still cache them for simplicity
            // but we could return the editor's document directly
        }

        if self.preview_cache.contains_key(path.as_path()) {
            return self.preview_cache.get(path.as_path());
        }

        let arc_path: Arc<Path> = path.clone().into();
        let preview = std::fs::metadata(&path)
            .and_then(|metadata| {
                if metadata.is_dir() {
                    // Get directory contents
                    let mut entries: Vec<(String, bool)> = Vec::new();
                    for entry in std::fs::read_dir(&path)? {
                        let entry = entry?;
                        let name = entry.file_name().to_string_lossy().to_string();
                        let is_dir = entry.file_type()?.is_dir();
                        if is_dir {
                            entries.push((format!("{}/", name), true));
                        } else {
                            entries.push((name, false));
                        }
                    }
                    entries.sort_by(|a, b| {
                        // Directories first, then alphabetical
                        match (a.1, b.1) {
                            (true, false) => std::cmp::Ordering::Less,
                            (false, true) => std::cmp::Ordering::Greater,
                            _ => a.0.cmp(&b.0),
                        }
                    });
                    Ok(CachedPreview::Directory(entries))
                } else if metadata.is_file() {
                    if metadata.len() > MAX_FILE_SIZE_FOR_PREVIEW {
                        return Ok(CachedPreview::LargeFile);
                    }
                    // Check if binary
                    let content_type = std::fs::File::open(&path).and_then(|file| {
                        self.read_buffer.clear();
                        let n = file.take(1024).read_to_end(&mut self.read_buffer)?;
                        let content_type = content_inspector::inspect(&self.read_buffer[..n]);
                        Ok(content_type)
                    })?;
                    if content_type.is_binary() {
                        return Ok(CachedPreview::Binary);
                    }
                    // Open document for preview
                    let doc = Document::open(
                        &path,
                        None,
                        false,
                        editor.config.clone(),
                        editor.syn_loader.clone(),
                    )
                    .map_err(|_| {
                        std::io::Error::new(std::io::ErrorKind::NotFound, "Cannot open document")
                    })?;
                    Ok(CachedPreview::Document(Box::new(doc)))
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "Not a file or directory",
                    ))
                }
            })
            .unwrap_or(CachedPreview::NotFound);

        self.preview_cache.insert(arc_path, preview);
        self.preview_cache.get(path.as_path())
    }

    /// Render the preview pane
    fn render_preview(&mut self, area: Rect, surface: &mut Surface, cx: &mut Context) {
        let background = cx.editor.theme.get("ui.background");
        let text = cx.editor.theme.get("ui.text");
        let directory_style = cx.editor.theme.get("ui.text.directory");
        surface.clear_with(area, background);

        // Get preview title from selected item
        let title = if self.input_mode == InputMode::Search
            || self.input_mode == InputMode::GoToFolder
        {
            self.selected_search_result()
                .map(|r| {
                    format!(
                        " {} ",
                        r.path.file_name().unwrap_or_default().to_string_lossy()
                    )
                })
                .unwrap_or_else(|| " Preview ".to_string())
        } else {
            self.selected_entry()
                .map(|e| {
                    format!(
                        " {} ",
                        e.path.file_name().unwrap_or_default().to_string_lossy()
                    )
                })
                .unwrap_or_else(|| " Preview ".to_string())
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(title, text));
        let inner = block.inner(area);
        let margin = Margin::horizontal(1);
        let inner = inner.inner(margin);
        block.render(area, surface);

        // Get scroll value before borrowing self for preview
        let scroll = self.preview_scroll;

        if let Some(preview) = self.get_preview(cx.editor) {
            match preview {
                CachedPreview::Document(doc) => {
                    // Apply scroll offset
                    let total_lines = doc.text().len_lines();
                    let max_scroll = total_lines.saturating_sub(inner.height as usize);
                    let actual_scroll = scroll.min(max_scroll);

                    let mut offset = ViewPosition::default();
                    if actual_scroll > 0 {
                        let text_slice = doc.text().slice(..);
                        offset.anchor = text_slice.line_to_char(actual_scroll);
                    }

                    let loader = cx.editor.syn_loader.load();

                    let syntax_highlighter =
                        EditorView::doc_syntax_highlighter(doc, offset.anchor, area.height, &loader);

                    let decorations = DecorationManager::default();

                    render_document(
                        surface,
                        inner,
                        doc,
                        offset,
                        &TextAnnotations::default(),
                        syntax_highlighter,
                        Vec::new(),
                        &cx.editor.theme,
                        decorations,
                    );
                }
                CachedPreview::Directory(entries) => {
                    let start = scroll.min(entries.len().saturating_sub(1));
                    let visible = entries.iter().skip(start).take(inner.height as usize);
                    for (i, (name, is_dir)) in visible.enumerate() {
                        let style = if *is_dir { directory_style } else { text };
                        surface.set_stringn(
                            inner.x,
                            inner.y + i as u16,
                            name,
                            inner.width as usize,
                            style,
                        );
                    }
                }
                _ => {
                    let placeholder = preview.placeholder();
                    let x = inner.x + inner.width.saturating_sub(placeholder.len() as u16) / 2;
                    let y = inner.y + inner.height / 2;
                    surface.set_stringn(x, y, placeholder, inner.width as usize, text);
                }
            }
        }
    }

    /// Render help overlay
    fn render_help(&self, area: Rect, surface: &mut Surface, cx: &mut Context) {
        let theme = &cx.editor.theme;
        let text_style = theme.get("ui.text");
        let title_style = theme.get("ui.text").add_modifier(Modifier::BOLD);

        surface.clear_with(area, theme.get("ui.background"));

        let block = Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(" Help ", title_style));
        let inner = block.inner(area);
        block.render(area, surface);

        let help_lines = [
            ("Navigation", ""),
            ("  j/k, ↑/↓", "Move cursor"),
            ("  h/l, ←/→", "Collapse/Expand"),
            ("  Enter", "Open file / Toggle dir"),
            ("  g", "Go to top"),
            ("  G", "Go to bottom"),
            ("  Ctrl+d/u", "Page down/up"),
            ("", ""),
            ("Search & Jump", ""),
            ("  /", "Search files (fuzzy)"),
            ("  Ctrl+g", "Go to folder"),
            ("  f", "Jump to current file"),
            ("", ""),
            ("Input (in search/add/rename)", ""),
            ("  Ctrl/Cmd+w", "Delete word"),
            ("  Ctrl/Cmd+u", "Clear line"),
            ("  Ctrl/Cmd+v", "Paste"),
            ("", ""),
            ("File Operations", ""),
            ("  a", "Add file/directory"),
            ("  d", "Delete"),
            ("  r", "Rename"),
            ("  y", "Copy path"),
            ("", ""),
            ("Preview & Other", ""),
            ("  p", "Toggle preview"),
            ("  Ctrl+j/k", "Scroll preview"),
            ("  R", "Refresh tree"),
            ("  Ctrl+s", "Open in h-split"),
            ("  ?", "Toggle help"),
            ("  q/Esc", "Close (use :ftr to resume)"),
        ];

        for (i, (key, desc)) in help_lines.iter().take(inner.height as usize).enumerate() {
            if key.is_empty() {
                continue;
            }
            let style = if desc.is_empty() { title_style } else { text_style };
            let line = if desc.is_empty() {
                key.to_string()
            } else {
                format!("{:<14} {}", key, desc)
            };
            surface.set_stringn(inner.x + 1, inner.y + i as u16, &line, inner.width as usize - 2, style);
        }
    }
}

impl Component for FileTree {
    fn handle_event(&mut self, event: &Event, cx: &mut Context) -> EventResult {
        let event = match event {
            Event::Key(event) => *event,
            _ => return EventResult::Ignored(None),
        };

        // Save state before closing
        let state_to_save = self.save_state();
        let close_fn: crate::compositor::Callback =
            Box::new(move |compositor: &mut Compositor, _cx: &mut Context| {
                save_last_state(state_to_save.clone());
                compositor.remove(ID);
            });

        let config = cx.editor.config();

        // Clear message on any key press
        self.message = None;

        match self.input_mode {
            InputMode::Search => {
                match event {
                    key!(Esc) => {
                        self.exit_input_mode();
                        return EventResult::Consumed(None);
                    }
                    key!(Enter) => {
                        if let Some(result) = self.selected_search_result().cloned() {
                            if result.is_dir {
                                // Expand to this directory and exit search
                                self.exit_input_mode();
                                // Expand all parent directories
                                let mut current = result.path.clone();
                                while current != self.root {
                                    self.expanded.insert(current.clone());
                                    if let Some(parent) = current.parent() {
                                        current = parent.to_path_buf();
                                    } else {
                                        break;
                                    }
                                }
                                self.rebuild_entries(cx.editor);
                                // Find and select the target
                                if let Some(pos) =
                                    self.entries.iter().position(|e| e.path == result.path)
                                {
                                    self.cursor = pos;
                                }
                            } else {
                                // Open file
                                let path = result.path.clone();
                                if let Err(e) = cx.editor.open(&path, Action::Replace) {
                                    cx.editor.set_error(format!("Failed to open file: {}", e));
                                }
                                return EventResult::Consumed(Some(close_fn));
                            }
                        }
                        return EventResult::Consumed(None);
                    }
                    key!(Backspace) => {
                        self.handle_input_backspace(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    // Delete word backwards (Ctrl+W or Cmd+Backspace)
                    _ if is_delete_word_event(&event) => {
                        self.handle_input_delete_word(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    // Clear entire line (Ctrl+U or Cmd+U)
                    _ if is_clear_line_event(&event) => {
                        self.handle_input_clear(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    // Paste from clipboard (Ctrl+V or Cmd+V)
                    _ if is_paste_event(&event) => {
                        self.handle_input_paste(cx, &config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    key!(Down) | ctrl!('n') => {
                        self.move_cursor(1);
                        return EventResult::Consumed(None);
                    }
                    key!(Up) | ctrl!('p') => {
                        self.move_cursor(-1);
                        return EventResult::Consumed(None);
                    }
                    ctrl!('s') => {
                        if let Some(result) = self.selected_search_result() {
                            if !result.is_dir {
                                let path = result.path.clone();
                                if let Err(e) = cx.editor.open(&path, Action::HorizontalSplit) {
                                    cx.editor.set_error(format!("Failed to open file: {}", e));
                                }
                                return EventResult::Consumed(Some(close_fn));
                            }
                        }
                        return EventResult::Consumed(None);
                    }
                    _ => {
                        // Handle character input for search
                        if let KeyCode::Char(c) = event.code {
                            if event.modifiers.is_empty() {
                                self.handle_input_char(c, &config.file_tree);
                                return EventResult::Consumed(None);
                            }
                        }
                    }
                }
                return EventResult::Consumed(None);
            }

            InputMode::GoToFolder => {
                match event {
                    key!(Esc) => {
                        self.exit_input_mode();
                        return EventResult::Consumed(None);
                    }
                    key!(Enter) => {
                        if let Some(result) = self.selected_search_result().cloned() {
                            // Navigate to this folder
                            self.exit_input_mode();
                            // Expand all parent directories including the target
                            let mut current = result.path.clone();
                            while current != self.root {
                                self.expanded.insert(current.clone());
                                if let Some(parent) = current.parent() {
                                    current = parent.to_path_buf();
                                } else {
                                    break;
                                }
                            }
                            self.rebuild_entries(cx.editor);
                            // Find and select the target
                            if let Some(pos) =
                                self.entries.iter().position(|e| e.path == result.path)
                            {
                                self.cursor = pos;
                            }
                        }
                        return EventResult::Consumed(None);
                    }
                    key!(Backspace) => {
                        self.handle_input_backspace(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    // Delete word backwards (Ctrl+W or Cmd+Backspace)
                    _ if is_delete_word_event(&event) => {
                        self.handle_input_delete_word(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    // Clear entire line (Ctrl+U or Cmd+U)
                    _ if is_clear_line_event(&event) => {
                        self.handle_input_clear(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    // Paste from clipboard (Ctrl+V or Cmd+V)
                    _ if is_paste_event(&event) => {
                        self.handle_input_paste(cx, &config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    key!(Down) | ctrl!('n') => {
                        self.move_cursor(1);
                        return EventResult::Consumed(None);
                    }
                    key!(Up) | ctrl!('p') => {
                        self.move_cursor(-1);
                        return EventResult::Consumed(None);
                    }
                    _ => {
                        // Handle character input for folder search
                        if let KeyCode::Char(c) = event.code {
                            if event.modifiers.is_empty() {
                                self.handle_input_char(c, &config.file_tree);
                                return EventResult::Consumed(None);
                            }
                        }
                    }
                }
                return EventResult::Consumed(None);
            }

            InputMode::Add => {
                match event {
                    key!(Esc) => {
                        self.exit_input_mode();
                        return EventResult::Consumed(None);
                    }
                    key!(Enter) => {
                        match self.create_entry(cx.editor) {
                            Ok(path) => {
                                let name = path.file_name().unwrap_or_default().to_string_lossy();
                                self.message = Some(format!("Created: {}", name));
                                self.input_mode = InputMode::Normal;
                                self.input_buffer.clear();
                            }
                            Err(e) => {
                                self.message = Some(e);
                            }
                        }
                        return EventResult::Consumed(None);
                    }
                    key!(Backspace) => {
                        self.handle_input_backspace(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    _ if is_delete_word_event(&event) => {
                        self.handle_input_delete_word(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    _ if is_clear_line_event(&event) => {
                        self.handle_input_clear(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    _ if is_paste_event(&event) => {
                        self.handle_input_paste(cx, &config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    _ => {
                        if let KeyCode::Char(c) = event.code {
                            if event.modifiers.is_empty() {
                                self.handle_input_char(c, &config.file_tree);
                                return EventResult::Consumed(None);
                            }
                        }
                    }
                }
                return EventResult::Consumed(None);
            }

            InputMode::Rename => {
                match event {
                    key!(Esc) => {
                        self.exit_input_mode();
                        return EventResult::Consumed(None);
                    }
                    key!(Enter) => {
                        match self.rename_entry(cx.editor) {
                            Ok(path) => {
                                let name = path.file_name().unwrap_or_default().to_string_lossy();
                                self.message = Some(format!("Renamed to: {}", name));
                                self.input_mode = InputMode::Normal;
                                self.input_buffer.clear();
                                self.rename_target = None;
                            }
                            Err(e) => {
                                self.message = Some(e);
                            }
                        }
                        return EventResult::Consumed(None);
                    }
                    key!(Backspace) => {
                        self.handle_input_backspace(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    _ if is_delete_word_event(&event) => {
                        self.handle_input_delete_word(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    _ if is_clear_line_event(&event) => {
                        self.handle_input_clear(&config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    _ if is_paste_event(&event) => {
                        self.handle_input_paste(cx, &config.file_tree);
                        return EventResult::Consumed(None);
                    }
                    _ => {
                        if let KeyCode::Char(c) = event.code {
                            if event.modifiers.is_empty() {
                                self.handle_input_char(c, &config.file_tree);
                                return EventResult::Consumed(None);
                            }
                        }
                    }
                }
                return EventResult::Consumed(None);
            }

            InputMode::ConfirmDelete => match event {
                key!(Esc) | key!('n') | key!('N') => {
                    self.input_mode = InputMode::Normal;
                    self.message = Some("Delete cancelled".to_string());
                    return EventResult::Consumed(None);
                }
                key!('y') | key!('Y') | key!(Enter) => {
                    if let Some(entry) = self.selected_entry() {
                        let name = entry
                            .path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        match self.delete_entry(cx.editor) {
                            Ok(()) => {
                                self.message = Some(format!("Deleted: {}", name));
                            }
                            Err(e) => {
                                self.message = Some(e);
                            }
                        }
                    }
                    self.input_mode = InputMode::Normal;
                    return EventResult::Consumed(None);
                }
                _ => {
                    return EventResult::Consumed(None);
                }
            },

            InputMode::Normal => {
                match event {
                    // Close
                    key!(Esc) | key!('q') => {
                        return EventResult::Consumed(Some(close_fn));
                    }
                    // Search
                    key!('/') => {
                        self.enter_search_mode();
                        return EventResult::Consumed(None);
                    }
                    // Go to folder
                    ctrl!('g') => {
                        self.enter_goto_folder_mode();
                        return EventResult::Consumed(None);
                    }
                    // Add file/directory
                    key!('a') => {
                        self.enter_add_mode();
                        return EventResult::Consumed(None);
                    }
                    // Delete
                    key!('d') => {
                        self.enter_delete_mode();
                        return EventResult::Consumed(None);
                    }
                    // Rename
                    key!('r') => {
                        self.enter_rename_mode();
                        return EventResult::Consumed(None);
                    }
                    // Navigation
                    key!('j') | key!(Down) => {
                        self.move_cursor(1);
                        return EventResult::Consumed(None);
                    }
                    key!('k') | key!(Up) => {
                        self.move_cursor(-1);
                        return EventResult::Consumed(None);
                    }
                    key!('g') => {
                        self.cursor = 0;
                        self.scroll_offset = 0;
                        return EventResult::Consumed(None);
                    }
                    key!('G') => {
                        if !self.entries.is_empty() {
                            self.cursor = self.entries.len() - 1;
                        }
                        return EventResult::Consumed(None);
                    }
                    ctrl!('d') | key!(PageDown) => {
                        self.move_cursor(10);
                        return EventResult::Consumed(None);
                    }
                    ctrl!('u') | key!(PageUp) => {
                        self.move_cursor(-10);
                        return EventResult::Consumed(None);
                    }
                    // Expand/collapse
                    key!('h') | key!(Left) => {
                        self.collapse_or_go_parent(cx.editor);
                        return EventResult::Consumed(None);
                    }
                    key!('l') | key!(Right) => {
                        if let Some(entry) = self.selected_entry() {
                            if entry.is_dir && !self.expanded.contains(&entry.path) {
                                self.toggle_expand(cx.editor);
                                return EventResult::Consumed(None);
                            }
                        }
                        return EventResult::Consumed(None);
                    }
                    // Open/toggle
                    key!(Enter) => {
                        if let Some(entry) = self.selected_entry().cloned() {
                            if entry.is_dir {
                                self.toggle_expand(cx.editor);
                            } else {
                                // Open file
                                let path = entry.path.clone();
                                if let Err(e) = cx.editor.open(&path, Action::Replace) {
                                    cx.editor.set_error(format!("Failed to open file: {}", e));
                                }
                                return EventResult::Consumed(Some(close_fn));
                            }
                        }
                        return EventResult::Consumed(None);
                    }
                    // Open in horizontal split
                    ctrl!('s') => {
                        if let Some(entry) = self.selected_entry() {
                            if !entry.is_dir {
                                let path = entry.path.clone();
                                if let Err(e) = cx.editor.open(&path, Action::HorizontalSplit) {
                                    cx.editor.set_error(format!("Failed to open file: {}", e));
                                }
                                return EventResult::Consumed(Some(close_fn));
                            }
                        }
                        return EventResult::Consumed(None);
                    }
                    // Open in vertical split
                    ctrl!('v') => {
                        if let Some(entry) = self.selected_entry() {
                            if !entry.is_dir {
                                let path = entry.path.clone();
                                if let Err(e) = cx.editor.open(&path, Action::VerticalSplit) {
                                    cx.editor.set_error(format!("Failed to open file: {}", e));
                                }
                                return EventResult::Consumed(Some(close_fn));
                            }
                        }
                        return EventResult::Consumed(None);
                    }
                    // Refresh tree
                    key!('R') => {
                        self.refresh(cx.editor);
                        return EventResult::Consumed(None);
                    }
                    // Copy path to clipboard
                    key!('y') => {
                        self.copy_path_to_clipboard(cx);
                        return EventResult::Consumed(None);
                    }
                    // Toggle preview
                    key!('p') => {
                        self.show_preview = !self.show_preview;
                        return EventResult::Consumed(None);
                    }
                    // Show help
                    key!('?') => {
                        self.input_mode = InputMode::Help;
                        return EventResult::Consumed(None);
                    }
                    // Jump to current file (gf - but we use 'g' for go to top, so use 'f' alone)
                    key!('f') => {
                        self.jump_to_current_file(cx.editor);
                        return EventResult::Consumed(None);
                    }
                    // Scroll preview down
                    ctrl!('j') => {
                        self.preview_scroll = self.preview_scroll.saturating_add(3);
                        return EventResult::Consumed(None);
                    }
                    // Scroll preview up
                    ctrl!('k') => {
                        self.preview_scroll = self.preview_scroll.saturating_sub(3);
                        return EventResult::Consumed(None);
                    }
                    _ => {}
                }
            }

            InputMode::Help => {
                // Any key exits help
                self.input_mode = InputMode::Normal;
                return EventResult::Consumed(None);
            }
        }

        EventResult::Ignored(None)
    }

    fn render(&mut self, area: Rect, surface: &mut Surface, cx: &mut Context) {
        let config = cx.editor.config();
        let theme = &cx.editor.theme;

        // Handle Help mode overlay
        if self.input_mode == InputMode::Help {
            self.render_help(area, surface, cx);
            return;
        }

        // Determine if we should show the preview pane
        let show_preview = self.show_preview && area.width > MIN_AREA_WIDTH_FOR_PREVIEW;

        // Split the area for tree and preview
        let (tree_area, preview_area) = if show_preview {
            let tree_width = area.width / 2;
            let tree_area = Rect::new(area.x, area.y, tree_width, area.height);
            let preview_area = Rect::new(area.x + tree_width, area.y, area.width - tree_width, area.height);
            (tree_area, Some(preview_area))
        } else {
            (area, None)
        };

        // Styles
        let text_style = theme.get("ui.text");
        let selected_style = theme.get("ui.menu.selected");
        let directory_style = theme
            .try_get("ui.text.directory")
            .unwrap_or_else(|| theme.get("ui.text"));
        let prompt_style = theme
            .try_get("ui.text.focus")
            .unwrap_or_else(|| theme.get("ui.text"));
        let info_style = theme
            .try_get("ui.text.info")
            .unwrap_or_else(|| theme.get("ui.text"));

        // Draw border with title for tree area
        let title = match self.input_mode {
            InputMode::Search => " Search ".to_string(),
            InputMode::GoToFolder => " Go to Folder ".to_string(),
            InputMode::Add => " Add (end with / for dir) ".to_string(),
            InputMode::Rename => " Rename ".to_string(),
            InputMode::ConfirmDelete => " Delete? (y/n) ".to_string(),
            InputMode::Help => " Help ".to_string(),
            InputMode::Normal => format!(" {} ", self.root.display()),
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(title, text_style));
        let inner = block.inner(tree_area);
        block.render(tree_area, surface);

        // Clear inner area
        surface.clear_with(inner, text_style);

        match self.input_mode {
            InputMode::Search => {
                // Render search mode
                let search_prompt = format!("/{}", self.input_buffer);
                surface.set_stringn(
                    inner.x,
                    inner.y,
                    &search_prompt,
                    inner.width as usize,
                    prompt_style,
                );

                let results_area_y = inner.y + 1;
                let results_height = inner.height.saturating_sub(1) as usize;

                if self.search_results.is_empty() {
                    let msg = if self.input_buffer.is_empty() {
                        "Type to search..."
                    } else {
                        "No results found"
                    };
                    surface.set_stringn(
                        inner.x,
                        results_area_y,
                        msg,
                        inner.width as usize,
                        text_style,
                    );
                    return;
                }

                self.adjust_scroll(results_height);

                let start = self.scroll_offset;
                let end = (start + results_height).min(self.search_results.len());

                for (i, result) in self.search_results[start..end].iter().enumerate() {
                    let y = results_area_y + i as u16;
                    let is_selected = start + i == self.cursor;

                    let icon = self.get_search_icon(result, &config.file_tree);
                    let prefix = if is_selected { "> " } else { "  " };
                    let icon_space = if !icon.is_empty() { " " } else { "" };

                    let line = format!("{}{}{}{}", prefix, icon, icon_space, result.relative_path);

                    let style = if is_selected {
                        selected_style
                    } else if result.is_dir {
                        directory_style
                    } else {
                        text_style
                    };

                    surface.set_stringn(inner.x, y, &line, inner.width as usize, style);
                }

                // Scrollbar
                if self.search_results.len() > results_height {
                    self.render_scrollbar(
                        surface,
                        inner.x + inner.width.saturating_sub(1),
                        results_area_y,
                        results_height,
                        self.search_results.len(),
                        theme,
                    );
                }
            }

            InputMode::GoToFolder => {
                // Render go-to-folder mode (folders only)
                let search_prompt = format!("Go: {}", self.input_buffer);
                surface.set_stringn(
                    inner.x,
                    inner.y,
                    &search_prompt,
                    inner.width as usize,
                    prompt_style,
                );

                let results_area_y = inner.y + 1;
                let results_height = inner.height.saturating_sub(1) as usize;

                if self.search_results.is_empty() {
                    let msg = if self.input_buffer.is_empty() {
                        "Type to search folders..."
                    } else {
                        "No folders found"
                    };
                    surface.set_stringn(
                        inner.x,
                        results_area_y,
                        msg,
                        inner.width as usize,
                        text_style,
                    );
                    return;
                }

                self.adjust_scroll(results_height);

                let start = self.scroll_offset;
                let end = (start + results_height).min(self.search_results.len());

                for (i, result) in self.search_results[start..end].iter().enumerate() {
                    let y = results_area_y + i as u16;
                    let is_selected = start + i == self.cursor;

                    let icon = self.get_search_icon(result, &config.file_tree);
                    let prefix = if is_selected { "> " } else { "  " };
                    let icon_space = if !icon.is_empty() { " " } else { "" };

                    let line = format!("{}{}{}{}", prefix, icon, icon_space, result.relative_path);

                    let style = if is_selected {
                        selected_style
                    } else {
                        directory_style
                    };

                    surface.set_stringn(inner.x, y, &line, inner.width as usize, style);
                }

                // Scrollbar
                if self.search_results.len() > results_height {
                    self.render_scrollbar(
                        surface,
                        inner.x + inner.width.saturating_sub(1),
                        results_area_y,
                        results_height,
                        self.search_results.len(),
                        theme,
                    );
                }
            }

            InputMode::Add | InputMode::Rename => {
                // Show input prompt
                let prompt_char = if self.input_mode == InputMode::Add {
                    "New: "
                } else {
                    "Name: "
                };
                let input_line = format!("{}{}", prompt_char, self.input_buffer);
                surface.set_stringn(
                    inner.x,
                    inner.y,
                    &input_line,
                    inner.width as usize,
                    prompt_style,
                );

                // Show target directory for add mode
                if self.input_mode == InputMode::Add {
                    let target = self.get_target_directory();
                    let target_str = target
                        .strip_prefix(&self.root)
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|_| ".".to_string());
                    let location = format!(
                        "In: {}/",
                        if target_str.is_empty() {
                            "."
                        } else {
                            &target_str
                        }
                    );
                    surface.set_stringn(
                        inner.x,
                        inner.y + 1,
                        &location,
                        inner.width as usize,
                        info_style,
                    );
                }

                // Show message if any
                if let Some(ref msg) = self.message {
                    let msg_y = if self.input_mode == InputMode::Add {
                        inner.y + 2
                    } else {
                        inner.y + 1
                    };
                    surface.set_stringn(inner.x, msg_y, msg, inner.width as usize, info_style);
                }
            }

            InputMode::ConfirmDelete => {
                // Show what will be deleted
                if let Some(entry) = self.selected_entry() {
                    let name = entry.path.file_name().unwrap_or_default().to_string_lossy();
                    let type_str = if entry.is_dir { "directory" } else { "file" };
                    let msg = format!("Delete {} '{}'?", type_str, name);
                    surface.set_stringn(inner.x, inner.y, &msg, inner.width as usize, prompt_style);

                    let hint = "Press 'y' to confirm, 'n' or Esc to cancel";
                    surface.set_stringn(
                        inner.x,
                        inner.y + 1,
                        hint,
                        inner.width as usize,
                        info_style,
                    );

                    if entry.is_dir {
                        let warning = "WARNING: This will delete all contents!";
                        surface.set_stringn(
                            inner.x,
                            inner.y + 2,
                            warning,
                            inner.width as usize,
                            info_style,
                        );
                    }
                }
            }

            InputMode::Normal => {
                // Show message if any (at the top)
                let content_start_y = if self.message.is_some() {
                    if let Some(ref msg) = self.message {
                        surface.set_stringn(
                            inner.x,
                            inner.y,
                            msg,
                            inner.width as usize,
                            info_style,
                        );
                    }
                    inner.y + 1
                } else {
                    inner.y
                };

                let content_height =
                    inner
                        .height
                        .saturating_sub(if self.message.is_some() { 1 } else { 0 })
                        as usize;

                if self.entries.is_empty() {
                    surface.set_stringn(
                        inner.x,
                        content_start_y,
                        "(empty)",
                        inner.width as usize,
                        text_style,
                    );
                    return;
                }

                self.adjust_scroll(content_height);

                let start = self.scroll_offset;
                let end = (start + content_height).min(self.entries.len());

                // Check which files are open in editor
                let open_files: HashSet<_> = cx
                    .editor
                    .documents()
                    .filter_map(|doc| doc.path().map(|p| p.to_path_buf()))
                    .collect();

                // Get styles for git status and open files
                let git_modified_style = theme
                    .try_get("diff.delta")
                    .unwrap_or_else(|| theme.get("ui.text"));
                let git_added_style = theme
                    .try_get("diff.plus")
                    .unwrap_or_else(|| theme.get("ui.text"));
                let git_deleted_style = theme
                    .try_get("diff.minus")
                    .unwrap_or_else(|| theme.get("ui.text"));
                let open_file_style = theme
                    .try_get("ui.text.focus")
                    .unwrap_or_else(|| theme.get("ui.text"))
                    .add_modifier(Modifier::BOLD);

                for (i, entry) in self.entries[start..end].iter().enumerate() {
                    let y = content_start_y + i as u16;
                    let is_selected = start + i == self.cursor;
                    let is_open = open_files.contains(&entry.path);

                    let indent = "  ".repeat(entry.depth);
                    let icon = self.get_icon(entry, &config.file_tree);
                    let name = entry.path.file_name().unwrap_or_default().to_string_lossy();

                    let prefix = if is_selected { "> " } else { "  " };
                    let icon_space = if !icon.is_empty() { " " } else { "" };

                    // Git status indicator
                    let git_indicator = self
                        .git_statuses
                        .get(&entry.path)
                        .map(|s| s.indicator())
                        .unwrap_or(" ");

                    // File size for files (not directories)
                    let size_str = if !entry.is_dir {
                        std::fs::metadata(&entry.path)
                            .map(|m| Self::format_file_size(m.len()))
                            .unwrap_or_default()
                    } else {
                        String::new()
                    };

                    // Calculate available space for name
                    let prefix_len = prefix.len() + indent.len() + icon.len() + icon_space.len();
                    let suffix_len = 2 + size_str.len(); // git indicator + space + size
                    let available = inner.width as usize - prefix_len - suffix_len;
                    let truncated_name: String = if name.len() > available {
                        format!("{}~", &name[..available.saturating_sub(1)])
                    } else {
                        name.to_string()
                    };

                    // Build the line
                    let name_part = format!("{}{}{}{}{}", prefix, indent, icon, icon_space, truncated_name);
                    let padding = inner.width as usize - name_part.len() - suffix_len;
                    let line = format!(
                        "{}{:padding$}{} {}",
                        name_part,
                        "",
                        git_indicator,
                        size_str,
                        padding = padding.max(0)
                    );

                    // Determine style based on state
                    let style = if is_selected {
                        selected_style
                    } else if is_open {
                        open_file_style
                    } else if entry.is_dir {
                        directory_style
                    } else {
                        // Apply git status color
                        match self.git_statuses.get(&entry.path) {
                            Some(GitStatus::Modified) => git_modified_style,
                            Some(GitStatus::Added) | Some(GitStatus::Untracked) => git_added_style,
                            Some(GitStatus::Deleted) => git_deleted_style,
                            _ => text_style,
                        }
                    };

                    surface.set_stringn(inner.x, y, &line, inner.width as usize, style);
                }

                // Scrollbar
                if self.entries.len() > content_height {
                    self.render_scrollbar(
                        surface,
                        inner.x + inner.width.saturating_sub(1),
                        content_start_y,
                        content_height,
                        self.entries.len(),
                        theme,
                    );
                }
            }

            InputMode::Help => {
                // Help is rendered separately and we return early, so this is unreachable
            }
        }

        // Render preview pane if we have space
        if let Some(preview_area) = preview_area {
            self.render_preview(preview_area, surface, cx);
        }
    }

    fn cursor(&self, area: Rect, _editor: &Editor) -> (Option<Position>, CursorKind) {
        // Help mode has no cursor
        if self.input_mode == InputMode::Help {
            return (None, CursorKind::Hidden);
        }

        // Calculate tree area (left half if wide enough for preview and preview is enabled)
        let tree_area = if self.show_preview && area.width > MIN_AREA_WIDTH_FOR_PREVIEW {
            Rect::new(area.x, area.y, area.width / 2, area.height)
        } else {
            area
        };

        match self.input_mode {
            InputMode::Search => {
                let block = Block::default().borders(Borders::ALL);
                let inner = block.inner(tree_area);
                let cursor_col = inner.x as usize + 1 + self.input_cursor;
                let cursor_row = inner.y as usize;
                (
                    Some(Position::new(cursor_row, cursor_col)),
                    CursorKind::Block,
                )
            }
            InputMode::GoToFolder => {
                let block = Block::default().borders(Borders::ALL);
                let inner = block.inner(tree_area);
                let prompt_len = "Go: ".len();
                let cursor_col = inner.x as usize + prompt_len + self.input_cursor;
                let cursor_row = inner.y as usize;
                (
                    Some(Position::new(cursor_row, cursor_col)),
                    CursorKind::Block,
                )
            }
            InputMode::Add => {
                let block = Block::default().borders(Borders::ALL);
                let inner = block.inner(tree_area);
                let prompt_len = "New: ".len();
                let cursor_col = inner.x as usize + prompt_len + self.input_cursor;
                let cursor_row = inner.y as usize;
                (
                    Some(Position::new(cursor_row, cursor_col)),
                    CursorKind::Block,
                )
            }
            InputMode::Rename => {
                let block = Block::default().borders(Borders::ALL);
                let inner = block.inner(tree_area);
                let prompt_len = "Name: ".len();
                let cursor_col = inner.x as usize + prompt_len + self.input_cursor;
                let cursor_row = inner.y as usize;
                (
                    Some(Position::new(cursor_row, cursor_col)),
                    CursorKind::Block,
                )
            }
            _ => (None, CursorKind::Hidden),
        }
    }

    fn id(&self) -> Option<&'static str> {
        Some(ID)
    }
}

impl FileTree {
    fn render_scrollbar(
        &self,
        surface: &mut Surface,
        x: u16,
        y: u16,
        viewport_height: usize,
        content_len: usize,
        theme: &helix_view::Theme,
    ) {
        let scrollbar_height = viewport_height as f64;
        let content_height = content_len as f64;
        let thumb_height = ((scrollbar_height / content_height) * scrollbar_height)
            .max(1.0)
            .min(scrollbar_height) as u16;
        let thumb_pos = ((self.scroll_offset as f64 / content_height) * scrollbar_height) as u16;

        let scrollbar_style = theme
            .try_get("ui.menu.scroll")
            .unwrap_or_else(|| theme.get("ui.text"));

        for i in 0..viewport_height as u16 {
            let char = if i >= thumb_pos && i < thumb_pos + thumb_height {
                '\u{2588}' // Full block
            } else {
                '\u{2591}' // Light shade
            };
            surface.set_string(x, y + i, char.to_string(), scrollbar_style);
        }
    }
}
