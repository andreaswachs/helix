# Helix (Fork with File Tree Viewer)

This is a **personal fork** of the [Helix editor](https://github.com/helix-editor/helix) with additional features tailored to my workflow. The main addition is a **NeoTree-style file tree viewer** with fuzzy search, preview pane, and git integration.

For documentation on the core Helix editor, please refer to the [official Helix documentation](https://docs.helix-editor.com/).

## Installation

### Homebrew (Recommended)

```bash
brew tap andreaswachs/tap
brew install helix
```

To upgrade:
```bash
brew upgrade helix
```

### From Source

```bash
git clone https://github.com/andreaswachs/helix.git
cd helix
cargo install --path helix-term --locked
```

Ensure `~/.cargo/bin` is in your `PATH`.

---

## Fork Features

### File Tree Viewer

A persistent, NeoTree-style file tree panel with preview, fuzzy search, and git status integration.

#### Opening the File Tree

| Command | Description |
|---------|-------------|
| `:ft` or `:tree` | Toggle file tree |
| `:ftr` or `:tree-resume` | Resume last file tree session (restores search, position, state) |

Or bind to a key in your `config.toml`:

```toml
[keys.normal.space]
f = "file_tree_toggle"
"'" = "file_tree_resume"   # Resume last session with Space+'
```

#### Keybindings (Inside File Tree)

**Navigation:**
| Key | Action |
|-----|--------|
| `j` / `k` or `↑` / `↓` | Move cursor |
| `h` / `l` or `←` / `→` | Collapse / Expand directory |
| `Enter` | Open file or toggle directory |
| `g` | Go to top |
| `G` | Go to bottom |
| `Ctrl+d` / `Ctrl+u` | Page down / up |

**Search & Jump:**
| Key | Action |
|-----|--------|
| `/` | Fuzzy search files (fzf-style) |
| `Ctrl+g` | Go to folder (fuzzy search directories) |
| `f` | Jump to currently open file |

**File Operations:**
| Key | Action |
|-----|--------|
| `a` | Add file (end with `/` for directory) |
| `d` | Delete file/directory |
| `r` | Rename |
| `y` | Copy path to clipboard |

**Preview:**
| Key | Action |
|-----|--------|
| `p` | Toggle preview pane |
| `Ctrl+j` / `Ctrl+k` | Scroll preview down / up |

**Other:**
| Key | Action |
|-----|--------|
| `R` | Refresh tree and git status |
| `Ctrl+s` | Open in horizontal split |
| `Ctrl+v` | Open in vertical split |
| `?` | Toggle help |
| `q` / `Esc` | Close (state is saved for resume) |

#### Fuzzy Search

The `/` search uses **fzf-style fuzzy matching**:

- **Non-contiguous matching**: Type `srcts` to match `src/components/test.ts`
- **Case insensitive**: Matches regardless of case
- **Path-aware**: Searches across full relative path (directories + filename)

Results are scored and sorted with best matches first:
- Consecutive character matches score higher
- Matches at word boundaries (`/`, `.`, `_`, `-`) score higher
- Matches in the filename score higher than directory matches
- Shorter paths are preferred when scores are equal

**Examples:**
- `ftrs` matches `helix-term/src/ui/file_tree.rs`
- `uift` matches `helix-term/src/ui/file_tree.rs`
- `cmdty` matches `helix-term/src/commands/typed.rs`

#### Configuration

Add to your `~/.config/helix/config.toml`:

```toml
[editor.file-tree]
# Display mode: "floating" (overlay) or "side" (persistent panel)
mode = "floating"

# Panel position for side mode: "left" or "right"
position = "left"

# Width in columns (for side mode)
width = 30

# Show file/folder icons
show-icons = true

# File filtering (inherited from explorer settings)
hidden = true          # Hide dotfiles
git-ignore = true      # Respect .gitignore
ignore = true          # Respect .ignore files
follow-symlinks = true # Follow symbolic links

[editor.file-tree.icons]
# Customize directory indicators
directory = "▸"
directory-open = "▾"
file = " "

# Custom icons by file extension
[editor.file-tree.icons.extensions]
rs = "🦀"
py = "🐍"
js = "📜"
ts = "📜"
go = "🐹"
rb = "💎"
md = "📝"
toml = "⚙️"
json = "📋"

# Custom icons by filename
[editor.file-tree.icons.filenames]
"Cargo.toml" = "📦"
"README.md" = "📖"
"Makefile" = "🔧"
".gitignore" = "🚫"
```

#### Features

- **Preview pane**: Shows file contents with syntax highlighting (like the built-in picker)
- **Git status indicators**: Shows `M` (modified), `A` (added), `D` (deleted), `?` (untracked)
- **File sizes**: Displayed next to each file
- **Open file highlighting**: Currently open files are highlighted in bold
- **Session resume**: Close with `q` and resume with `:ftr` to restore your exact state

---

## Upstream Helix

This fork tracks the upstream [Helix editor](https://github.com/helix-editor/helix). For:

- **Core documentation**: [docs.helix-editor.com](https://docs.helix-editor.com/)
- **Keybindings**: [Keymap documentation](https://docs.helix-editor.com/keymap.html)
- **Themes**: [Theme documentation](https://docs.helix-editor.com/themes.html)
- **Languages**: [Language support](https://docs.helix-editor.com/lang-support.html)
- **Troubleshooting**: [Wiki](https://github.com/helix-editor/helix/wiki/Troubleshooting)

---

## License

[MPL-2.0](./LICENSE) - Same as upstream Helix.
