# User Data & Configuration Locations

This project follows the [XDG Base Directory Specification](
https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html) for storing user data and
configuration.

---

## Directory Structure

### Project Directory

The project directory contains read-only defaults and temporary logs:

```text
<project>/
├── config/
│   └── defaults.json       # Project defaults (in Git)
├── results/                # Optional: can be symlinked to user results
└── logs/                   # Benchmark and web logs (in project root)
```

### User Directories (XDG Standard)

User-specific data is stored in standard XDG locations:

```text
~/.config/lm-studio-bench/
├── defaults.json           # User configuration overrides (optional)
└── presets/
    ├── my_fast_test.json   # User preset example
    └── my_quality.json     # User preset example

~/.local/share/lm-studio-bench/results/
├── benchmark_results_<timestamp>.json
├── benchmark_results_<timestamp>.csv
├── benchmark_results_<timestamp>.pdf
├── benchmark_results_<timestamp>.html
├── benchmark_cache.db      # SQLite benchmark cache
└── model_metadata.db       # Model metadata cache
```

---

## Configuration Loading

Configuration is loaded with the following priority:

1. **CLI Arguments** (highest priority)
2. **User Config** (`~/.config/lm-studio-bench/defaults.json`)
3. **Project Config** (`config/defaults.json`)
4. **Hard-coded Defaults** (in code)

### Example

**Project** (`config/defaults.json`):

```json
{
  "num_runs": 3,
  "context_length": 2048,
  "lmstudio": {
    "use_rest_api": false
  }
}
```

**User** (`~/.config/lm-studio-bench/defaults.json`):

```json
{
  "num_runs": 5,
  "lmstudio": {
    "use_rest_api": true
  }
}
```

**Result** (merged configuration):

```json
{
  "num_runs": 5,              // User override
  "context_length": 2048,     // Project default
  "lmstudio": {
    "use_rest_api": true      // User override
  }
}
```

**With CLI**:

```bash
./run.py --runs 10 --context 4096
```

Final configuration:

- `num_runs`: 10 (CLI)
- `context_length`: 4096 (CLI)
- `use_rest_api`: true (User config)

---

## Creating User Configuration

### Step 1: Create Config Directory

```bash
mkdir -p ~/.config/lm-studio-bench
```

### Step 2: Create User Config File

```bash
nano ~/.config/lm-studio-bench/defaults.json
```

### Step 3: Add Your Overrides

Only include fields you want to override:

```json
{
  "num_runs": 5,
  "context_length": 4096,
  "inference": {
    "temperature": 0.7
  }
}
```

**Important**: You only need to specify fields you want to change. All other values will use project defaults.

---

## Data Migration

On first run, the tool automatically:

1. Creates user data directories (`~/.config/...` and `~/.local/share/...`)
2. Places new results in `~/.local/share/lm-studio-bench/results/`

**Note**: If you have existing data in a legacy `results/` directory, you'll need to manually move it to the new location.

---

## Benefits of XDG Structure

### For Users

- ✅ **Persistent User Settings**: Configuration survives project updates
- ✅ **Cleaner Project Directory**: User data separated from code
- ✅ **Standard Locations**: Follows Linux conventions
- ✅ **Easy Backups**: Backup `~/.local/share/lm-studio-bench/` and `~/.config/lm-studio-bench/`
- ✅ **Multi-User Support**: Each user has their own data

### For Developers

- ✅ **No Git Conflicts**: User data not in version control
- ✅ **Clean Updates**: `git pull` doesn't affect user data
- ✅ **Portable**: Project directory can be moved/deleted without losing user data

---

## Environment Variables

You can override paths with environment variables:

```bash
# Override config directory
export XDG_CONFIG_HOME="$HOME/my-configs"

# Override data directory
export XDG_DATA_HOME="$HOME/my-data"

# Now config is in: $HOME/my-configs/lm-studio-bench/defaults.json
# Now results are in: $HOME/my-data/lm-studio-bench/results/
```

---

## FAQ

### Q: Where are my benchmark results stored?

**A**: `~/.local/share/lm-studio-bench/results/`

The project `results/` directory is now a symlink for backward compatibility.

### Q: Where do I put custom configuration?

**A**: `~/.config/lm-studio-bench/defaults.json`

Only include fields you want to override from project defaults.

### Q: Where are user presets stored?

**A**: `~/.config/lm-studio-bench/presets/`

Built-in readonly presets (`default`, `quick_test`, `high_quality`,
`resource_limited`) are not stored as files.

### Q: What happens to my old results?

**A**: On first run, they are automatically migrated to `~/.local/share/lm-studio-bench/results/`

### Q: Can I use the old `config/defaults.json`?

**A**: Yes! It's still used as project defaults. User config in `~/.config/` overrides it.

### Q: How do I reset to project defaults?

**A**: Delete your user config:

```bash
rm ~/.config/lm-studio-bench/defaults.json
```

### Q: How do I backup my data?

**A**: Backup these directories:

```bash
# Configuration
tar -czf lms-bench-config.tar.gz ~/.config/lm-studio-bench/

# Results and cache
tar -czf lms-bench-data.tar.gz ~/.local/share/lm-studio-bench/
```

### Q: What about logs?

**A**: Logs remain in the project directory: `<project>/logs/`

This is intentional for development/debugging purposes.

---

## See Also

- [Configuration Reference](CONFIGURATION.md) - All configuration options
- [Architecture Documentation](ARCHITECTURE.md) - System design
- [XDG Base Directory Spec](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html) - Standard specification
