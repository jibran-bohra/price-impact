## Price Impact

Utilities and scripts for modeling and visualizing trade price impact using simple linear (OW) and nonlinear (AFS) models. Figures are saved to `report/figures` and data is read from `data/merged_data.csv`.

### Prerequisites

- **Python**: exactly `3.11.9` (managed via `pyenv`)
- **uv** (fast Python package manager by Astral)
- macOS or Linux shell with `bash`/`zsh`

Install helpers (macOS examples):

```bash
# pyenv (manage Python versions)
brew install pyenv

# uv (choose one)
# Option A: Homebrew
brew install uv
# Option B: Official installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Quickstart

```bash
# 1) Clone and enter the repo
git clone <your-fork-or-this-repo-url>
cd price-impact

# 2) Ensure Python 3.11.9 is available and selected
pyenv install 3.11.9 -s
pyenv local 3.11.9

# 3) Create and activate a virtual environment
uv venv
source .venv/bin/activate

# 4) Install locked dependencies
uv sync

# 5) Run the visualization script (generates figures under report/figures)
uv run python scripts/visualize_price_impact_models.py
```

If you prefer using the activated venv directly instead of `uv run`:

```bash
python scripts/visualize_price_impact_models.py
```

### Data

- The script expects a CSV at `data/merged_data.csv`.
- The repository includes a sample `merged_data.csv`; replace it with your dataset to run on your own data.

### Outputs

Generated figures are written to `report/figures/`:
- `01_exploratory_analysis.png`
- `02_model_comparison.png`
- `03_detailed_analysis.png`
- `04_3d_impact_surfaces.png`
- `05_volume_bucket_analysis.png`

### Code organization

- `scripts/visualize_price_impact_models.py`: entry-point to fit models and save plots
- `src/impact_models.py`: OW and AFS model implementations
- `src/plotting.py`: plotting utilities used by the script
- `src/utils.py`: data loading helpers
- `data/merged_data.csv`: example input data
- `report/figures/`: output directory for plots
- `pyproject.toml`: project metadata and dependencies (managed by `uv`)

### Notes

- Python version is pinned to `3.11.9` in `pyproject.toml`. Using a different minor version may fail installation.
- Dependencies are resolved and installed from `uv.lock` for reproducibility via `uv sync`.
- If you see an error like "can't open file ... visualize_price_impact.py", double‑check the script name is `scripts/visualize_price_impact_models.py`.