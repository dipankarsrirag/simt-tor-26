"""Single source of truth for absolute paths — imported everywhere else."""

from pathlib import Path

REPO_ROOT = Path("/g/data/ba39/dipankar/simt-tor-26")

# Shared model weights root (see HOUSEKEEPING §5). Never hard-code below.
MODEL_BASE = Path("/g/data/po67/dipankar/models")

# Shared po67 cache (see HOUSEKEEPING §6.3). Never $HOME, never scratch.
HF_CACHE = Path("/g/data/po67/dipankar/cache")

# Data lives outside the repo; ./data is a symlink to this.
DATA_ROOT = Path("/g/data/po67/dipankar/data/simt-tor-26")

# Primary 2B backbone (HOUSEKEEPING §5). Same model annotates and fine-tunes.
PRIMARY_BACKBONE = MODEL_BASE / "Qwen3.5-2B"

# Cross-family ablation at matched size.
ABLATION_BACKBONE = MODEL_BASE / "gemma-4-E2B-it"

# Scorers on disk.
COMET_DA = MODEL_BASE / "wmt22-comet-da"
COMET_KIWI = MODEL_BASE / "wmt22-cometkiwi-da"

# Fixed defaults — override in configs, not in code.
DEFAULT_SEED = 42
