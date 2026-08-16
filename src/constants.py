"""Single source of truth for absolute paths — imported everywhere else."""

from pathlib import Path

REPO_ROOT = Path("/g/data/ba39/dipankar/simt-tor-26")

# Shared model weights root (see HOUSEKEEPING §5). Never hard-code below.
MODEL_BASE = Path("/g/data/po67/dipankar/models")

# Shared po67 cache (see HOUSEKEEPING §6.3). Never $HOME, never scratch.
HF_CACHE = Path("/g/data/po67/dipankar/cache")

# Data lives outside the repo; ./data is a symlink to this.
DATA_ROOT = Path("/g/data/po67/dipankar/data/simt-tor-26")

# Primary backbone: Gemma-4-E2B base (LOG.md 2026-08-15 decision —
# supersedes the earlier -it choice). Base model matches METHOD §1's raw
# next-token-prediction setup without instruction-tuning prompt confounds.
# Same model annotates and fine-tunes per METHOD §5.
PRIMARY_BACKBONE = MODEL_BASE / "gemma-4-E2B"

# Instruction-tuned variant kept for cross-checks — chat-template runs
# during Phase 1 used this. Not the primary going forward.
PRIMARY_BACKBONE_IT = MODEL_BASE / "gemma-4-E2B-it"

# Cross-family ablation partner at matched ~2B size — isolates family from
# scale in the annotator-model ablation (EXPERIMENTS grid, row 2).
ABLATION_BACKBONE = MODEL_BASE / "Qwen3.5-2B"

# Scale-up candidate — only used after Gate 1 passes on PRIMARY_BACKBONE.
SCALE_BACKBONE = MODEL_BASE / "gemma-4-E4B"
SCALE_BACKBONE_IT = MODEL_BASE / "gemma-4-E4B-it"

# Scorers on disk.
COMET_DA = MODEL_BASE / "wmt22-comet-da"
COMET_KIWI = MODEL_BASE / "wmt22-cometkiwi-da"

# Fixed defaults — override in configs, not in code.
DEFAULT_SEED = 42
