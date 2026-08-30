"""Runtime paths + YAML config loader.

Replaces the old `src/constants.py` (hardcoded absolute paths). Everything
non-portable comes from either:
  1. environment variables (SIMT_*)  — set by bin/_env.sh, portable.
  2. YAML configs under configs/{tag}.yaml — one per experiment.

Only fully-generic filesystem constants live here — repo root, data
symlink target, model base. Backbone paths, tokenizer paths, seeds,
hyperparameters all belong in the tag's YAML.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ─────────── infrastructure paths (env-overridable) ───────────
REPO_ROOT = Path(os.environ.get(
    "SIMT_REPO_ROOT",
    Path(__file__).resolve().parents[1],
))

DATA_ROOT = Path(os.environ.get(
    "SIMT_DATA_ROOT",
    REPO_ROOT / "data",
))

MODEL_BASE = Path(os.environ.get(
    "SIMT_MODEL_BASE",
    Path.home() / ".cache" / "simt-models",
))

HF_CACHE = Path(os.environ.get(
    "HF_HOME",
    Path.home() / ".cache" / "huggingface",
))


# ─────────── YAML config loader ───────────
import re
_ENVVAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _expand_env(obj: Any) -> Any:
    """Recursively replace ${ENV_VAR} in string leaves of a nested dict/list.

    Missing env vars leave the placeholder untouched (fail-loud at path use).
    """
    if isinstance(obj, str):
        return _ENVVAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    return obj


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a configs/{tag}.yaml as a nested dict.

    Env-var placeholders `${VAR}` in string values are expanded from the
    process environment. Missing vars leave the placeholder in place.

    Every pipeline entry-point (scripts/01_..04_*.py) accepts
    `--config configs/{tag}.yaml`. See configs/example.yaml for the schema.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML required for config loading — pip install pyyaml"
        ) from e
    raw = yaml.safe_load(Path(path).read_text())
    return _expand_env(raw)


def resolve_backbone_path(config: dict[str, Any]) -> Path:
    """Return the backbone's on-disk path, preferring config.backbone.local_path
    over MODEL_BASE / {hf_id}. HF id fallback matches the standard cache layout.
    """
    b = config.get("backbone", {})
    if lp := b.get("local_path"):
        return Path(lp)
    if hf_id := b.get("hf_id"):
        # HF cache layout: models--{org}--{name}
        parts = hf_id.replace("/", "--")
        return HF_CACHE / "hub" / f"models--{parts}"
    raise ValueError("config.backbone must have local_path or hf_id")
