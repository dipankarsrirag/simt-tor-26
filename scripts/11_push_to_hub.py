"""Push a trained SFT checkpoint + logs + eval results to HuggingFace Hub.

Packages one experiment tag into a private HF repo for cross-check /
review. Contents pushed:

    {org}/simt-{tag}/
    ├── config.safetensors / .bin / model shards       (from results/train/{tag}/final/)
    ├── tokenizer files                                 (from results/train/{tag}/final/)
    ├── config.yaml                                     (the experiment YAML that produced this run)
    ├── manifest.json                                   (git sha, hostname, ngpus, timestamps)
    ├── logs/                                            (all per-stage logs from bin/run)
    ├── eval/                                            (all eval JSONs — hyps + refs + metrics)
    └── README.md                                       (auto-generated model card)

Usage:
    bin/11_push_to_hub configs/{NN_tag}.yaml [--org OWNER] [--private] [--dry_run]

Requires:
    - `huggingface-cli login` (or HF_TOKEN env var)
    - `pip install huggingface_hub`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import REPO_ROOT, load_config


MODEL_CARD_TEMPLATE = """---
tags:
- simultaneous-translation
- east
- ot-annotation
language: {langs}
license: mit
---

# {tag}

Experiment `{tag}` from the *Teacher-Free Read/Write Annotation for Simultaneous
Machine Translation* project.

## Recipe

- **Backbone:** `{backbone}`
- **Corpus:** `{corpus}`
- **Annotator:** `{annotator}`
- **Criterion:** `{criterion}` (τ = {tau})
- **Latencies:** {latencies}

## Files in this repo

- `config.yaml` — the exact experiment config that produced this run.
- `manifest.json` — git sha, hostname, GPUs, timestamps.
- `logs/` — per-stage stdout+stderr from `bin/run`.
- `eval/` — every landed eval-JSON cell (hypothesis, reference, AL, BLEU).
- SFT checkpoint (`*.safetensors` + tokenizer).

## Reproduce

```bash
git clone https://github.com/dipankarsrirag/simt-tor-26.git
cd simt-tor-26
cp .simtrc.example .simtrc  # edit paths for your setup
bin/run configs/{tag_filename} --ngpus N
```

Git commit at time of run: `{git_sha}`
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path,
                    help="Path to configs/{tag}.yaml.")
    ap.add_argument("--org", default=None,
                    help="HF organisation for the target repo. Default: "
                         "HF_HUB_ORG env var, else 'unswnlporg'. "
                         "Write access to unswnlporg is required — join at "
                         "https://huggingface.co/unswnlporg.")
    ap.add_argument("--private", action="store_true", default=True,
                    help="Create a private HF repo (default: TRUE).")
    ap.add_argument("--public", action="store_true",
                    help="Override --private and make the repo public.")
    ap.add_argument("--repo_name", default=None,
                    help="Override the repo name. Default: 'tor-simt-{tag}'.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print what would be uploaded without pushing.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tag = cfg["tag"]

    org = args.org or os.environ.get("HF_HUB_ORG", "unswnlporg")
    # HF repo names use hyphens, not underscores (convention). Convert the
    # local tag (underscore-separated, e.g. gemma_2b_curated) → hyphenated
    # repo name (gemma-2b-curated).
    repo_name = args.repo_name or f"tor-simt-{tag.replace('_', '-')}"
    repo_id = f"{org}/{repo_name}"
    is_private = not args.public   # --private is the default; --public overrides

    train_dir = REPO_ROOT / "results" / "train" / tag / "final"
    log_dir = REPO_ROOT / "logs" / tag
    eval_dir = REPO_ROOT / "results" / "eval" / tag

    missing = []
    if not train_dir.exists(): missing.append(str(train_dir))
    if not log_dir.exists():   missing.append(str(log_dir))
    if not eval_dir.exists():  missing.append(str(eval_dir))
    if missing:
        print(f"MISSING (run bin/run configs/{args.config.name} first):", file=sys.stderr)
        for m in missing: print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    # Build the model card
    manifest = json.loads((log_dir / "manifest.json").read_text()) if (log_dir / "manifest.json").exists() else {}
    langs = sorted({d.split("-")[0] for d in cfg["source_pool"]["directions"]} |
                   {d.split("-")[1] for d in cfg["source_pool"]["directions"]})
    card = MODEL_CARD_TEMPLATE.format(
        tag=tag,
        tag_filename=args.config.name,
        backbone=cfg["backbone"].get("hf_id") or cfg["backbone"]["local_path"],
        corpus=cfg["source_pool"]["corpus"],
        annotator=cfg["annotate"].get("annotator", "same_as_backbone"),
        criterion=cfg["annotate"]["criterion"],
        tau=cfg["annotate"]["tau"],
        latencies=cfg["annotate"]["latency_bins"],
        langs=langs,
        git_sha=manifest.get("git_sha", "unknown"),
    )

    # Assemble a staging dir (symlinks where possible)
    import tempfile
    stage = Path(tempfile.mkdtemp(prefix=f"simt-push-{tag}-"))
    print(f"Staging in: {stage}")
    # Copy the config + manifest + model card
    (stage / "config.yaml").write_text(args.config.read_text())
    if (log_dir / "manifest.json").exists():
        (stage / "manifest.json").write_text((log_dir / "manifest.json").read_text())
    (stage / "README.md").write_text(card)
    # Symlink model files, logs/, eval/
    for f in train_dir.iterdir():
        (stage / f.name).symlink_to(f.resolve())
    (stage / "logs").symlink_to(log_dir.resolve())
    (stage / "eval").symlink_to(eval_dir.resolve())

    # Include the annotator matrices this tag OWNS (i.e. produced with its
    # own backbone). Cross-annotation experiments don't upload matrices —
    # they pulled them from another tag's HF repo.
    annotator = cfg["annotate"].get("annotator", "same_as_backbone")
    if annotator == "same_as_backbone":
        annotator_name = Path(cfg["backbone"].get("local_path") or cfg["backbone"]["hf_id"]).name
        matrices_root = REPO_ROOT / "results" / "annotate" / annotator_name
        if matrices_root.exists():
            (stage / "annotate").symlink_to(matrices_root.resolve())
            print(f"  including annotator matrices: results/annotate/{annotator_name}/")

    print(f"\nTarget repo: {repo_id}  (private={is_private})")
    print(f"Contents to upload:")
    for p in sorted(stage.iterdir()):
        size = p.stat().st_size if p.is_file() else "<dir>"
        print(f"  {p.name}  {size}")

    if args.dry_run:
        print("\nDRY-RUN — not pushing. Staging dir left at:", stage)
        return

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    try:
        create_repo(repo_id, private=is_private, exist_ok=True)
    except Exception as e:
        sys.exit(f"\nERROR creating {repo_id}: {e}\n"
                 f"You need write access to the '{org}' organisation.\n"
                 f"Join at https://huggingface.co/{org} and ask an owner for write role.")
    print(f"\nUploading to https://huggingface.co/{repo_id} ...")
    api.upload_folder(
        folder_path=str(stage),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"{repo_name} — git@{manifest.get('git_sha', 'unknown')[:8]}",
    )
    print(f"\nDone: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
