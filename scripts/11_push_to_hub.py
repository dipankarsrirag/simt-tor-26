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
    ap.add_argument("--model_dir", type=Path, default=None,
                    help="Override checkpoint dir. Default: results/train/{tag}/final/. "
                         "Useful for pushing archived baselines from _archive/…/final/.")
    ap.add_argument("--eval_dir", type=Path, default=None,
                    help="Override eval-JSON dir. Default: results/eval/{tag}/.")
    ap.add_argument("--log_dir", type=Path, default=None,
                    help="Override logs dir. Default: logs/{tag}/.")
    ap.add_argument("--annotate_dir", type=Path, default=None,
                    help="Override annotator-matrices dir (that gets uploaded as 'annotate/'). "
                         "Default: results/annotate/{annotator}/. Useful when matrices live in _archive.")
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

    train_dir = args.model_dir or (REPO_ROOT / "results" / "train" / tag / "final")
    log_dir   = args.log_dir   or (REPO_ROOT / "logs" / tag)
    eval_dir  = args.eval_dir  or (REPO_ROOT / "results" / "eval" / tag)

    missing = []
    if not train_dir.exists(): missing.append(f"model_dir: {train_dir}")
    if not log_dir.exists():   missing.append(f"log_dir:   {log_dir}")
    if not eval_dir.exists():  missing.append(f"eval_dir:  {eval_dir}")
    if missing:
        print(f"MISSING (run bin/run configs/{args.config.name} first, "
              f"or pass --model_dir / --log_dir / --eval_dir to override):",
              file=sys.stderr)
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
    # HF's upload_folder walks the on-disk tree with os.walk (not following
    # symlinks to directories) — top-level dir symlinks get silently ignored.
    # Solution: mirror each directory as a real dir, and symlink each FILE
    # into it. That way os.walk enters the dir and sees the file symlinks
    # (which upload_folder does follow).
    def mirror_tree(src: Path, dst: Path):
        n = 0
        for f in src.rglob("*"):
            if f.is_file():
                rel = f.relative_to(src)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(f.resolve())
                n += 1
        return n

    # Model + tokenizer files: flat top-level (safetensors, tokenizer.json, ...)
    n_model = 0
    for f in train_dir.iterdir():
        if f.is_file():
            (stage / f.name).symlink_to(f.resolve())
            n_model += 1
    print(f"  model files:      {n_model} ({train_dir})")

    n_logs = mirror_tree(log_dir, stage / "logs")
    print(f"  logs:             {n_logs} ({log_dir})")

    n_eval = mirror_tree(eval_dir, stage / "eval")
    print(f"  eval JSONs:       {n_eval} ({eval_dir})")

    # Include the annotator matrices this tag OWNS (i.e. produced with its
    # own backbone). Cross-annotation experiments don't upload matrices —
    # they pulled them from another tag's HF repo.
    annotator = cfg["annotate"].get("annotator", "same_as_backbone")
    if annotator == "same_as_backbone":
        annotator_name = Path(cfg["backbone"].get("local_path") or cfg["backbone"]["hf_id"]).name
        matrices_root = args.annotate_dir or (REPO_ROOT / "results" / "annotate" / annotator_name)
        if matrices_root.exists():
            n_annot = mirror_tree(matrices_root, stage / "annotate")
            print(f"  annot matrices:   {n_annot} ({matrices_root})")

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
