"""Pull an experiment tag's artifacts from HuggingFace Hub into the local tree.

Companion to scripts/11_push_to_hub.py. Places files into the same layout
they were pushed from, so the local repo can act as if the experiment was
run here.

Files retrieved:
    annotate/           → results/annotate/{annotator}/{pair}/matrices.jsonl
    eval/               → results/eval/{tag}/*.json
    logs/               → logs/{tag}/*
    (model files)       → results/train/{tag}/final/{safetensors, tokenizer, ...}
    config.yaml         → configs/{filename from config.yaml comment}.yaml (only if --with_config)
    manifest.json       → logs/{tag}/manifest.json

Common uses:
    # Quang pulls Dipankar's Gemma-2B annotator matrices for cross-annot experiments
    bin/12_pull_from_hub --repo dipankarsrirag/simt-gemma_2b_curated --only annotate

    # Full pull (annot matrices + model + logs + evals)
    bin/12_pull_from_hub --repo dipankarsrirag/simt-gemma_2b_curated
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import REPO_ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tag", help="Experiment tag (short-form). Resolves to "
                                    "unswnlporg/tor-simt-{tag}. Use --org to override.")
    src.add_argument("--repo", help="Full HF repo id, e.g. unswnlporg/tor-simt-gemma-2b-curated.")
    ap.add_argument("--org", default=None,
                    help="HF organisation for --tag lookups. Default: "
                         "HF_HUB_ORG env var, else 'unswnlporg'.")
    ap.add_argument("--only", choices=["annotate", "model", "logs", "eval", "all"],
                    default="all",
                    help="Restrict what to pull (default: all).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite local files if present. Default: skip existing.")
    args = ap.parse_args()

    import os as _os
    if args.tag:
        org = args.org or _os.environ.get("HF_HUB_ORG", "unswnlporg")
        # HF repo names use hyphens; local tags may use underscores. Accept
        # either form and normalise for the URL.
        tag_slug = args.tag.replace("_", "-")
        repo = f"{org}/tor-simt-{tag_slug}"
    else:
        repo = args.repo

    from huggingface_hub import snapshot_download

    print(f"Pulling {repo} (--only {args.only})")

    # Download snapshot into a scratch dir first, then move pieces into place.
    if args.only == "all":
        allow_patterns = None
    else:
        allow_patterns = {
            "annotate": ["annotate/*", "annotate/**/*", "config.yaml"],
            "model":    ["*.safetensors", "*.bin", "*.json", "tokenizer*",
                          "special_tokens_map.json", "config.yaml"],
            "logs":     ["logs/*", "logs/**/*", "manifest.json"],
            "eval":     ["eval/*", "eval/**/*"],
        }[args.only]

    try:
        snap = Path(snapshot_download(
            repo_id=repo,
            repo_type="model",
            allow_patterns=allow_patterns,
        ))
    except Exception as e:
        sys.exit(f"\nERROR pulling {repo}: {e}\n"
                 f"You need read access. If this is a private unswnlporg repo, "
                 f"join at https://huggingface.co/unswnlporg and run "
                 f"`huggingface-cli login`.")
    print(f"  snapshot: {snap}")

    # Read the config to learn the tag + annotator
    cfg_path = snap / "config.yaml"
    if not cfg_path.exists():
        print(f"  WARNING: no config.yaml in snapshot; can't derive tag / annotator", file=sys.stderr)
        cfg = None
        tag = repo.split("/")[-1].replace("tor-simt-", "").replace("simt-", "")
        annotator_name = None
    else:
        cfg = load_config(cfg_path)
        tag = cfg["tag"]
        annotator = cfg["annotate"].get("annotator", "same_as_backbone")
        if annotator == "same_as_backbone":
            annotator_name = Path(cfg["backbone"].get("local_path") or cfg["backbone"]["hf_id"]).name
        else:
            annotator_name = Path(annotator).name
    print(f"  tag={tag}  annotator={annotator_name}")

    # Helper: copy src → dst, skipping existing files unless --force
    def move(src: Path, dst: Path):
        if not src.exists():
            return 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        if src.is_dir():
            for p in src.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(src)
                    target = dst / rel
                    if target.exists() and not args.force:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
                    n += 1
        else:
            if not dst.exists() or args.force:
                shutil.copy2(src, dst)
                n = 1
        return n

    total = 0
    if args.only in ("annotate", "all") and annotator_name:
        n = move(snap / "annotate", REPO_ROOT / "results" / "annotate" / annotator_name)
        print(f"  annotator matrices → results/annotate/{annotator_name}/  ({n} files)")
        total += n

    if args.only in ("eval", "all"):
        n = move(snap / "eval", REPO_ROOT / "results" / "eval" / tag)
        print(f"  eval JSONs → results/eval/{tag}/  ({n} files)")
        total += n

    if args.only in ("logs", "all"):
        n = move(snap / "logs", REPO_ROOT / "logs" / tag)
        # manifest.json lives at snapshot root, not inside logs/
        if (snap / "manifest.json").exists():
            (REPO_ROOT / "logs" / tag).mkdir(parents=True, exist_ok=True)
            target = REPO_ROOT / "logs" / tag / "manifest.json"
            if not target.exists() or args.force:
                shutil.copy2(snap / "manifest.json", target)
                n += 1
        print(f"  logs + manifest → logs/{tag}/  ({n} files)")
        total += n

    if args.only in ("model", "all"):
        # Model files live at snapshot root (safetensors, tokenizer, config.json)
        target_train = REPO_ROOT / "results" / "train" / tag / "final"
        target_train.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in snap.iterdir():
            if p.is_dir(): continue
            if p.name in ("README.md", "config.yaml", "manifest.json", ".gitattributes"):
                continue
            target = target_train / p.name
            if not target.exists() or args.force:
                shutil.copy2(p, target)
                n += 1
        print(f"  model files → results/train/{tag}/final/  ({n} files)")
        total += n

    print(f"\nDone: {total} files pulled from {repo}")


if __name__ == "__main__":
    main()
