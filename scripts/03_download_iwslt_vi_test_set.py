"""Download IWSLT'15 EN-VI (Luong+Manning) + PhoMT test sets from HuggingFace.

Outputs canonical vi-en / en-vi eval files. Defaults land under
$SIMT_TESTSETS_ROOT/../eval/vi-en/ (Gadi convention); override with
--out_dir on any filesystem.

Usage:
    bin/download_vi_en_test_sets [--out_dir /path/to/eval/vi-en]
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import HF_CACHE

# Turn OFF offline mode for this download (only run on login/copyq/laptop)
for var in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
    os.environ.pop(var, None)

# Inherit HF cache from src.config (which respects SIMT_HF_CACHE / HF_HOME env).
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--out_dir", type=Path,
                default=Path(os.environ.get(
                    "SIMT_VIEN_EVAL_DIR",
                    str(Path(__import__("os").environ.get("SIMT_TESTSETS_ROOT", str(Path.home() / "data" / "simt-tor-26"))) / "eval" / "vi-en"),
                )),
                help="Output directory for the {iwslt15,phomt}.{vi-en,en-vi}.{src,ref} files.")
args, _ = ap.parse_known_args()
OUT_DIR = args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

from datasets import load_dataset
from huggingface_hub import snapshot_download

def write_pair(tag: str, en_lines, vi_lines):
    (OUT_DIR / f"{tag}.vi-en.src").write_text("\n".join(vi_lines) + "\n")
    (OUT_DIR / f"{tag}.vi-en.ref").write_text("\n".join(en_lines) + "\n")
    (OUT_DIR / f"{tag}.en-vi.src").write_text("\n".join(en_lines) + "\n")
    (OUT_DIR / f"{tag}.en-vi.ref").write_text("\n".join(vi_lines) + "\n")
    print(f"    wrote {tag}.{{vi-en,en-vi}}.{{src,ref}} ({len(en_lines)} sents)")

# ------------------------------------------------------------------
# 1. IWSLT'15 EN-VI — try the IWSLT/ namespace (current) and parse whatever
#    raw files are shipped (train.en, train.vi, tst2012.en, tst2012.vi,
#    tst2013.en, tst2013.vi — Luong+Manning's original Stanford release).
# ------------------------------------------------------------------
for repo in ["IWSLT/mt_eng_vietnamese"]:
    print("=" * 72)
    print(f"Snapshot download: {repo}")
    print("=" * 72)
    try:
        p = snapshot_download(repo_id=repo, repo_type="dataset")
        print(f"  Snapshot dir: {p}")
        import os as _os
        for root, dirs, files in _os.walk(p):
            for f in files:
                fp = _os.path.join(root, f)
                print(f"    {fp}  ({_os.path.getsize(fp)} bytes)")
        # Try to identify tst2012/tst2013 en/vi files
        found = {}
        for root, _, files in _os.walk(p):
            for f in files:
                if f.startswith("tst2012") and f.endswith(".en"): found["tst2012.en"] = _os.path.join(root, f)
                elif f.startswith("tst2012") and f.endswith(".vi"): found["tst2012.vi"] = _os.path.join(root, f)
                elif f.startswith("tst2013") and f.endswith(".en"): found["tst2013.en"] = _os.path.join(root, f)
                elif f.startswith("tst2013") and f.endswith(".vi"): found["tst2013.vi"] = _os.path.join(root, f)
        print(f"  Located test files: {list(found.keys())}")
        for year in ["tst2012", "tst2013"]:
            if f"{year}.en" in found and f"{year}.vi" in found:
                en_lines = Path(found[f"{year}.en"]).read_text().strip().split("\n")
                vi_lines = Path(found[f"{year}.vi"]).read_text().strip().split("\n")
                write_pair(f"iwslt15_{year}", en_lines, vi_lines)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

# ------------------------------------------------------------------
# 2. Helsinki-NLP/opus-100 en-vi — parquet, script-free, has test split
# ------------------------------------------------------------------
print()
print("=" * 72)
print("Downloading Helsinki-NLP/opus-100 en-vi")
print("=" * 72)
try:
    ds = load_dataset("Helsinki-NLP/opus-100", "en-vi")
    print(f"  Splits: {list(ds.keys())}")
    for split in ["test", "validation"]:
        if split not in ds:
            continue
        rows = ds[split]
        print(f"  {split}: {len(rows)} rows, cols: {rows.column_names}")
        en_lines, vi_lines = [], []
        for r in rows:
            tr = r["translation"]
            en_lines.append(tr["en"].strip())
            vi_lines.append(tr["vi"].strip())
        write_pair(f"opus100_{split}", en_lines, vi_lines)
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")

print()
print("=" * 72)
print("Final listing:")
print("=" * 72)
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size} bytes)")
