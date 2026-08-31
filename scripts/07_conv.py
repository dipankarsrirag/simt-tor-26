"""
Stage 2 alternative — Conversational SiMT chunking (Wang et al. 2024).

Produces an SFT-ready dataset JSON directly (same output schema as
`scripts/07_waitk.py` and `scripts/08_build_sft_dataset.py`). Stage 3
(`08_build_sft_dataset.py`) is skipped for conv-simt configs.

Recipe (Wang et al. 2024 §2):
    1. Run `awesome-align` on each parallel sentence to obtain per-word
       source↔target alignments.
    2. Segment source into fixed-size chunks of `k_cv` source words.
    3. For each source chunk, the corresponding target chunk includes
       every target word aligned to any source word in that chunk (in
       target-order; monotonicity is enforced post-hoc if needed).
    4. Format each (source_chunk, target_chunk) pair as one turn of a
       multi-turn dialogue for SFT.

STATUS: NOT YET IMPLEMENTED.

Blockers (must resolve before this runs):

  [1] `awesome-align` is not installed in the shared venv. Install path
      (do this on copyq, then commit to create-venv.sh):

          /g/data/po67/dipankar/uv/bin/uv pip install \\
              --python /scratch/po67/ds9561/.venv-fil/bin/python \\
              awesome-align

      Verify: `python -c 'import awesome_align'` returns no error.

  [2] `awesome-align` needs a base multilingual model (mBERT by default)
      under HF cache. Pull once on copyq:

          hf download bert-base-multilingual-cased \\
              --local-dir /g/data/po67/dipankar/models/mBERT

      Set the ALIGN_MODEL constant below to the on-disk path.

  [3] Wang 2024's exact chunking rule for target words that align to
      multiple source chunks (or none) is under-specified in the paper.
      Read Wang et al. 2024 §2.1 carefully and log the choice as a
      decision in LOG.md before implementing `alignment_to_chunks()`.

Usage (once implemented):
    bin/07_conv --config configs/04_east_8b_conv.yaml \\
                --output results/sft_dataset/east_8b_conv/sft_dataset.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import load_config


ALIGN_MODEL = "/g/data/po67/dipankar/models/mBERT"   # set once mBERT is pulled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["annotate"]["criterion"] != "conv-simt":
        sys.exit(f"config criterion is {cfg['annotate']['criterion']!r}, expected 'conv-simt'")

    raise NotImplementedError(
        "conv-simt is not yet implemented. See the module docstring for the "
        "3 blockers that must be resolved (awesome-align install, mBERT pull, "
        "Wang 2024 §2 recipe read-through). Config was validated OK: "
        f"tag={cfg['tag']}, k_cv={cfg['annotate'].get('k_cv')}, "
        f"latency_bins={cfg['annotate']['latency_bins']}."
    )


if __name__ == "__main__":
    main()
