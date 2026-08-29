"""
Verify that the SFT loss actually flows through the EAST special tokens.

Loads a trained model + tokenizer, tokenises a real training string
(EAST-interleaved), computes per-position loss, and checks:
  * are the loss values at special-token positions non-zero?
  * how do they compare to loss at regular-content positions?

If special-token losses are ~0 or the labels are -100 there, we have a
data-collator masking bug and the model can't learn tags no matter how
long we train. If losses are non-zero and comparable to content losses,
the tags-not-emerging problem is a training-length issue (fix by scaling
data or epochs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch
import torch.nn.functional as F

from src.annotator.east_format import SPECIAL_TOKENS, EastRow, interleave
from src.constants import DATA_ROOT, REPO_ROOT

CORPUS = DATA_ROOT / "SiMT-De-En-660K" / "SiMT-De-En-660K.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--tokenizer_dir", type=Path,
                    default=REPO_ROOT / "results" / "phase2" / "tokenizer-extended")
    ap.add_argument("--idx", type=int, default=190712, help="Corpus row to probe.")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    special_ids = {t: tok.encode(t, add_special_tokens=False)[0] for t in SPECIAL_TOKENS}
    print(f"Special token ids: {special_ids}")

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).cuda().eval()

    with open(CORPUS) as f:
        rows = json.load(f)
    row = next(r for r in rows if r["index"] == args.idx)

    east_str = interleave(EastRow(
        source=row["source"], target=row["target"],
        src_lang=row["src_lang"], tgt_lang=row["tgt_lang"], latency=row["latency"],
        source_chunks=list(row["source_chunks"]), target_chunks=list(row["target_chunks"]),
    ))
    # Tokenize as trl SFTTrainer would (with special tokens + EOS).
    enc = tok(east_str, return_tensors="pt", add_special_tokens=True)
    # Manually add EOS to mirror trl.
    if enc.input_ids[0, -1].item() != tok.eos_token_id:
        eos_t = torch.tensor([[tok.eos_token_id]])
        enc["input_ids"] = torch.cat([enc.input_ids, eos_t], dim=1)
        enc["attention_mask"] = torch.cat([enc.attention_mask, torch.ones_like(eos_t)], dim=1)
    input_ids = enc.input_ids.cuda()
    print(f"\nTraining string: {east_str[:180]}...")
    print(f"Tokenised length: {input_ids.shape[1]}")

    # Standard next-token loss (this is what SFTTrainer computes with
    # completion_only_loss=False).
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0]  # (L, V)
    # Next-token prediction: logit at position t predicts token at t+1.
    shift_logits = logits[:-1]  # (L-1, V)
    shift_labels = input_ids[0, 1:]  # (L-1,)
    per_pos_loss = F.cross_entropy(shift_logits.float(), shift_labels, reduction="none")  # (L-1,)

    # Classify each shifted position by whether the target token is special.
    ids_list = input_ids[0].tolist()
    print(f"\nPer-position loss analysis (target = token at t+1):")
    special_losses, content_losses = [], []
    for t in range(len(per_pos_loss)):
        target = ids_list[t + 1]
        target_str = tok.decode([target], skip_special_tokens=False)
        loss_t = per_pos_loss[t].item()
        is_special = target in special_ids.values()
        (special_losses if is_special else content_losses).append(loss_t)

    def stats(name, xs):
        if not xs:
            print(f"  {name}: (none)")
            return
        xs_s = sorted(xs)
        print(f"  {name}: n={len(xs)}  mean={sum(xs)/len(xs):.3f}  "
              f"median={xs_s[len(xs)//2]:.3f}  min={xs_s[0]:.3f}  max={xs_s[-1]:.3f}")

    stats("SPECIAL-token targets", special_losses)
    stats("content-token targets", content_losses)

    print(f"\nDetailed: EAST tokens (target column) losses:")
    for t in range(len(per_pos_loss)):
        target = ids_list[t + 1]
        if target in special_ids.values():
            target_str = tok.decode([target], skip_special_tokens=False)
            print(f"  pos {t}: predict {target_str!r} loss={per_pos_loss[t].item():.3f}  "
                  f"top-1={tok.decode([shift_logits[t].argmax().item()], skip_special_tokens=False)!r}")


if __name__ == "__main__":
    main()
