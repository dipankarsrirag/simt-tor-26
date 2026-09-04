"""Translate a test set with an untouched backbone, before any SFT.

Gives the floor a fine-tuned checkpoint should be compared against: if a
config's scores look low, this says whether the backbone could translate that
direction at all. Plain chat prompt, greedy decoding, offline (full source).

Note that src/eval/extrinsic.py --mode offline expects the v6 chat format and
the extended tokenizer, so it cannot score a stock backbone; this script talks
to the model directly instead.

Usage:
    python scripts/probe_base_backbone_mt.py \\
        --model_dir ${SIMT_MODEL_BASE}/Meta-Llama-3-8B-Instruct \\
        --src ${SIMT_TESTSETS_ROOT}/eval/en-ar/iwslt17.en-ar.src \\
        --ref ${SIMT_TESTSETS_ROOT}/eval/en-ar/iwslt17.en-ar.ref \\
        --src_lang English --tgt_lang Arabic --n_sentences 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "You are a helpful assistant."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--src_lang", required=True, help="language name, e.g. English")
    ap.add_argument("--tgt_lang", required=True, help="language name, e.g. Arabic")
    ap.add_argument("--n_sentences", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda().eval()

    srcs = args.src.read_text(encoding="utf-8").splitlines()[:args.n_sentences]
    refs = args.ref.read_text(encoding="utf-8").splitlines()[:args.n_sentences]

    # One greedy generation per sentence, full source in the prompt.
    hyps = []
    for i, src in enumerate(srcs):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate the following text from "
                                        f"{args.src_lang} into {args.tgt_lang}.\n\n{src}"},
        ]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True)["input_ids"].cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        hyp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        hyps.append(hyp.strip().replace("\n", " "))
        if i % 50 == 0:
            print(f"  [{i}/{len(srcs)}]", flush=True)

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    length_ratio = (sum(len(h.split()) for h in hyps)
                    / max(1, sum(len(r.split()) for r in refs)))
    print(f"BLEU = {bleu.score:.2f}  length ratio = {length_ratio:.3f}  n = {len(hyps)}")
    if args.output:
        args.output.write_text(json.dumps(
            {"model_dir": args.model_dir, "n_sentences": len(hyps),
             "bleu": bleu.score, "bleu_signature": str(bleu.get_signature()),
             "length_ratio": length_ratio, "hyps": hyps},
            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
