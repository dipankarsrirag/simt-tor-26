"""
Probe EAST-8B tokenizer + model for compatibility with our streaming eval.

Checks:
  1. Which special tokens are added? (EOR, EOW, latency tokens?)
  2. Are the string names identical to ours (<|end-of-read|> etc.) or
     different (<|EOR|>, <|read_end|>, ...)?
  3. Does the tokenizer have a chat template? What role/content shape?
  4. Model config: hidden_size, num_hidden_layers (sanity 8B check).
  5. Can we run a 2-sentence streaming smoke? (offline generation to sanity
     the tag emission shape.)

Output: prints report; also writes _archive/results/gemma_2b_curated/east_8b_compat.json.

Usage:  python -u scripts/probe_east_8b_compat.py [--model_dir PATH]
Default model_dir: /g/data/po67/dipankar/models/EAST-8B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODEL_BASE, REPO_ROOT

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str,
                    default=str(MODEL_BASE / "EAST-8B"),
                    help="HF id or absolute path to the backbone to probe. "
                         "Default: $SIMT_MODEL_BASE/EAST-8B.")
    ap.add_argument("--output", type=str,
                    default=str(REPO_ROOT / "_archive" / "results" / "gemma_2b_curated" / "east_8b_compat.json"),
                    help="Where to write the probe report JSON.")
    args = ap.parse_args()

    print(f"=== EAST-8B compat probe ===")
    print(f"model_dir: {args.model_dir}")

    # 1) Config
    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=False)
    print(f"\n[config] model_type={cfg.model_type}")
    print(f"[config] hidden_size={getattr(cfg,'hidden_size','?')}  "
          f"n_layers={getattr(cfg,'num_hidden_layers','?')}  "
          f"vocab_size={getattr(cfg,'vocab_size','?')}")

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
    print(f"\n[tok] class={type(tok).__name__}")
    print(f"[tok] vocab_size={len(tok)}  "
          f"bos={tok.bos_token!r}  eos={tok.eos_token!r}  "
          f"pad={tok.pad_token!r}")
    # Special tokens map + added tokens
    print(f"[tok] special_tokens_map: {tok.special_tokens_map}")
    added = tok.get_added_vocab()  # dict token->id
    print(f"[tok] {len(added)} added tokens")
    for t, i in sorted(added.items(), key=lambda kv: kv[1]):
        print(f"    id={i:>6}  {t!r}")

    # 3) Look for EAST-style tokens by common naming
    candidates = [
        "<|end-of-read|>", "<|end-of-write|>",
        "<|end_of_read|>", "<|end_of_write|>",
        "<|EOR|>", "<|EOW|>", "<|eor|>", "<|eow|>",
        "<|low-latency|>", "<|medium-latency|>", "<|high-latency|>",
        "<|low_latency|>", "<|medium_latency|>", "<|high_latency|>",
        "<|latency-low|>", "<|latency-medium|>", "<|latency-high|>",
    ]
    print(f"\n[tok] Candidate-string presence:")
    for c in candidates:
        ids = tok(c, add_special_tokens=False).input_ids
        # A "clean" special token maps to exactly one id AND is in added vocab
        clean = len(ids) == 1 and c in added
        print(f"    {c!r:<32} → ids={ids}  in_added_vocab={c in added}  clean={clean}")

    # 4) Chat template
    print(f"\n[tok] chat_template present: {tok.chat_template is not None}")
    if tok.chat_template:
        print(f"[tok] chat_template (first 500 chars):")
        print(f"    {tok.chat_template[:500]}")

    # 5) Quick model load smoke (bf16, cpu-only for now — no GPU access from copyq)
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\n[model] Loading in bf16 on {device} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        # Try tiny generation with candidate prompt formats
        test_src = "The quick brown fox jumps over the lazy dog."
        # Try Llama-3 chat template with EAST-style instruction
        if tok.chat_template:
            messages = [
                {"role": "user",
                 "content": (
                     f"Translate the following text from English into German "
                     f"with medium latency.\n\n{test_src}"
                 )}
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"\n[gen] chat-template prompt (first 300):\n{prompt[:300]}")
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=64, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
            print(f"[gen] generated (first 300 chars, WITH specials): {gen[:300]!r}")
    else:
        print("\n[model] No GPU available on this node — skipping model load smoke.")

    # 6) Persist findings
    out = {
        "model_dir": args.model_dir,
        "model_type": cfg.model_type,
        "hidden_size": getattr(cfg, "hidden_size", None),
        "n_layers": getattr(cfg, "num_hidden_layers", None),
        "vocab_size_config": getattr(cfg, "vocab_size", None),
        "vocab_size_tokenizer": len(tok),
        "bos": tok.bos_token,
        "eos": tok.eos_token,
        "pad": tok.pad_token,
        "added_vocab_size": len(added),
        "added_tokens": {t: int(i) for t, i in added.items()},
        "candidate_presence": {c: (c in added) for c in candidates},
        "has_chat_template": tok.chat_template is not None,
        "chat_template_preview": (tok.chat_template[:500] if tok.chat_template else None),
    }
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWritten: {outp}")


if __name__ == "__main__":
    main()
