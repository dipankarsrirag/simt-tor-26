"""
Loader smoke for Gemma-4-E2B-it.

Answers three questions before we write the annotator:
  1. Does AutoModelForCausalLM load Gemma-4 directly, or do we need the
     multimodal wrapper (Gemma4ForConditionalGeneration) and access
     `.language_model` under it?
  2. Does one teacher-forced forward pass on a short De-En pair return
     usable logits (shape and finite values)?
  3. Is `get_input_embeddings().weight` addressable — the annotator needs
     it for the OT ground cost.

CPU is fine for a one-shot 2B forward on <=64 tokens; login node OK.
"""

import sys
import time
import traceback

import torch
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")
from src.constants import PRIMARY_BACKBONE  # noqa: E402


MODEL_PATH = str(PRIMARY_BACKBONE)


def load_model():
    """Try CausalLM first; fall back to ConditionalGeneration and pull
    the language_model out. Report which path won."""
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        print("[load] AutoModelForCausalLM: OK")
        return model, "causal_lm"
    except Exception as e:
        print(f"[load] AutoModelForCausalLM refused: {type(e).__name__}: {e}")

    try:
        from transformers import Gemma4ForConditionalGeneration
        wrapper = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        print("[load] Gemma4ForConditionalGeneration: OK")
        return wrapper, "conditional_generation"
    except Exception as e:
        print(f"[load] Gemma4ForConditionalGeneration failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


def main():
    print(f"Model path: {MODEL_PATH}")

    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    print(f"[cfg] model_type={cfg.model_type}, text.vocab={cfg.text_config.vocab_size}, "
          f"text.hidden={cfg.text_config.hidden_size}, text.layers={cfg.text_config.num_hidden_layers}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"[tok] class={type(tok).__name__}, vocab={tok.vocab_size}, "
          f"pad_id={tok.pad_token_id}, eos_id={tok.eos_token_id}, bos_id={tok.bos_token_id}")

    t0 = time.time()
    model, mode = load_model()
    print(f"[load] path={mode} weights_loaded_in={time.time()-t0:.1f}s")

    # For text-only forward, grab the underlying language_model when the
    # wrapper is multimodal. AutoModelForCausalLM path returns it directly.
    if mode == "conditional_generation":
        # Gemma-4 multimodal wrappers expose .language_model or .model.
        lm = getattr(model, "language_model", None) or getattr(model, "model", None)
        if lm is None:
            raise RuntimeError("Wrapper has no .language_model / .model attribute")
        print(f"[lm] extracted inner LM: {type(lm).__name__}")
    else:
        lm = model

    lm.eval()

    # Input embedding matrix — required for OT ground cost.
    emb = lm.get_input_embeddings()
    W = emb.weight
    print(f"[emb] weight.shape={tuple(W.shape)}, dtype={W.dtype}")

    # Teacher-forced forward on a short De-En pair. Not a real chat template
    # — just enough to prove logits come out finite with the right shape.
    src = "Die Kommission kündigte gestern neue Maßnahmen an."
    tgt = "The Commission announced new measures yesterday."
    text = f"{src}\n{tgt}"
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"]
    print(f"[fwd] input tokens: {ids.shape[1]}")

    t0 = time.time()
    with torch.no_grad():
        out = lm(input_ids=ids)
    dt = time.time() - t0
    logits = out.logits if hasattr(out, "logits") else out[0]
    print(f"[fwd] logits.shape={tuple(logits.shape)} dtype={logits.dtype} in {dt:.2f}s")
    print(f"[fwd] finite?={torch.isfinite(logits).all().item()} "
          f"any_nan?={torch.isnan(logits).any().item()}")
    argmax_next = int(logits[0, -1].argmax().item())
    print(f"[fwd] argmax last position -> id={argmax_next} tok={tok.decode([argmax_next])!r}")

    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
