"""Cross-backbone PPL probe for the multilingual v5 paper story.

Same PPL protocol as `phase2_probe_multilang_ppl.py` (FLORES-200 devtest,
100 sents × 10 directions, teacher-forced NLL), but sweeps 4 backbones:
  gemma-4-E2B  (2B, current anchor)
  gemma-4-E4B  (4B, EAST-scale replication candidate)
  Qwen3.5-2B   (2B, alternative small backbone)
  Qwen3.5-4B   (4B, alternative larger backbone)

Purpose: justify the self-annotation claim — the backbone knows enough of
each language pair for its own next-token distributions to be a meaningful
commit signal. If PPL is uniformly < 30 across all 10 directions × 4 backbones,
the annotator warrants across all four backbone choices.

Also verifies base-model status (no chat_template) so we know we're not
accidentally probing an instruct variant.

Output: results/phase2/probe_multilang_ppl_multibackbone.json
Wall: ~40 min total (10 min per backbone × 4).
"""
import json
import sys
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

REPO = Path(__file__).resolve().parents[1]
FLORES_DIR = Path("/g/data/ba39/dipankar/simul-mt/data/raw/flores200/flores200_dataset/devtest")
OUT = REPO / "results/phase2/probe_multilang_ppl_multibackbone_v2.json"

N_SENTS = 100

BACKBONES = [
    ("gemma-4-E2B",     "/g/data/po67/dipankar/models/gemma-4-E2B"),
    ("gemma-4-E4B",     "/g/data/po67/dipankar/models/gemma-4-E4B"),
    ("Qwen3.5-2B-Base", "/g/data/po67/dipankar/models/Qwen3.5-2B-Base"),
    ("Qwen3.5-4B-Base", "/g/data/po67/dipankar/models/Qwen3.5-4B-Base"),
]

LANG_TO_FILE = {
    "en": "eng_Latn.devtest",
    "de": "deu_Latn.devtest",
    "ar": "arb_Arab.devtest",
    "ru": "rus_Cyrl.devtest",
    "zh": "zho_Hans.devtest",
    "vi": "vie_Latn.devtest",
}

PAIRS = [("de", "en"), ("en", "de"),
         ("ar", "en"), ("en", "ar"),
         ("ru", "en"), ("en", "ru"),
         ("zh", "en"), ("en", "zh"),
         ("vi", "en"), ("en", "vi")]


def load_flores_lines(lang: str, n: int):
    path = FLORES_DIR / LANG_TO_FILE[lang]
    with open(path) as f:
        return [l.strip() for l in f.readlines()[:n]]


def measure_pair_ppl(model, tok, src_lang: str, tgt_lang: str, sep: str = "\n"):
    src_lines = load_flores_lines(src_lang, N_SENTS)
    tgt_lines = load_flores_lines(tgt_lang, N_SENTS)
    total_loss = 0.0
    total_tokens = 0
    n_sents_processed = 0
    for src, tgt in zip(src_lines, tgt_lines):
        src_ids = tok(src, add_special_tokens=False).input_ids
        sep_ids = tok(sep, add_special_tokens=False).input_ids
        tgt_ids = tok(tgt, add_special_tokens=False).input_ids
        if not tgt_ids:
            continue
        bos_id = tok.bos_token_id
        prefix = [bos_id] if bos_id is not None else []
        input_ids = prefix + src_ids + sep_ids + tgt_ids
        input_t = torch.tensor([input_ids], device="cuda", dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=input_t)
            logits = out.logits[0]
        prefix_len = len(prefix) + len(src_ids) + len(sep_ids)
        query_positions = list(range(prefix_len - 1, prefix_len - 1 + len(tgt_ids)))
        labels = torch.tensor(tgt_ids, device="cuda", dtype=torch.long)
        target_logits = logits[query_positions]
        loss = F.cross_entropy(target_logits.float(), labels, reduction="sum")
        total_loss += float(loss.item())
        total_tokens += len(tgt_ids)
        n_sents_processed += 1
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "n_sents": n_sents_processed,
        "total_tokens": total_tokens,
        "avg_nll": avg_loss,
        "ppl": ppl,
    }


def verdict(ppl: float) -> str:
    if ppl < 10: return "✅ excellent"
    if ppl < 30: return "✅ adequate"
    if ppl < 100: return "⚠️ marginal"
    return "❌ insufficient"


def load_backbone(model_path: str):
    """Load model, handling both plain and multimodal architectures."""
    cfg = AutoConfig.from_pretrained(model_path)
    mtype = getattr(cfg, "model_type", "")
    if mtype == "gemma3n":
        from transformers import Gemma3nForCausalLM
        model = Gemma3nForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda"
        )
    model.eval()
    return model


def main():
    all_results = {"config": {"n_sents": N_SENTS, "flores_dir": str(FLORES_DIR)},
                   "per_backbone": {}}
    for backbone_name, model_path in BACKBONES:
        print(f"\n{'='*72}")
        print(f"BACKBONE: {backbone_name}  ({model_path})")
        print(f"{'='*72}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_path)
        has_chat = tok.chat_template is not None
        base_status = "❌ INSTRUCT (chat_template present)" if has_chat else "✅ BASE (no chat_template)"
        print(f"  base-model check: {base_status}", flush=True)
        try:
            model = load_backbone(model_path)
            print(f"  model loaded", flush=True)
        except Exception as e:
            print(f"  ❌ LOAD FAILED: {type(e).__name__}: {e}", flush=True)
            all_results["per_backbone"][backbone_name] = {"error": str(e), "is_base": not has_chat}
            continue
        per_dir = []
        print(f"\n  {'direction':<10} {'n':>4} {'toks':>6} {'nll':>8} {'PPL':>8}   verdict")
        print(f"  {'-'*60}")
        for src, tgt in PAIRS:
            r = measure_pair_ppl(model, tok, src, tgt)
            r["verdict"] = verdict(r["ppl"])
            per_dir.append(r)
            print(f"  {src+'→'+tgt:<10} {r['n_sents']:>4} {r['total_tokens']:>6} {r['avg_nll']:>8.4f} {r['ppl']:>8.2f}   {r['verdict']}", flush=True)
        all_results["per_backbone"][backbone_name] = {
            "model_path": model_path,
            "is_base": not has_chat,
            "per_direction": per_dir,
        }
        # Free GPU
        del model
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nWrote {OUT}")

    # Summary matrix
    print(f"\n{'='*72}")
    print(f"SUMMARY — PPL across backbone × direction (100 sents FLORES-200 devtest)")
    print(f"{'='*72}")
    pair_labels = [f"{s}→{t}" for s, t in PAIRS]
    print(f"  {'backbone':<14} {'base?':<7} " + " ".join(f"{p:>7}" for p in pair_labels))
    for backbone_name, _ in BACKBONES:
        b = all_results["per_backbone"].get(backbone_name, {})
        if "error" in b:
            print(f"  {backbone_name:<14} {'?':<7} ERROR: {b['error']}")
            continue
        ppls = [f"{r['ppl']:>7.2f}" for r in b["per_direction"]]
        base_flag = "yes" if b.get("is_base") else "NO"
        print(f"  {backbone_name:<14} {base_flag:<7} " + " ".join(ppls))


if __name__ == "__main__":
    main()
