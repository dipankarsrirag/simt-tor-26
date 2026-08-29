"""Score all v6b flores extrinsic outputs with wmt22-comet-da (ref-based).

Reads: results/phase2/extrinsic/flores_stream_{PREFIX}_checkargmax_{LAT}_{DIR}_n50.json
For each JSON, reads `hyps`, `refs`, and the source sentences (re-loaded from
FLORES devtest since the eval JSONs don't cache the raw src). Computes COMET
per sentence + mean, writes:
  results/phase2/extrinsic/comet_scores_{PREFIX}.json

Runs inside the isolated /g/data/po67/dipankar/venvs/comet venv (comet 2.2.7).
Model checkpoint: /g/data/po67/dipankar/models/wmt22-comet-da/checkpoints/model.ckpt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXTR = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/extrinsic")
FLORES = Path("/g/data/ba39/dipankar/simul-mt/data/raw/flores200/flores200_dataset/devtest")
LANG_FILE = {
    "en": "eng_Latn.devtest", "de": "deu_Latn.devtest", "ar": "arb_Arab.devtest",
    "ru": "rus_Cyrl.devtest",  "vi": "vie_Latn.devtest",
}
LAT = ["low", "low_medium", "medium", "medium_high", "high"]
DIR = ["de-en", "en-de", "ar-en", "en-ar", "ru-en", "en-ru", "vi-en", "en-vi"]

CKPT = "/g/data/po67/dipankar/models/wmt22-comet-da/checkpoints/model.ckpt"
XLMR_LOCAL = "/g/data/po67/dipankar/models/xlm-roberta-large"


def _patch_xlmr_resolution():
    """Redirect XLM-R model-name lookups to the local snapshot dir.

    transformers>=4.55 breaks name-based cache resolution for the cached
    xlm-roberta-large tokenizer (falls through convert_slow_tokenizer with
    vocab_file=None). Redirecting the model name to a directory with all
    tokenizer files avoids the cache resolver entirely.

    Same idea as adm/scripts/score_eval_cometkiwi.py's _patch_infoxlm_resolution.
    """
    from transformers import (
        XLMRobertaTokenizer, XLMRobertaTokenizerFast,
        XLMRobertaModel, XLMRobertaConfig,
    )
    for cls in (XLMRobertaTokenizer, XLMRobertaTokenizerFast,
                XLMRobertaModel, XLMRobertaConfig):
        orig = cls.from_pretrained.__func__
        def make_patched(orig=orig):
            def patched(inner_cls, pretrained, *a, **kw):
                if pretrained == "xlm-roberta-large":
                    pretrained = XLMR_LOCAL
                return orig(inner_cls, pretrained, *a, **kw)
            return patched
        cls.from_pretrained = classmethod(make_patched())


def load_source_sentences(src_lang: str, n: int) -> list[str]:
    fp = FLORES / LANG_FILE[src_lang]
    with fp.open() as f:
        lines = [ln.rstrip("\n") for ln in f]
    return lines[:n]


def score_one_file(model, prefix: str, lat: str, direction: str, batch_size: int, n_suffix: str = "n50") -> dict | None:
    fp = EXTR / f"{prefix}_{lat}_{direction}_{n_suffix}.json"
    if not fp.exists():
        return None
    payload = json.loads(fp.read_text())
    hyps = payload.get("hyps") or []
    refs = payload.get("refs") or []
    if not hyps or not refs:
        return {"n": 0, "note": "empty hyps/refs"}
    src_lang = direction.split("-")[0]
    srcs = load_source_sentences(src_lang, len(hyps))
    if len(srcs) != len(hyps):
        return {"n": 0, "note": f"src/hyp length mismatch {len(srcs)}!={len(hyps)}"}
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    out = model.predict(data, batch_size=batch_size, gpus=1, progress_bar=False)
    scores = list(out["scores"])
    return {
        "n": len(scores),
        "comet_mean": float(sum(scores) / max(len(scores), 1)),
        "comet_system": float(out["system_score"]),
        "per_sent": [float(s) for s in scores],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", nargs="+", required=True,
                    help="e.g. flores_stream_v6bm3rb_checkargmax flores_stream_v6bmerged3_checkargmax")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_suffix", default="n50",
                    help="Input filename N suffix (e.g. 'n50', 'n1012'). The output "
                         "COMET-scores JSON also gets this suffix so scores at "
                         "different N don't overwrite each other.")
    args = ap.parse_args()

    _patch_xlmr_resolution()

    from comet import load_from_checkpoint
    print(f"loading {CKPT}", flush=True)
    model = load_from_checkpoint(CKPT)

    for prefix in args.prefixes:
        out = {}
        for lat in LAT:
            for d in DIR:
                key = f"{lat}__{d}"
                res = score_one_file(model, prefix, lat, d, args.batch_size, n_suffix=args.n_suffix)
                if res is None:
                    continue
                out[key] = res
                summary = f"n={res.get('n',0)} mean={res.get('comet_mean', float('nan')):.4f}" if res.get('n', 0) else res.get('note', '?')
                print(f"  {prefix}  {lat:>12s}  {d}  {summary}", flush=True)
        tag = prefix.replace('flores_stream_','').replace('_checkargmax','')
        suffix = f"_{args.n_suffix}" if args.n_suffix != "n50" else ""
        out_fp = EXTR / f"comet_scores_{tag}{suffix}.json"
        out_fp.write_text(json.dumps(out, indent=2))
        print(f"wrote {out_fp}", flush=True)


if __name__ == "__main__":
    main()
