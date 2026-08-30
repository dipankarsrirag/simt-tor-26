"""Render the per-stage command list for `bin/run <config>`.

Reads a YAML experiment config, prints a series of shell command lines
grouped by pipeline stage. `bin/run` parses this output and executes
each stage.

Format (fixed, `bin/run` depends on it):
    ##STAGE_<N>_<STAGE_NAME>
    <command 1>
    <command 2>
    ##STAGE_<N>_<STAGE_NAME>
    ...

Stages emitted:
    1  build_source_pool
    2  annotate                (one command per language pair)
    3  build_sft_dataset
    4  train                   (uses torchrun when ngpus > 1)
    5  eval                    (one command per test-set × pair × latency)
    6  plot

The commands reference `${SIMT_REPO_ROOT}` and `${SIMT_MODEL_BASE}`, which
`bin/_env.sh` sets before invoking this script.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import load_config, resolve_backbone_path


def _sh(*parts) -> str:
    """Shell-quote each part and join with spaces."""
    return " ".join(shlex.quote(str(p)) for p in parts)


def render(cfg: dict, ngpus: int) -> list[str]:
    tag: str = cfg["tag"]
    backbone_path = str(resolve_backbone_path(cfg))
    annotator = cfg["annotate"].get("annotator", "same_as_backbone")
    if annotator == "same_as_backbone":
        annotator_path = backbone_path
        annotator_name = Path(backbone_path).name
    else:
        annotator_path = annotator
        annotator_name = Path(annotator).name

    corpus = cfg["source_pool"]["corpus"]
    dirs = list(cfg["source_pool"]["directions"].keys())
    tokenizer_dir = cfg["backbone"]["tokenizer_dir"]

    out: list[str] = []

    # ─── Stage 1 ───
    out.append("##STAGE_1_BUILD_SOURCE_POOL")
    out.append(_sh("bin/06_build_source_pool", "--config", cfg["__config_path"]))

    # ─── Stage 2 ───
    out.append("##STAGE_2_ANNOTATE")
    for pair in dirs:
        in_json = f"results/sft_dataset/{tag}/per_direction/{pair}.json"
        out_dir = f"results/annotate/{annotator_name}/{pair}"
        out.append(_sh(
            "bin/07_annotate",
            "--input_json", in_json,
            "--model_path", annotator_path,
            "--output_dir", out_dir,
            "--criterion", cfg["annotate"]["criterion"],
            "--taus", str(cfg["annotate"]["tau"]),
            "--lookahead_k", str(cfg["annotate"].get("lookahead_k", 0)),
            "--resume",
        ))

    # ─── Stage 3 ───
    out.append("##STAGE_3_BUILD_SFT_DATASET")
    matrix_glob = f"results/annotate/{annotator_name}/*/matrices.jsonl"
    dset_json = f"results/sft_dataset/{tag}/sft_dataset.json"
    src_pool = f"results/sft_dataset/{tag}/source_pool.json"
    args = [
        "bin/08_build_sft_dataset",
        "--matrices", matrix_glob,       # supports glob via shell expansion
        "--corpus_json", src_pool,
        "--tokenizer_path", tokenizer_dir,
        "--tau", str(cfg["annotate"]["tau"]),
        "--output", dset_json,
    ]
    if cfg["sft_dataset"].get("merge_small_chunks", False):
        args.append("--merge_small_chunks")
        args += ["--min_src_words", str(cfg["sft_dataset"].get("min_src_words", 2))]
    out.append(_sh(*args))

    # ─── Stage 4 ───
    out.append("##STAGE_4_TRAIN")
    train_out = f"results/train/{tag}"
    train_cfg = cfg["train"]
    launcher = (
        f"torchrun --nproc_per_node={ngpus}"
        if ngpus and ngpus > 1
        else f"{'${PYTHON}'} -u"
    )
    train_args = [
        "src/train/sft.py",
        "--corpus_file", dset_json,
        "--model_name",  backbone_path,
        "--tokenizer_dir", tokenizer_dir,
        "--output_dir", train_out,
        "--num_train_epochs", str(train_cfg["num_epochs"]),
        "--per_device_train_batch_size", str(train_cfg["per_device_batch_size"]),
        "--gradient_accumulation_steps", str(train_cfg["grad_accum_steps"]),
        "--learning_rate", str(train_cfg["learning_rate"]),
        "--warmup_steps", str(train_cfg["warmup_steps"]),
        "--eval_steps", str(train_cfg["eval_steps"]),
        "--save_steps", str(train_cfg["save_steps"]),
    ]
    if train_cfg.get("bf16", True):
        train_args.append("--bf16")
    out.append(launcher + " " + _sh(*train_args))

    # ─── Stage 5 ───
    out.append("##STAGE_5_EVAL")
    eval_out = f"results/eval/{tag}"
    eval_cfg = cfg["eval"]
    n_sentences = eval_cfg.get("n_sentences", -1)   # -1 = full test set
    for test_set, pairs in eval_cfg["test_sets"].items():
        for pair in pairs:
            for latency in eval_cfg["latencies"]:
                src_lang, tgt_lang = pair.split("-")
                # Test-set file convention (override via SIMT_TESTSETS_ROOT).
                # Layout: {eval_root}/{src-tgt}/{ds}.{pair}.{src,ref}
                src_file = f"${{SIMT_TESTSETS_ROOT:-${{SIMT_DATA_ROOT}}}}/eval/{src_lang}-{tgt_lang}/{test_set}.{pair}.src"
                ref_file = f"${{SIMT_TESTSETS_ROOT:-${{SIMT_DATA_ROOT}}}}/eval/{src_lang}-{tgt_lang}/{test_set}.{pair}.ref"
                out_json = f"{eval_out}/{test_set}_stream_{tag}_{eval_cfg['policy']}_{latency}_{pair}.json"
                out.append(_sh(
                    "${PYTHON}", "-u", "src/eval/extrinsic.py",
                    "--model_dir", f"{train_out}/final",
                    "--tokenizer_dir", tokenizer_dir,
                    "--dev_src", src_file,
                    "--dev_ref", ref_file,
                    "--n_sentences", str(n_sentences),
                    "--src_lang", src_lang,
                    "--tgt_lang", tgt_lang,
                    "--use_chat_template",
                    "--latency", latency,
                    "--mode", eval_cfg["mode"],
                    "--policy", eval_cfg["policy"],
                    "--output", out_json,
                ))

    # ─── Stage 6 ───
    out.append("##STAGE_6_PLOT")
    fig_dir = f"figures/{tag}"
    out.append(f"SIMT_EVAL_DIR={eval_out} SIMT_FIG_DIR={fig_dir} bin/09_plot_bleu_al")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ngpus", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["__config_path"] = str(args.config)
    print("\n".join(render(cfg, args.ngpus)))


if __name__ == "__main__":
    main()
