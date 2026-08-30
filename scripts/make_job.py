#!/usr/bin/env python3
"""
Generate PBS job scripts (Gadi conventions by default; portable via env).

Ported from ../simul-mt/scripts/make_job.py with two differences:
  * Only gpuhopper and copyq queues are supported — see HOUSEKEEPING §6.
  * Caches point at the shared po67 gdata cache
    ($SIMT_HF_CACHE, defaulting to ~/.cache/huggingface).

Usage:
    python scripts/make_job.py \\
        --name annot_smoke \\
        --queue gpuhopper \\
        --ngpus 1 \\
        --walltime 00:30:00 \\
        --script "python src/annotator/annotate.py --config configs/annot_smoke.yaml" \\
        --output jobs/annot_smoke.pbs
"""

# Kept stdlib-only and py3.6-safe so it runs on the bare login-node python.

import argparse
import sys
from pathlib import Path

# Hardware specs per queue — verified from NCI documentation.
# cpus_per_gpu: enforced by scheduler on gpuhopper (12 * ngpus).
# mem_per_gpu_gb: usable memory per GPU (leave headroom under node total).
# jobfs_per_gpu_gb: local SSD per GPU.
# max_gpus: max GPUs per node — never span nodes for a single job.

QUEUE_SPECS = {
    "gpuhopper": {
        "gpu_model": "H200-141GB",
        "cpus_per_gpu": 12,
        "mem_per_gpu_gb": 240,
        "jobfs_per_gpu_gb": 120,
        "max_gpus": 4,
        "has_gpu": True,
    },
    # copyq: CPU-only, internet-enabled. HF downloads, WMT pulls, wandb sync.
    "copyq": {
        "gpu_model": None,
        "cpus_fixed": 1,
        "mem_gb_fixed": 8,
        "jobfs_gb_fixed": 100,
        "max_gpus": 0,
        "has_gpu": False,
    },
}

import os
PROJECT = os.environ.get("SIMT_PBS_PROJECT", "po67")
STORAGE = os.environ.get("SIMT_PBS_STORAGE", "gdata/ba39+gdata/po67")  # override per site
WORK_DIR = os.environ.get("SIMT_REPO_ROOT", str(Path(__file__).resolve().parents[1]))
# Venv activated by the PBS job. Override with SIMT_VENV.
VENV = os.environ.get("SIMT_VENV", str(Path.home() / ".venv")) + "/bin/activate"
LOG_DIR = f"{WORK_DIR}/logs"
# Shared HF/torch cache; override with SIMT_HF_CACHE.
HF_CACHE = os.environ.get("SIMT_HF_CACHE", str(Path.home() / ".cache" / "huggingface"))


def make_job(name, queue, ngpus, walltime, script, output=None, extra_modules=None):
    if queue not in QUEUE_SPECS:
        known = ", ".join(QUEUE_SPECS.keys())
        sys.exit(f"Unknown queue '{queue}'. Known queues: {known}")

    specs = QUEUE_SPECS[queue]

    if specs["has_gpu"]:
        if ngpus > specs["max_gpus"]:
            sys.exit(
                f"Queue {queue} supports max {specs['max_gpus']} GPUs per node. "
                f"Requested {ngpus}. Submit multiple jobs instead."
            )
        if ngpus < 1:
            sys.exit("ngpus must be >= 1 for GPU queues")
        ncpus_total = specs["cpus_per_gpu"] * ngpus
        mem_gb = specs["mem_per_gpu_gb"] * ngpus
        jobfs_gb = specs["jobfs_per_gpu_gb"] * ngpus
        ngpus_line = f"#PBS -l ngpus={ngpus}\n"
        header_line = f"# Queue: {queue} | GPU: {specs['gpu_model']} x{ngpus}"
    else:
        # copyq fixed allocation.
        ncpus_total = specs["cpus_fixed"]
        mem_gb = specs["mem_gb_fixed"]
        jobfs_gb = specs["jobfs_gb_fixed"]
        ngpus_line = ""
        header_line = f"# Queue: {queue} | CPU-only (data-mover / internet access)"

    modules = ["python3/3.10.4"] + (extra_modules or [])
    module_lines = "\n".join(f"module load {m}" for m in modules)

    # Cache exports — always point at the shared po67 gdata cache.
    # Non-copyq queues additionally get offline flags so a live HF lookup
    # cannot hang the job when the network is absent.
    cache_exports = f"""# Shared po67 cache — persists across scratch purges, shared with sibling projects.
export HF_HOME={HF_CACHE}
export HF_HUB_CACHE={HF_CACHE}/hub
export HUGGINGFACE_HUB_CACHE={HF_CACHE}/hub
export TRANSFORMERS_CACHE={HF_CACHE}/transformers
export HF_DATASETS_CACHE={HF_CACHE}/datasets
export TORCH_HOME={HF_CACHE}/torch
export TRITON_CACHE_DIR={HF_CACHE}/triton
export XDG_CACHE_HOME={HF_CACHE}
export PIP_CACHE_DIR={HF_CACHE}/pip
export TMPDIR={HF_CACHE}/tmp
mkdir -p $HF_HOME $HF_HUB_CACHE $TRANSFORMERS_CACHE $HF_DATASETS_CACHE \\
         $TORCH_HOME $TRITON_CACHE_DIR $PIP_CACHE_DIR $TMPDIR
"""

    if specs["has_gpu"]:
        offline_and_activate = f"""# gpuhopper has no internet — force HF offline mode.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd {WORK_DIR}
{module_lines}
source {VENV}

# Bind thread counts to the allocated CPU slice.
export OMP_NUM_THREADS=${{PBS_NCPUS:-{ncpus_total}}}
export MKL_NUM_THREADS=${{PBS_NCPUS:-{ncpus_total}}}
"""
    else:
        # copyq: keep the network live; do NOT set OFFLINE flags.
        offline_and_activate = f"""cd {WORK_DIR}
{module_lines}
source {VENV}
"""

    pbs = f"""#!/bin/bash
#PBS -N {name}
#PBS -P {PROJECT}
#PBS -q {queue}
#PBS -l ncpus={ncpus_total}
{ngpus_line}#PBS -l mem={mem_gb}GB
#PBS -l jobfs={jobfs_gb}GB
#PBS -l walltime={walltime}
#PBS -l storage={STORAGE}
#PBS -l wd
#PBS -j oe
#PBS -k oed
#PBS -o {LOG_DIR}/{name}.log

{header_line}
# Auto-calibrated: {ncpus_total} CPUs, {mem_gb}GB RAM, {jobfs_gb}GB jobfs

set -eo pipefail
mkdir -p {LOG_DIR}

{cache_exports}
{offline_and_activate}
{script}
"""
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(pbs)
        print(f"Written: {output}")
        if specs["has_gpu"]:
            print(f"  Queue:   {queue} ({specs['gpu_model']})")
            print(f"  GPUs:    {ngpus}")
        else:
            print(f"  Queue:   {queue} (CPU-only, internet-enabled)")
        print(f"  CPUs:    {ncpus_total}")
        print(f"  Memory:  {mem_gb}GB")
        print(f"  Jobfs:   {jobfs_gb}GB")
        print(f"  Submit:  qsub {output}")
    return pbs


def main():
    parser = argparse.ArgumentParser(description="Generate GADI PBS job scripts for simt-tor-26")
    parser.add_argument("--name", required=True, help="Job name")
    parser.add_argument("--queue", required=True, choices=list(QUEUE_SPECS.keys()))
    parser.add_argument("--ngpus", type=int, default=0, help="Number of GPUs (ignored for copyq)")
    parser.add_argument("--walltime", default="08:00:00", help="Walltime HH:MM:SS (default: 08:00:00)")
    parser.add_argument("--script", required=True, help="Shell command to run inside the job")
    parser.add_argument("--output", help="Output .pbs file path (prints to stdout if omitted)")
    parser.add_argument("--modules", nargs="*", default=[], help="Additional modules to load")
    args = parser.parse_args()

    result = make_job(
        name=args.name,
        queue=args.queue,
        ngpus=args.ngpus,
        walltime=args.walltime,
        script=args.script,
        output=args.output,
        extra_modules=args.modules,
    )
    if not args.output:
        print(result)


if __name__ == "__main__":
    main()
