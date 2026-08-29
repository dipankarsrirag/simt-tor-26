#!/bin/bash
# Template — generates + qsubs 10 direction-specific annotator PBS files.
# Each direction gets its own short PBS job (~1h walltime, resume-capable,
# chain-at-start) — matches "hack the scheduler" preference.
#
# Usage:  bash jobs/phase2_annot_ot_multilang_TEMPLATE.sh
# Emits:  jobs/phase2_annot_ot_multilang_<direction>.pbs (10 files)
# Fires:  qsub each one.
#
# Directions annotated: de-en, en-de, ar-en, en-ar, ru-en, en-ru,
#                       zh-en, en-zh, vi-en, en-vi  (10 dirs — all bidirectional)
#
# All 10 directions are annotated fresh on the Multi-90K/TED2020 sources for
# corpus-consistency across the multilingual v5 dataset. v4's DE→EN matrices
# (SiMT-660K subset, results/phase2/annot_ot_n10k/matrices.jsonl) remain as
# a v4-only artefact and are NOT reused here — different underlying corpus.
#
# Prereq: results/phase2/multilingual_source_pool_v5_per_direction/<DIR>.json
#         (built by phase2_build_multilingual_source_pool.pbs)

set -eo pipefail
REPO=/g/data/ba39/dipankar/simt-tor-26
DIRECTIONS=("de-en" "en-de" "ar-en" "en-ar" "ru-en" "en-ru" "zh-en" "en-zh" "vi-en" "en-vi")

# Optional: control what to do
SUBMIT="${SUBMIT:-1}"   # set SUBMIT=0 to just generate PBS files without qsubbing
LOOKAHEAD_K="${LOOKAHEAD_K:-0}"   # 0 = current (D vs P_full); k>=1 = look-ahead

# Namespace suffix keeps k=0 artefacts (annot_ot_multi_<dir>) untouched
# while k>0 lands in annot_ot_multi_la<k>_<dir>.
if [ "$LOOKAHEAD_K" -gt 0 ]; then
    NS_SUFFIX="_la${LOOKAHEAD_K}"
    JOB_NS="la${LOOKAHEAD_K}_"
else
    NS_SUFFIX=""
    JOB_NS=""
fi

for DIR in "${DIRECTIONS[@]}"; do
    PBS_FILE="${REPO}/jobs/phase2_annot_ot_multilang${NS_SUFFIX}_${DIR}.pbs"
    LOG_FILE="${REPO}/logs/phase2_annot_ot_multilang${NS_SUFFIX}_${DIR}.log"
    OUT_DIR="${REPO}/results/phase2/annot_ot_multi${NS_SUFFIX}_${DIR}"
    INPUT_JSON="${REPO}/results/phase2/multilingual_source_pool_v5_per_direction/${DIR}.json"

    cat > "$PBS_FILE" <<EOF
#!/bin/bash
#PBS -N annot_ml_${JOB_NS}${DIR}
#PBS -P po67
#PBS -q gpuhopper
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=240GB
#PBS -l jobfs=20GB
#PBS -l walltime=01:30:00
#PBS -l storage=gdata/ba39+gdata/po67
#PBS -l wd
#PBS -j oe
#PBS -k oed
#PBS -o ${LOG_FILE}

# OT annotator for ${DIR} direction (multilingual v5).
# Reads pre-sampled source pool → writes matrices.jsonl in output_dir.
# Resume-safe; chain-at-start submits successor for walltime protection.
# Expected ~30-60 min for 10K rows on H200; MAX_SHARDS=3 ceiling.

set -eo pipefail
mkdir -p /g/data/ba39/dipankar/simt-tor-26/logs

export HF_HOME=/g/data/po67/dipankar/cache
export HF_HUB_CACHE=/g/data/po67/dipankar/cache/hub
export HUGGINGFACE_HUB_CACHE=/g/data/po67/dipankar/cache/hub
export TRANSFORMERS_CACHE=/g/data/po67/dipankar/cache/transformers
export HF_DATASETS_CACHE=/g/data/po67/dipankar/cache/datasets
export TORCH_HOME=/g/data/po67/dipankar/cache/torch
export TRITON_CACHE_DIR=/g/data/po67/dipankar/cache/triton
export XDG_CACHE_HOME=/g/data/po67/dipankar/cache
export PIP_CACHE_DIR=/g/data/po67/dipankar/cache/pip
export TMPDIR=/g/data/po67/dipankar/cache/tmp
mkdir -p \$HF_HOME \$HF_HUB_CACHE \$TRANSFORMERS_CACHE \$HF_DATASETS_CACHE \\
         \$TORCH_HOME \$TRITON_CACHE_DIR \$PIP_CACHE_DIR \$TMPDIR

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /g/data/ba39/dipankar/simt-tor-26
module load python3/3.10.4
source /scratch/po67/ds9561/.venv-fil/bin/activate

export OMP_NUM_THREADS=\${PBS_NCPUS:-12}
export MKL_NUM_THREADS=\${PBS_NCPUS:-12}

OUT_DIR=${OUT_DIR}
STATE_DIR=\$OUT_DIR/pbs_state
DONE_MARKER=\$STATE_DIR/DONE
mkdir -p "\$STATE_DIR"

if [ -f "\$DONE_MARKER" ]; then
    echo "SKIP: \$DONE_MARKER already present."
    exit 0
fi

# Chain-at-START — submit successor now so walltime-kill triggers resume.
SHARD_COUNTER=\$STATE_DIR/shard_counter
SHARD_N=\$(cat "\$SHARD_COUNTER" 2>/dev/null || echo 0)
SHARD_N=\$((SHARD_N + 1))
echo "\$SHARD_N" > "\$SHARD_COUNTER"
MAX_SHARDS=3

if [ "\$SHARD_N" -lt "\$MAX_SHARDS" ]; then
    NEXT_JOB=\$(qsub -W depend=afterany:\$PBS_JOBID ${PBS_FILE})
    echo "Shard \$SHARD_N of \$MAX_SHARDS. Next queued: \$NEXT_JOB"
fi

python -u scripts/phase1_tau_sweep.py \\
    --input_json ${INPUT_JSON} \\
    --criterion ot \\
    --taus 0.30 \\
    --model_path /g/data/po67/dipankar/models/gemma-4-E2B \\
    --output_dir \$OUT_DIR \\
    --lookahead_k ${LOOKAHEAD_K} \\
    --resume

# Mark DONE only if the DONE marker was written by the tau sweep itself
if [ -f "\$OUT_DIR/DONE" ]; then
    touch "\$DONE_MARKER"
    echo "Annotation complete — marker written to \$DONE_MARKER"
fi
EOF

    chmod +x "$PBS_FILE"
    echo "Generated: $PBS_FILE"
    if [ "$SUBMIT" = "1" ]; then
        # Skip submit if input JSON doesn't exist yet (source pool not built)
        if [ ! -f "$INPUT_JSON" ]; then
            echo "  ⚠ input JSON $INPUT_JSON not found — skipping qsub"
            continue
        fi
        # Skip submit if annotation already done
        if [ -f "${OUT_DIR}/pbs_state/DONE" ]; then
            echo "  SKIP: already done for $DIR"
            continue
        fi
        # Skip submit if a job for this direction is already Q/R/H — prevents
        # duplicate qsubs when re-running the template to just add missing dirs.
        if qstat -u "$USER" 2>/dev/null | grep -q "annot_ml_${JOB_NS}${DIR}[[:space:]]"; then
            echo "  SKIP: annot_ml_${JOB_NS}$DIR already in queue"
            continue
        fi
        JOB_ID=$(qsub "$PBS_FILE")
        echo "  qsub → $JOB_ID"
    fi
done

echo ""
echo "Done. To re-submit later without regenerating: bash $0"
echo "To generate PBS files without submitting: SUBMIT=0 bash $0"
