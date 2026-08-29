#!/bin/bash
# v6b-CTRL (α=1) FULL FLORES eval — 5 PBS jobs, one per latency.
# Each PBS: 8 directions × N=1012 FLORES devtest (full split).
# Model: sft_multilingual_v6b_ctrl/final (best-by-eval-loss).
#
# Sizing: earlier N=50 ctrl job ran 8 dirs × 50 sents in ~15-20 min including
# 8 model-load overheads (~2-3 min each). N=1012 scales the generation
# portion 20× → est ~150-180 min wall / PBS. 4h walltime for margin.
#
# Metric focus: AL (Ma 2019) as primary, per user direction 2026-08-22.

set -eo pipefail
REPO=/g/data/ba39/dipankar/simt-tor-26
LATENCIES=("low" "low-medium" "medium" "medium-high" "high")
N=1012

for LAT in "${LATENCIES[@]}"; do
    LAT_TAG=$(echo "$LAT" | tr '-' '_')
    PBS_FILE="${REPO}/jobs/phase2_extrinsic_stream_v6bctrl_full_${LAT_TAG}.pbs"
    LOG_FILE="${REPO}/logs/phase2_extrinsic_stream_v6bctrl_full_${LAT_TAG}.log"

    cat > "$PBS_FILE" <<EOF
#!/bin/bash
#PBS -N v6bC_full_${LAT_TAG}
#PBS -P po67
#PBS -q gpuhopper
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=240GB
#PBS -l jobfs=40GB
#PBS -l walltime=04:00:00
#PBS -l storage=gdata/ba39+gdata/po67
#PBS -l wd
#PBS -j oe
#PBS -k oed
#PBS -o ${LOG_FILE}

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

MODEL=results/phase2/sft_multilingual_v6b_ctrl/final
FLORES=/g/data/ba39/dipankar/simul-mt/data/raw/flores200/flores200_dataset/devtest
OUT=results/phase2/extrinsic
LAT="${LAT}"
N=${N}

declare -A LANG_FILE=(
  [en]=eng_Latn.devtest
  [de]=deu_Latn.devtest
  [ar]=arb_Arab.devtest
  [ru]=rus_Cyrl.devtest
  [vi]=vie_Latn.devtest
)

DIRS=("de-en" "en-de" "ar-en" "en-ar" "ru-en" "en-ru" "vi-en" "en-vi")

for D in "\${DIRS[@]}"; do
    SRC=\${D%%-*}
    TGT=\${D##*-}
    SRC_FILE="\$FLORES/\${LANG_FILE[\$SRC]}"
    TGT_FILE="\$FLORES/\${LANG_FILE[\$TGT]}"
    OUT_FILE="\$OUT/flores_stream_v6bctrl_checkargmax_${LAT_TAG}_\${D}_n${N}.json"
    if [ -f "\$OUT_FILE" ]; then echo "SKIP: \$OUT_FILE"; continue; fi
    echo ""
    echo "===== v6b-ctrl FULL | check_argmax | latency='\$LAT' | \$D | \$N sents FLORES ====="
    python -u src/eval/extrinsic.py \\
        --model_dir "\$MODEL" \\
        --tokenizer_dir results/phase2/tokenizer-extended-v6 \\
        --dev_src "\$SRC_FILE" \\
        --dev_ref "\$TGT_FILE" \\
        --n_sentences \$N \\
        --src_lang "\$SRC" \\
        --tgt_lang "\$TGT" \\
        --use_chat_template \\
        --latency "\$LAT" \\
        --mode streaming --policy check_argmax \\
        --output "\$OUT_FILE"
done
EOF

    echo "Generated: $PBS_FILE"
    JOB_ID=$(qsub "$PBS_FILE" 2>&1)
    echo "  qsub → $JOB_ID"
done
