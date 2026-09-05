#!/bin/bash
# Submit the 20 WMT22 eval cells for one tag: 4 directions x 5 latencies.
# Usage: bash jobs/eval/submit_wmt22_sweep.sh gemma_2b_m90k
set -eo pipefail
TAG=$1
Q="-l walltime=02:00:00 -l select=1:ncpus=8:mem=100gb:ngpus=1"
for LAT in low low-medium medium medium-high high; do
  for P in de-en en-de ru-en en-ru; do
    qsub $Q -v PAIR=$P,LAT=$LAT jobs/eval/stream_${TAG}.pbs
  done
done
