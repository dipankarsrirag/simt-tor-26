#!/bin/bash
# Submit the 55 streaming eval cells for one tag: 11 direction-sets x 5 latencies.
# Usage: bash jobs/eval/submit_stream_sweep.sh gemma_4b_curated
set -eo pipefail
TAG=$1
Q="-l walltime=02:00:00 -l select=1:ncpus=8:mem=100gb:ngpus=1"

sub () { qsub $Q -v TS=$1,PAIR=$2,LAT=$3 jobs/eval/stream_${TAG}.pbs; }

for LAT in low low-medium medium medium-high high; do
  sub wmt15 de-en $LAT
  for P in de-en en-de ru-en en-ru; do sub wmt22 $P $LAT; done
  for P in de-en en-de ar-en en-ar; do sub iwslt17 $P $LAT; done
  for P in vi-en en-vi; do sub iwslt15 $P $LAT; done
done
