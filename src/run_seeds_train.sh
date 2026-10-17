#!/usr/bin/env bash
# Seed robustness + hybrid fold models (post hoc): B4/B5/E6 seeds 43, 44, then B4/B5 fold models (folds 1-4) for seeds 42-44.
# Two trainings at a time on the GPU. Logs: logs/run/train_<tag>.log
cd "$(dirname "$0")"
PY=../.venv/bin/python
{
  for s in 43 44; do for m in b4_cnn b5_crnn e6_cnn_balanced; do echo "$m $s"; done; done
  for s in 42 43 44; do for k in 1 2 3 4; do for m in b4_cnn b5_crnn; do echo "$m $s $k"; done; done; done
} | xargs -P 2 -L 1 sh -c 'tag=$(echo "$@" | tr " " "_"); [ -f ../logs/run/train_$tag.done ] || { '"$PY"' deep.py "$@" > ../logs/run/train_$tag.log 2>&1 && touch ../logs/run/train_$tag.done; }' _
echo ALLDONE
