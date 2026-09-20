# How the eight logs were made

Pod: Lambda 8×A100-SXM4-80GB (Prime Intellect pod muoff-8seed), driver 570.148.08, PyTorch 2.11.0+cu128 (CUDA 12.8), 2026-09-20 20:25–23:12 UTC.
Data: FineWeb10B shards fetched with `data/cached_fineweb10B.py` (103 train shards + validation).
Each seed, one after another, from the repository root:

```bash
for seed in 0 1 2 3 4 5 6 7; do
  /root/venv211/bin/torchrun --standalone --nproc_per_node=8 \
    records/track_3_optimization/results/20260920_muon_maxentslop_3080/train_gpt_muon_maxentslop.py \
    --seed "$seed" --train_steps 3080 || exit
done
```

`seedN.txt` is the script's own file log (embedded source, the environment line, every validation). `summary.tsv` holds the eight
losses at every validation boundary from 3000, their mean and the statistic, computed by `boxlogs/jstudy/pr_build_8seed.py` in
the research repository from these logs.
