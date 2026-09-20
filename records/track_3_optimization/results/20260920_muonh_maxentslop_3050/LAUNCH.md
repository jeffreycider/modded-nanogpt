# How the eight logs were made

Pod: Lambda 8×A100-SXM4-80GB (Prime Intellect pod muoff-8seed), driver 570.148.08, PyTorch 2.11.0+cu128 (CUDA 12.8), 2026-09-20 17:36–20:25 UTC.
Data: FineWeb10B shards fetched with `data/cached_fineweb10B.py` (103 train shards + validation).
Each seed, one after another on the idle pod, from the repository root:

```bash
for seed in 0 1 2 3 4 5 6 7; do
  /root/venv211/bin/torchrun --standalone --nproc_per_node=8 \
    records/track_3_optimization/results/20260920_muonh_maxentslop_3050/train_gpt_muonh_maxentslop.py \
    --seed "$seed" --train_steps 3125 || exit
done
```

`seedN.txt` is the script's own file log (it embeds the executed source, the environment line and every validation).
`summary.tsv` holds the eight losses at every validation boundary from 3000, their mean, and the statistic; computed by
`boxlogs/jstudy/pr_build_8seed.py` in the research repository from these logs.
`control_kmaxwell_A100_seed*.txt` are the earlier same-hardware runs of PR #359's script, unchanged, seeds 0–7 (console logs).
