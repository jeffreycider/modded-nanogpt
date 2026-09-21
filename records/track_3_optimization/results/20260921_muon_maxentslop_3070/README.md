# MaxEntSlop momentum on the tuned Muon baseline — 3050 steps on a 3070-step clock (n=8)

Collaborator: [Jeffrey Cheng](https://github.com/jeffreycider)

## Summary

This entry starts from [PR #357](https://github.com/KellerJordan/modded-nanogpt/pull/357) (K-Maxwell momentum on the tuned
Muon + aux AdamW baseline, result #36) and changes two things:

1. From step 780, Muon's momentum direction is a weighted sum of the parameter's last 256 raw gradients: the MaxEntSlop kernel
   pair, two kernels solved at startup by maximum entropy from four desiderata each (mean gradient age 90 then 24.05, mean
   log-age, the two newest weights), interpolated linearly over 2340 steps. Before step 780 the momentum is the baseline's
   single-EMA Nesterov momentum.
2. Muon's weight decay is 0.1 at the peak learning rate and follows the learning-rate schedule, instead of a constant 0.05.

Model, data, batch size, learning-rate schedule, aux AdamW and Newton–Schulz are unchanged. The training clock is 3070 steps.

## Eight-seed result

Eight consecutive seeds (0–7) of this exact script, 3070-step clock, validation every 125 steps and every 10 steps from 2900,
on Lambda 8×A100-80GB (see `LAUNCH.md`). The Track 3 statistic (3.28 − mean) · √8 ≥ 0.004 first passes at **3050 steps** and
holds through the end of the clock:

| boundary | mean val loss (n=8) | (3.28 − mean)·√8 | |
|---|---|---|---|
| 3040 | 3.27869 | 0.00369 | fails |
| 3050 | 3.27817 | 0.00517 | passes |
| 3060 | 3.27780 | 0.00621 | passes |
| 3070 | 3.27751 | 0.00704 | passes |
| 3070 | 3.27738 | 0.00740 | passes |

Final losses at 3070 by seed: 3.27929, 3.27737, 3.27629, 3.27782, 3.27706, 3.27840, 3.27535, 3.27748. Every boundary from 3000 is in `summary.tsv`.

PR #357 reports 3160 steps (n=8, H100) as its first passing boundary on a 3250-step clock. This configuration's first passing
boundary is 3050 on a 3070-step clock, on A100s.

## Files

- `train_gpt_muon_maxentslop.py` — PR #357's script with the two changes; SHA256 5031c72defe6e54139814147d6727a66c382332e52fe7886e446d820138e5e59, identical to the source embedded in every `seedN.txt`.
- `seed0.txt` … `seed7.txt` — the eight file logs (embedded source, environment, every validation).
- `summary.tsv` — losses at every boundary from 3000, mean and statistic, computed from the logs.
- `LAUNCH.md` — hardware, environment and the exact commands.
- `README_harness_runs.md`, `harness_N3070_seed0..2.txt` — the earlier three-seed development runs of the same configuration in our research harness (not part of the claim).

## Method

```python
# from step 780 (before it, the baseline's Nesterov EMA momentum), per Muon matrix parameter:
history.push(g)                                                   # RawGradientHistory: the last 256 raw gradients
kernel = interpolate_momentum_kernels(early_kernel, late_kernel,   # MomentumKernel: kernel[k] weights the gradient
                                      momentum_anneal_fraction(step))   #   k steps old; early/late solved from desiderata
direction = apply_momentum_kernel_to_raw_gradient_history(kernel, history)
update = muon_update_maxentslop(g, direction)                     # Newton-Schulz + aspect scale, unchanged
p.mul_(1 - lr_t * 0.1 * eta_t); p.add_(update, alpha=-lr_t)        # decay follows the rate schedule
```
