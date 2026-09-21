# MaxEntSlop momentum on the tuned Muon baseline — 3080 steps (n=3, eight-seed confirmation pending)

Collaborator: [Jeffrey Cheng](https://github.com/jeffreycider)

## Summary

This entry starts from [PR #357](https://github.com/KellerJordan/modded-nanogpt/pull/357) (K-Maxwell momentum on the tuned
Muon + aux AdamW baseline, result #36, 3160 steps) and changes two things:

1. From step 780, Muon's momentum direction is a weighted sum of the parameter's last 256 raw gradients (the MaxEntSlop
   kernel pair: early kernel of mean gradient age 90, late kernel of mean age 24.05, linear anneal over 2340 steps).
2. Muon's weight decay is 0.1 at the peak learning rate and follows the learning-rate schedule, instead of a constant 0.05.

The diff against PR #357's script is the diff of this branch. The configuration was developed and measured in our research
harness (same model, data, batch, schedule and aux AdamW as the script; three seeds per step count, Lambda 8×A100-80GB);
the logs here are those runs. A fresh eight-seed confirmation of this exact script is the next step.

| train steps | final val loss, seeds 0/1/2 | mean | (3.28 − mean)·√3 | first boundary below 3.28 |
|---|---|---|---|---|
| 3080 | 3.27924, 3.27718, 3.27573 | 3.27738 | 0.0045 | 3055, 3020, 3000 |
| 3100 | 3.27767, 3.27628, 3.27487 | 3.27627 | 0.0065 | 3045, 3025, 3010 |
| 3120 | 3.27687, 3.27505, 3.27393 | 3.27528 | 0.0082 | 3055, 3030, 3020 |

PR #357 reports 3160 steps (n=8, H100) as its first passing boundary on a 3250-step training clock. This configuration's
3080-step clock is 80 steps shorter than that boundary at three seeds; the eight-seed rule requires mean ≤ 3.27859 over eight
seeds, which three seeds cannot establish.

## Files

- `train_gpt_muon_maxentslop.py` — PR #357's script with the two changes above; `--seed S --train_steps N`.
- `A100_N{3080,3100,3120}_seed{0,1,2}.txt` — the harness logs (validation every 5 steps over the last 250). The script keeps PR #357's cadence (every 10 steps from 2900), so the fresh check reads its boundaries on that grid.
- The harness runs used the harness's own optimizer (eager weighted sum and Newton-Schulz); this script compiles those two pieces separately, so their arithmetic can differ in low bits. That is why the eight-seed check runs this exact script.
- With a 3080-step clock the cooldown starts earlier in absolute steps than in PR #357's 3160-step runs; the schedule formula is unchanged, the trajectory is not.
- `summary.tsv` — final loss, first passing boundary and the three-seed statistic per step count, computed from the logs.

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
