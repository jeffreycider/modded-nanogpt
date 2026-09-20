# MaxEntSlop momentum on MuonH fast-slow decay — 3050 steps (n=8)

Collaborator: [Jeffrey Cheng](https://github.com/jeffreycider)

## Summary

This entry starts from [PR #359](https://github.com/KellerJordan/modded-nanogpt/pull/359) (K-Maxwell on
[PR #351](https://github.com/KellerJordan/modded-nanogpt/pull/351)'s MuonH fast-slow-decay trainer) and changes one thing:
from step 750, the first moment is no longer a mixture of six exponential moving averages but a weighted sum of the
parameter's last 256 gradients, the MaxEntSlop kernel pair. The two kernels are solved at startup by maximum entropy
from four desiderata each (mean gradient age, mean log-age, the two newest weights; total weight 1 and zero response to a
period-2 alternation are always imposed); the kernel anneals linearly from the early kernel (mean age 90 steps) at step 750
toward the late kernel (mean age 24.05 steps), which the last executed update approaches but does not reach.
Newton–Schulz, the hyperball projection, the parameter groups, the learning-rate schedule and the auxiliary AdamW are unchanged.

## Eight-seed result

Eight consecutive seeds (0–7) of this exact script, fixed 3125-step clock, validation every 125 steps and every 5 steps from 3000,
on Lambda 8×A100-80GB (see `LAUNCH.md`). The Track 3 statistic (3.28 − mean) · √8 ≥ 0.004 first passes at **3050 steps**:

| boundary | mean val loss (n=8) | (3.28 − mean)·√8 | |
|---|---|---|---|
| 3040 | 3.27900 | 0.00284 | fails |
| 3045 | 3.27874 | 0.00356 | fails |
| 3050 | 3.27839 | 0.00455 | passes |
| 3055 | 3.27823 | 0.00500 | passes |
| 3065 | 3.27754 | 0.00695 | passes |
| 3070 | 3.27730 | 0.00764 | passes |
| 3125 | 3.27549 | 0.01274 | passes |

Final losses at 3125 by seed: 3.27500, 3.27452, 3.27517, 3.27727, 3.27553, 3.27570, 3.27424, 3.27653. Every boundary from 3000 is in `summary.tsv`.

## Control on the same hardware

PR #359's script, unchanged, was run earlier with the same eight seeds on the same hardware type (Lambda 8×A100-80GB;
PR #359's own logs are H100). On these A100s its statistic first passes at 3070 (mean 3.27847, statistic 0.00432; 3065 fails at
0.00359), five steps after its H100 claim of 3065. The kernel pair passes at 3050, twenty steps earlier on the same hardware.
Paired by seed at the common 3125-step boundary, this fresh run minus that control: mean −0.00112, standard deviation 0.00036,
every seed lower. Those control console logs are `control_kmaxwell_A100_seed*.txt`.

## Files

- `train_gpt_muonh_maxentslop.py` — PR #359's script with the first-moment change; SHA256 7de18f6a9eb79402a337a3935ca5b8c40340c1b2d5cec14562244520bd94a4bf, identical to the source embedded in every `seedN.txt`.
- `seed0.txt` … `seed7.txt` — the eight file logs (embedded source, environment, every validation).
- `summary.tsv` — losses at every boundary from 3000, mean and statistic, computed from the logs.
- `LAUNCH.md` — hardware, environment and the exact commands.
- `control_kmaxwell_A100_seed0.txt` … `seed7.txt` — PR #359's script on the same hardware type, seeds 0–7.

## Method

```python
# from step 750 (steps below 750 use the trainer's ordinary Nesterov momentum), per MuonH matrix parameter:
history.push(g)                                                   # RawGradientHistory: the last 256 raw gradients (recorded from step 0)
kernel = interpolate_momentum_kernels(early_kernel, late_kernel,   # MomentumKernel: kernel[k] weights the gradient
                                      momentum_anneal_fraction(step))   #   k steps old; early/late solved from desiderata
direction = apply_momentum_kernel_to_raw_gradient_history(kernel, history)
update = muon_update_maxentslop(g, direction)                     # Newton-Schulz + aspect scale, unchanged
```
