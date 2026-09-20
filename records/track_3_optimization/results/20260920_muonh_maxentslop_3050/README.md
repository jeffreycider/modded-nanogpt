# MaxEntSlop momentum on MuonH fast-slow decay — 3050 steps (n=8)

Collaborator: [Jeffrey Cheng](https://github.com/jeffreyscheng)

## Summary

This entry starts from [PR #359](https://github.com/KellerJordan/modded-nanogpt/pull/359) (K-Maxwell on
[PR #351](https://github.com/KellerJordan/modded-nanogpt/pull/351)'s MuonH fast-slow-decay trainer) and changes one thing:
from step 750, the first moment is no longer a mixture of six exponential moving averages but a weighted sum of the
parameter's last 256 gradients, the MaxEntSlop kernel pair. The kernel anneals linearly
from its start kernel (mean gradient age 90 steps) at step 750 to its end kernel (mean age 24 steps) at the last step.
Newton–Schulz, the hyperball projection, the parameter groups, the learning-rate schedule and the auxiliary AdamW are unchanged.
The diff against PR #359's script is the diff of this branch.

Eight consecutive seeds (0–7), fixed 3125-step clock, validation every 125 steps and every 5 steps from step 3000; the Track 3 statistic
(3.28 − mean) · √8 ≥ 0.004 first passes at **3050 steps**:

| boundary | mean val loss (n=8) | (3.28 − mean)·√8 | |
|---|---|---|---|
| 3040 | 3.27888 | 0.00316 | fails |
| 3050 | 3.27826 | 0.00493 | passes |
| 3065 | 3.27740 | 0.00737 | passes |
| 3125 | 3.27535 | 0.01316 | passes |

## Control on the same hardware

PR #359's script was run as is with the same eight seeds on the same pods (Lambda 8×A100-80GB, not the H100s of
PR #359's own logs). On this hardware its statistic first passes at 3070 (mean 3.27847), five steps after its H100 claim of 3065;
the kernel pair passes at 3050, twenty steps earlier on the same hardware:

| boundary | mean val loss (n=8) | (3.28 − mean)·√8 | |
|---|---|---|---|
| 3040 | 3.28024 | -0.00068 | fails |
| 3050 | 3.27960 | 0.00115 | fails |
| 3065 | 3.27873 | 0.00359 | fails |
| 3070 | 3.27847 | 0.00432 | passes |
| 3125 | 3.27661 | 0.00958 | passes |

Paired by seed at the common 3125-step boundary, MaxEntSlop minus K-Maxwell:
mean -0.00127, standard deviation 0.00047, t = -7.7 on 7 degrees of freedom; every seed is lower.

## Files

- `train_gpt_muonh_maxentslop.py` — PR #359's script with the first-moment change. Seeding (`--seed`, per-rank offsets) is PR #359's own and is unchanged; the control runs used PR #359's script as is.
- Provenance: the eight candidate runs executed the version of this script at commit 69a5f24 of this branch, which carried the two kernels as literal arrays; the current version solves them at startup and reproduces those arrays to 5e-12 (one float32 entry differs in its last bit). The file-only logs with the embedded source were not synced off the pods before termination; the `.txt` files here are the console streams. The fresh eight-seed check runs the current source and will attach its file logs.
- `A100_seed0..7.txt` — the eight candidate logs. `control_kmaxwell_A100_seed0..7.txt` — the eight control logs.
- `summary.tsv` — validation loss per seed at 3040, 3050, 3065 and 3125 steps and the first boundary below 3.28, for both arms.

## Method

For each MuonH matrix parameter, from step 750 (steps below 750 use the trainer's ordinary Nesterov momentum):

```python
history.push(g)                                                   # RawGradientHistory: the last 256 raw gradients (recorded from step 0)
kernel = interpolate_momentum_kernels(early_kernel, late_kernel,   # MomentumKernel: kernel[k] weights the gradient
                                      momentum_anneal_fraction(step))   #   k steps old; early/late solved from desiderata
direction = apply_momentum_kernel_to_raw_gradient_history(kernel, history)
update = muon_update_maxentslop(g, direction)                     # Newton-Schulz + aspect scale, unchanged
```

The two kernels are computed at startup by `solve_for_momentum_kernel_given_desiderata()`: the maximum-entropy weight vectors over ages 0..255 given
their mean age, their mean log(1+age), their two newest weights and an exact zero response to a period-2 alternation. The four
constraint values are the moments of the production MaxEntSlop schedule's own endpoint kernels. The solve reproduces the
literal weights the eight logged runs used to 5e-12 (one of the 512 float32 values differs in its last bit).
