# Record: Track 3 Optimization -- Tail-EMA readout started earlier with a longer horizon -- 2685 steps (n=8)

## TL;DR

Not a new optimizer. This is record #46 (PR [#328](https://github.com/KellerJordan/modded-nanogpt/pull/328): SOAP-Muon base
from #321 + tail-EMA eval readout from #325 + RowFloor + Cautious Weight Decay) with **two constants of the eval-time readout
changed** and nothing else:

| constant | record #46 | this PR |
|---|---:|---:|
| `TAILEMA_START` (step at which the eval-time EMA of the weights begins) | 2400 | **1800** |
| `TAILEMA_TAU` (EMA horizon, steps) | 150 | **200** |
| `TAILEMA_LAMBDA` (blend: eval weights = (1−λ)·w + λ·ema) | 0.6 | 0.6 |

Training is untouched: the readout is evaluation-only, the trained weights and every optimizer setting are those of #46.
The schedule is still 2900 steps; the reported step is the first validation boundary at which the eight-seed mean of the
readout's validation loss satisfies the track rule (3.28 − mean)·√8 ≥ 0.004, i.e. mean ≤ 3.2785858, with the preceding
five-step boundary failing.

Result (from the eight replication runs of the env-driven re-implementation; the filing runs of this script are in progress): **2685 steps**, eight-seed mean **3.27833** at that boundary (preceding boundary 2680: mean 3.27862, fails);
record #46 reports 2690. Final (step 2900) eight-seed mean 3.27204.

## How the two constants were chosen

A bank of 48 readouts (start ∈ {1800, 2000, 2200, 2400} × horizon ∈ {150, 200, 250, 300} × blend ∈ {0.5, 0.6, 0.7}) was evaluated
on the same two training trajectories (seeds 0 and 1 of the unmodified #46 trainer), every 5 steps from 2500 to 2900, each
readout kept as its own fp32 buffer alongside the run. Selection rule, written before the bank ran: a member is eligible only
if its loss is below the production readout (2400 / 150 / 0.6) on **both** seeds at step 2595 **and** at the member's own
boundary; among eligible members, the earliest boundary at which the two-seed mean is ≤ 3.2785858 wins, ties broken by the
mean loss there, then lexicographically. The winner was 1800 / 200 / 0.6: two-seed boundary 2675 versus 2685 for the production
readout, paired differences −0.00157 / −0.00156 at step 2595 (seed 0 / seed 1). Every readout starting at 1800 or 2000 beat the
production readout at 2595 by 0.0011–0.0017; differences at step 2900 are within ±0.0002, so the change moves the crossing,
not the final loss. Selection seeds 0 and 1 are disjoint from the filing seeds 2–9.

## Files

- `train_gpt_readout1800.py`: `train_gpt_cwd_SOTA.py` from record #46 with four line changes (the two constants above; the
  seed printed to stdout; validation every 5 steps from step 2500 so the first boundary is observed). sha256
  `866257dfc60a03f3d1b8f86a4de937d413b12c9ac40d9e3e1953c54d798f52e1`.
- `logs/pr_readout1800_s{2..9}.txt`: the eight filing runs of this script (8×H100, torch 2.14+cu130), seeds 2–9, launched as
  `torchrun --standalone --nproc_per_node=8 train_gpt_readout1800.py --seed <k>` with no environment overrides.
- `formal.txt`: the eight-seed rule computed from those logs (`formal_crossing.py`, included).
- `replication/`: eight earlier runs (seeds 2–9) of an env-driven re-implementation of the #46 trainer with the same two
  constants, used to pick the tuple before the filing runs; kept as cross-implementation evidence, not pooled with the filing runs.
- `selection/`: the two 48-member bank outputs (`bank_s0.checked.json`, `bank_s1.checked.json`) and the choice (`readout_choice.json`, `select_readout.py`).

## Hardware note

The filing runs were made on 8×H100. Record #46's logs are A40; the same #46 trainer run unchanged on this H100 setup crosses
at 2675 (seed 0, readout) versus 2660 on A40, so cross-hardware step differences of this size are not attributable to code.
The comparison that supports this PR is paired: the same trajectory read out two ways, on two seeds, plus the eight fresh seeds above.
