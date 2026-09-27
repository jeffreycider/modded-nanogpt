#!/usr/bin/env python3
"""Formal step-count rule over eight seed logs (modded-nanogpt rule as used for record #46):
the reported step is the first validation boundary at which (3.28 - mean over the eight seeds of
the reported validation loss) * sqrt(8) >= 0.004, i.e. mean <= 3.2785858. The reported loss is the
trainer's val_ema_loss (the blended tail-average readout that the record files).
Exit 0 only when a first passage is verified: eight distinct logs of seeds 2..9 with identical
complete dense grids (2500..2900 step 5), a crossing above 2500 whose preceding five-update
boundary fails. Exit 2 on malformed input, 3 when the rule is not met or not verifiable.
usage: formal_crossing.py log1 ... log8
"""
import re, sys, math
from pathlib import Path

PAT = re.compile(r'step:(\d+)/(\d+) val_loss:([0-9.]+)(?: val_avg_loss:([0-9.]+))? val_ema_loss:([0-9.]+)')
TARGET = 3.28 - 0.004 / math.sqrt(8)
DENSE = set(range(2500, 2901, 5))


def parse(path):
    rows = {}
    sched = None
    for line in Path(path).read_text(errors='replace').splitlines():
        if 'val_loss:' not in line:
            continue
        m = PAT.search(line)
        if m is None or (m.end() < len(line) and not line[m.end()].isspace()):
            raise ValueError('Malformed validation row')
        s, n, raw, avg, ema = int(m.group(1)), int(m.group(2)), float(m.group(3)), (float(m.group(4)) if m.group(4) is not None else float(m.group(3))), float(m.group(5))
        if n != 2900 or s in rows or not all(math.isfinite(v) for v in (raw, avg, ema)):
            raise ValueError('Invalid or duplicate validation row')
        sched = n
        rows[s] = (raw, ema)
    return sched, rows


def main(paths):
    if len(paths) != 8:
        print(f'need exactly 8 logs, got {len(paths)}'); return 2
    if len({Path(p).resolve() for p in paths}) != 8:
        print('DUPLICATE_LOG'); return 2
    seeds = [re.findall(r'Using seed=(\d+)', Path(p).read_text(errors='replace')) for p in paths]
    if seeds != [[str(k)] for k in range(2, 10)]:
        print('FILING_SEED_IDENTITY_MISMATCH', seeds); return 2
    try:
        data = [parse(p) for p in paths]
    except ValueError as e:
        print(f'MALFORMED_LOG {e}'); return 2
    scheds = {d[0] for d in data}
    if scheds != {2900}:
        print(f'schedule mismatch: {scheds}'); return 2
    if any(not DENSE <= set(d[1]) for d in data):
        print('INCOMPLETE_DENSE_GRID'); return 2
    if any(set(d[1]) != set(data[0][1]) for d in data[1:]):
        print('validation grids differ'); return 2
    steps = sorted(s for s in set.intersection(*[set(d[1]) for d in data]) if s >= 2500)
    print(f'target mean <= {TARGET:.7f}; boundaries common to all eight: {len(steps)} (last {steps[-1]})')
    first_ema = first_raw = None
    for s in steps:
        me = sum(d[1][s][1] for d in data) / 8
        mr = sum(d[1][s][0] for d in data) / 8
        if first_ema is None and me <= TARGET:
            first_ema = s
        if first_raw is None and mr <= TARGET:
            first_raw = s
    print(f'RAW_CROSSING {first_raw}')
    print('final means: ema %.5f raw %.5f' % (sum(d[1][2900][1] for d in data) / 8, sum(d[1][2900][0] for d in data) / 8))
    if first_ema is None:
        print('FORMAL_EMA_CROSSING none (mean never <= target)')
        return 3
    i = steps.index(first_ema)
    prev = steps[i - 1] if i > 0 else None
    mp = sum(d[1][prev][1] for d in data) / 8 if prev is not None else None
    print('per-seed val_ema_loss at the candidate boundary and the preceding boundary:')
    for p, d in zip(paths, data):
        print(f'  {p.split("/")[-1]:44s} ema@{first_ema}={d[1][first_ema][1]:.5f}'
              + (f' ema@{prev}={d[1][prev][1]:.5f}' if prev is not None else ''))
    if first_ema <= 2500 or prev != first_ema - 5 or mp is None or mp <= TARGET:
        print(f'FIRST_PASSAGE_UNVERIFIED candidate {first_ema} prev {prev} prev_mean {mp}')
        return 3
    me = sum(d[1][first_ema][1] for d in data) / 8
    print(f'FORMAL_EMA_CROSSING {first_ema} mean {me:.7f} ; preceding boundary {prev} mean {mp:.7f} FAILS')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
