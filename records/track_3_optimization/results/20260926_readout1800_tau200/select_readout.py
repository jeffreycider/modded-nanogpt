#!/usr/bin/env python3
"""Select one tail-readout tuple from the seed-0 and seed-1 bank outputs.

Rule (out/soap/bridge/readout_pr.md, AGREE line, 2026-09-26 13:1x PT):
  1. Eligible: the member's averaged loss is below the production readout's averaged loss
     (start 2400, tau 150, blend 0.6) on BOTH seeds at update 2595 AND at the member's
     selected boundary (paired, same trajectory, same seed).
  2. Selected boundary: the earliest dense update (2500..2900 step 5) at which the
     two-seed mean of the member's loss is <= 3.2785858 (= 3.28 - 0.004/sqrt(8)).
     A member that already passes at 2500 (left-censored) has no earlier coverage and
     is reported but not selectable.
  3. Rank eligible members by earliest boundary, then by the two-seed mean loss there,
     then lexicographically by (start, tau, blend).
Prints a table and writes the choice as JSON. Exit 0 with a choice, 3 with none.
"""
import json, sys

THRESH = 3.28 - 0.004 / (8 ** 0.5)


def load(path):
    d = json.load(open(path))
    assert d['status'] == 'READOUT_BANK_COMPLETE_CHECKED', d['status']
    assert d['native_parity'] is True
    return d


def main(p0, p1, out):
    b0, b1 = load(p0), load(p1)
    assert b0['seed'] == 0 and b1['seed'] == 1, (b0['seed'], b1['seed'])
    assert b0['dense_steps'] == b1['dense_steps']
    dense = b0['dense_steps']
    m0 = {m['name']: m for m in b0['members']}
    m1 = {m['name']: m for m in b1['members']}
    assert list(m0) == list(m1)
    prod = b0['production']
    assert prod == b1['production'] == 's2400_tau150_blend0.6', prod
    p0l, p1l = m0[prod]['dense_losses'], m1[prod]['dense_losses']
    rows, eligible = [], []
    for name in m0:
        l0, l1 = m0[name]['dense_losses'], m1[name]['dense_losses']
        mean = {str(s): 0.5 * (l0[str(s)] + l1[str(s)]) for s in dense}
        boundary = next((s for s in dense if mean[str(s)] <= THRESH), None)
        left = boundary == dense[0]
        d0_2595 = l0['2595'] - p0l['2595']
        d1_2595 = l1['2595'] - p1l['2595']
        if boundary is not None:
            d0_b = l0[str(boundary)] - p0l[str(boundary)]
            d1_b = l1[str(boundary)] - p1l[str(boundary)]
        else:
            d0_b = d1_b = float('nan')
        ok = (boundary is not None and not left and d0_2595 < 0 and d1_2595 < 0
              and d0_b < 0 and d1_b < 0)
        if name == prod:
            ok = False  # the production readout is the comparator, not a candidate
        row = dict(name=name, start=m0[name]['start'], tau=m0[name]['tau'], blend=m0[name]['blend'],
                   boundary=boundary, left_censored=left, mean_at_boundary=(mean[str(boundary)] if boundary else None),
                   d0_2595=d0_2595, d1_2595=d1_2595, d0_boundary=d0_b, d1_boundary=d1_b, eligible=ok)
        rows.append(row)
        if ok:
            eligible.append(row)
    prod_boundary = next((s for s in dense if 0.5 * (p0l[str(s)] + p1l[str(s)]) <= THRESH), None)
    eligible.sort(key=lambda r: (r['boundary'], r['mean_at_boundary'], r['start'], r['tau'], r['blend']))
    print(f"threshold {THRESH:.7f}; production {prod} two-seed boundary {prod_boundary}")
    print(f"{'member':24s} {'bnd':>5s} {'mean@bnd':>9s} {'d0@2595':>9s} {'d1@2595':>9s} {'d0@bnd':>9s} {'d1@bnd':>9s} elig")
    for r in sorted(rows, key=lambda r: (r['boundary'] or 9999, r['start'], r['tau'], r['blend'])):
        print(f"{r['name']:24s} {str(r['boundary']):>5s} {r['mean_at_boundary'] if r['mean_at_boundary'] else float('nan'):9.5f} "
              f"{r['d0_2595']:+9.5f} {r['d1_2595']:+9.5f} {r['d0_boundary']:+9.5f} {r['d1_boundary']:+9.5f} {'Y' if r['eligible'] else '-'}")
    choice = eligible[0] if eligible else None
    json.dump(dict(threshold=THRESH, production=prod, production_boundary=prod_boundary,
                   choice=choice, eligible=eligible, all=rows), open(out, 'w'), indent=1)
    if choice is None:
        print('NO_ELIGIBLE_MEMBER')
        return 3
    print(f"CHOICE {choice['name']} start={choice['start']} tau={choice['tau']} blend={choice['blend']} boundary={choice['boundary']}")
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:4]))
