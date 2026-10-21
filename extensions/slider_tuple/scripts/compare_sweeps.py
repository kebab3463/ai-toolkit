"""Side-by-side per-multiplier table for several slider sweeps (runs or checkpoints).

For each sweep folder (slider_sweep.py --out), per multiplier: the paired
measure change against m = 0 (delta), the share of images moving in the
direction of m (right_sign) and mean background / face LPIPS. Use it to
compare training runs on the same held-out prompts and seeds, or a run's
step checkpoints to see whether the effect is still growing.

Folders whose names end in a step number (..._000000250) are ordered by it,
with a folder without one (the final save) last; others keep the given order.

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/compare_sweeps.py \\
        output/sweeps/weight_v1_turbo output/sweeps/weight_v2_turbo
    .venv/bin/python extensions/slider_tuple/scripts/compare_sweeps.py output/sweeps/run_steps/* --csv steps.csv
"""

import argparse
import csv
import math
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from extensions.slider_tuple.sweep import per_multiplier, read_csv  # noqa: E402


def order(folders):
    """Step checkpoints by step, the final save (no step number) last; other names keep their order."""
    steps = [re.search(r"_(\d+)$", os.path.basename(os.path.normpath(f))) for f in folders]
    if not any(steps):
        return list(folders)
    return [f for _, f in sorted(zip(steps, folders), key=lambda x: (x[0] is None, int(x[0].group(1)) if x[0] else 0))]


def label(folder):
    return os.path.basename(os.path.normpath(folder))


def table(folders):
    """{label: {m: per-multiplier row}} for each folder with a sweep.csv."""
    out = {}
    for folder in folders:
        path = os.path.join(folder, "sweep.csv")
        if not os.path.exists(path):
            raise SystemExit(f"no sweep.csv in {folder}")
        out[label(folder)] = {r["m"]: r for r in per_multiplier(read_csv(path))}
    return out


def fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+")
    p.add_argument("--csv", help="also write the table as long-format CSV")
    args = p.parse_args(argv)

    data = table(order(args.folders))
    ms = sorted({m for rows in data.values() for m in rows})
    names = list(data)
    width = max(22, *(len(n) for n in names))
    lines = ["delta (right_sign) | lpips_bg / face_lpips, per multiplier",
             "     m  " + "  ".join(n[-width:].rjust(width) for n in names)]
    for m in ms:
        cells = []
        for n in names:
            r = data[n].get(m)
            if r is None or math.isclose(m, 0.0):
                cells.append(("n/a" if r is None else "ref").rjust(width))
                continue
            cells.append(f"{fmt(r['delta'], '+.3f')} ({fmt(r['right_sign'], '.2f')}) | "
                         f"{fmt(r['lpips_bg'], '.2f')}/{fmt(r['face_lpips'], '.2f')}".rjust(width))
        lines.append(f"{m:+6.2f}  " + "  ".join(cells))
    print("\n".join(lines))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sweep", "m", "n", "delta", "delta_sd", "right_sign",
                                                   "lpips_bg", "face_lpips"])
            writer.writeheader()
            for n in names:
                for m in ms:
                    if m in data[n]:
                        writer.writerow({"sweep": n, **data[n][m]})
    return data


if __name__ == "__main__":
    main()
