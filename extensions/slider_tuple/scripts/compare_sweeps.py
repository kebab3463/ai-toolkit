"""Side-by-side per-multiplier table for several slider sweeps (runs or checkpoints).

For each sweep folder (slider_sweep.py --out), per multiplier: the paired
measure change against m = 0 (delta), the share of images moving in the
direction of m (right_sign) and mean background / face LPIPS. Use it to
compare training runs on the same held-out prompts and seeds, or a run's
step checkpoints to see whether the effect is still growing.

Folders are grouped by run (in the order given): a run's step checkpoints
(<run>_000000250, ...) by step, then its final save (<run>).

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

from extensions.slider_tuple.sweep import parity, per_multiplier, read_csv  # noqa: E402


def order(folders):
    """Grouped by run (first appearance), each run's step checkpoints by step with its final
    save (the folder named like the run, no step number) last."""
    runs, keys = [], []
    for f in folders:
        name = os.path.basename(os.path.normpath(f))
        match = re.search(r"^(.*)_(\d+)$", name)
        run, step = (match.group(1), int(match.group(2))) if match else (name, None)
        if run not in runs:
            runs.append(run)
        keys.append((runs.index(run), step is None, step or 0))
    return [f for _, f in sorted(zip(keys, folders), key=lambda x: x[0])]


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
    p.add_argument("--parity", action="store_true",
                   help="also print, per sweep (rows) and |m|, the odd (slider) and even (shared drift) parts")
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

    if args.parity:
        mags = sorted({m for rows in data.values() for m in rows if m > 0 and -m in rows})
        out = ["", "parity: odd / even part of delta, bg LPIPS at +m / -m (a linear slider has even ~0)",
               f"{'sweep':>{width}}" + "".join(f"  |m|={a:g}: odd   even   bg+  bg-" for a in mags)]
        for n in names:
            sym = {r["m"]: r for r in parity([data[n][m] for m in sorted(data[n])])}
            cells = []
            for a in mags:
                r = sym.get(a)
                if r is None:
                    cells.append(" " * 31 + "n/a")
                    continue
                cells.append(f"  {fmt(r['odd'], '+.3f'):>8} {fmt(r['even'], '+.3f'):>6} "
                             f"{fmt(r['lpips_bg_plus'], '.2f'):>5} {fmt(r['lpips_bg_minus'], '.2f'):>4}")
            out.append(f"{n[-width:]:>{width}}" + "".join(cells))
        print("\n".join(out))

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
