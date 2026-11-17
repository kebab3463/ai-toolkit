"""Set a manifest's member levels from your own measurements (a CSV).

For measures made outside this extension (e.g. mask areas, a classifier score):
one CSV row per tuple, one column per member, matched to members by the
member image's folder (or an explicit --column COLUMN=FOLDER mapping):

    stem,neg,pos                      tuples.jsonl member images
    000292,382930,452156      ->      thin/000292.png   (column neg)
                                      heavy/000292.png  (column pos)

Each member's level becomes transform(value). Only differences within a tuple
matter (multipliers are centered per tuple), so with ``level_norm: median_span``
a tuple trains at multipliers +-(its span / 2) / kappa, kappa = the median span:
a tuple whose edit changed the concept twice as much as a typical one trains
the LoRA at twice the strength, and m = 1 is one median tuple's change.

Transforms:
    log      log(value): spans are log ratios, independent of image size and
             framing (areas, counts and widths scale with the photo); the default
    linear   value as is: for measures already comparable across images
             (e.g. a score in fixed units)

Tuples without a CSV row, or with a missing / non-positive (log) value, are left
out and listed. --min-ratio drops tuples whose top/bottom value ratio is below
it (edits that barely changed anything); the trainer also drops spans under
slider.min_span x kappa.

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/levels_from_csv.py \\
        datasets/slider_tuples/unmeasured_tuples.jsonl datasets/slider_tuples/measurements.csv \\
        --column neg=thin --column pos=heavy --out datasets/slider_tuples/tuples.jsonl
Then split_holdout.py on the new manifest and train with level_norm: median_span.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from extensions.slider_tuple.manifest import rebase_entry  # noqa: E402

TRANSFORMS = {"log": math.log, "linear": float}


def read_csv(path, key):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if key not in reader.fieldnames:
            raise SystemExit(f"{path}: no '{key}' column (have {reader.fieldnames})")
        return reader.fieldnames, {row[key].strip(): row for row in reader}


def member_folder(member):
    return os.path.basename(os.path.dirname(os.path.normpath(member["image"])))


def column_map(pairs, fieldnames, key):
    """{folder: column}: from COLUMN=FOLDER pairs, else every non-key column names its folder."""
    if not pairs:
        return {name: name for name in fieldnames if name != key}
    mapping = {}
    for pair in pairs:
        column, sep, folder = pair.partition("=")
        if not sep or column not in fieldnames:
            raise SystemExit(f"--column {pair}: expected COLUMN=FOLDER with COLUMN one of {fieldnames}")
        mapping[folder] = column
    return mapping


def set_levels(entry, row, folders, transform):
    """The entry with measured levels, or (None, reason)."""
    out = json.loads(json.dumps(entry))
    values = []
    for member in out["members"]:
        folder = member_folder(member)
        if folder not in folders:
            return None, f"no column for member folder '{folder}'"
        raw = (row.get(folders[folder]) or "").strip()
        try:
            value = float(raw)
            member["level"] = round(transform(value), 6)
        except ValueError:
            return None, f"{folders[folder]}: bad value '{raw}'"
        values.append(value)
    if len({m["level"] for m in out["members"]}) == 1:
        return None, "all members measured the same"
    return out, max(values) / min(values) if min(values) > 0 else math.inf


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("csv")
    p.add_argument("--out", required=True, help="manifest to write")
    p.add_argument("--key", default="stem", help="CSV column holding the tuple id (default: stem)")
    p.add_argument("--column", action="append", metavar="COLUMN=FOLDER",
                   help="CSV column for the members in FOLDER (repeat; default: columns named like the folders)")
    p.add_argument("--transform", choices=sorted(TRANSFORMS), default="log")
    p.add_argument("--min-ratio", type=float, default=1.0, help="drop tuples whose max/min value ratio is below this")
    args = p.parse_args(argv)

    fieldnames, rows = read_csv(args.csv, args.key)
    folders = column_map(args.column, fieldnames, args.key)
    with open(args.manifest, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    kept, skipped, ratios = [], [], []
    for entry in entries:
        row = rows.get(str(entry["id"]))
        if row is None:
            skipped.append((entry["id"], "no CSV row"))
            continue
        out, info = set_levels(entry, row, folders, TRANSFORMS[args.transform])
        if out is None:
            skipped.append((entry["id"], info))
        elif info < args.min_ratio:
            skipped.append((entry["id"], f"ratio {info:.3f} < --min-ratio {args.min_ratio}"))
        else:
            kept.append(out)
            ratios.append(info)
    if not kept:
        raise SystemExit("no tuples measured; check --key / --column")

    src_dir = os.path.dirname(os.path.abspath(args.manifest))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    with open(args.out, "w", encoding="utf-8") as f:
        for entry in kept:
            entry = rebase_entry(entry, src_dir, out_dir)  # paths stay relative to the manifest
            f.write(json.dumps(entry) + "\n")

    spans = sorted(max(m["level"] for m in e["members"]) - min(m["level"] for m in e["members"]) for e in kept)
    kappa = statistics.median(spans)
    print(f"{len(kept)}/{len(entries)} tuples -> {args.out}")
    print(f"max/min value ratio: min {min(ratios):.3f}, median {statistics.median(ratios):.3f}, max {max(ratios):.3f}")
    print(f"spans ({args.transform}): min {spans[0]:.4f}, median {kappa:.4f} (= kappa with median_span), "
          f"max {spans[-1]:.4f}")
    print(f"with median_span, members train at +-{spans[0] / kappa / 2:.2f} .. +-{spans[-1] / kappa / 2:.2f} "
          f"(+-0.5 for a median tuple's two ends)")
    small = sum(s < 0.3 * kappa for s in spans)
    if small:
        print(f"{small} tuples have span < 0.3 kappa and will be dropped by the trainer's default min_span")
    for tuple_id, why in skipped:
        print(f"skipped {tuple_id}: {why}")
    return args.out


if __name__ == "__main__":
    main()
