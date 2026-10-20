"""Measure every member (and the original) of a tuple dataset with the body-width measure.

Checks whether the measure orders each tuple the way its levels say (e.g.
heavy > original > thin), and can write a manifest with the measured values
as levels (for slider.level_norm: median_span), optionally with the original
photo as an extra "real" member. See body_measure.py for what the score is.

Outputs (next to --out, or in --report-dir):
    measurements.csv            tuple, member, given level, score, why unmeasured
    measure_preview/<id>.jpg    with --preview: members side by side, band shaded, score

Usage (from the repo root):
    # check the measure on a dataset (levels unchanged)
    .venv/bin/python extensions/slider_tuple/scripts/measure_tuples.py \\
        datasets/weight_slider_tuples/tuples.jsonl --preview
    # write measured levels, with the original as a third member
    ... --out datasets/weight_slider_tuples/tuples_measured.jsonl --add-original

Tuples where any image cannot be measured are left out of --out (listed at the end).
Run split_holdout.py on the new manifest afterwards.
"""

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from extensions.slider_tuple.body_measure import draw_measurement, measure_body  # noqa: E402
from extensions.slider_tuple.manifest import rebase_entry  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--out", help="manifest to write with measured levels")
    p.add_argument("--add-original", action="store_true",
                   help="with --out: add each tuple's original as a role 'real' member at its measured level")
    p.add_argument("--report-dir", help="folder for measurements.csv / measure_preview (default: next to --out, "
                                        "else next to the manifest)")
    p.add_argument("--kp-score", type=float, default=1.0, help="minimum shoulder/hip keypoint score")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    if args.add_original and not args.out:
        p.error("--add-original needs --out")
    return args


def spearman(x, y):
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).correlation) if len(x) > 2 else None


def main(argv=None, models=None):
    args = parse_args(argv)
    src_dir = os.path.dirname(os.path.abspath(args.manifest))
    report_dir = args.report_dir or os.path.dirname(os.path.abspath(args.out or args.manifest))
    with open(args.manifest, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if models is None:
        from extensions.slider_tuple.tuple_masks import load_mask_models
        models = load_mask_models(args.device)
    matte_fn, keypoints_fn = models

    rows, out_entries, unmeasured = [], [], []
    for n, entry in enumerate(entries, 1):
        images = [(m["image"], m["level"], m.get("role", "edit")) for m in entry["members"]]
        if entry.get("original"):
            images.append((entry["original"], None, "original"))
        scores, previews = [], []
        for path, level, role in images:
            absolute = path if os.path.isabs(path) else os.path.join(src_dir, path)
            img = Image.open(absolute).convert("RGB")
            m = measure_body(matte_fn(img), keypoints_fn(img), min_score=args.kp_score)
            scores.append(m.score)
            rows.append({"id": entry["id"], "image": path, "role": role, "level": "" if level is None else level,
                         "score": "" if m.score is None else round(m.score, 5), "torso_px": round(m.torso_px, 1),
                         **{f"width_{k}": round(v, 4) for k, v in m.widths.items()}, "reason": m.reason})
            if args.preview:
                previews.append(draw_measurement(np.asarray(img), m))
        member_scores = scores[:len(entry["members"])]
        shown = " ".join(f"{lvl:+g}:{'n/a' if s is None else f'{s:.3f}'}" for (_, lvl, _), s in zip(images, scores)
                         if lvl is not None)
        orig = f" original:{scores[-1]:.3f}" if entry.get("original") and scores[-1] is not None else ""
        print(f"[{n}/{len(entries)}] {entry['id']}: {shown}{orig}", flush=True)

        if args.preview:
            h = min(p.shape[0] for p in previews)
            strip = np.concatenate([np.asarray(Image.fromarray(p).resize((round(p.shape[1] * h / p.shape[0]), h)))
                                    for p in previews], axis=1)
            os.makedirs(os.path.join(report_dir, "measure_preview"), exist_ok=True)
            sheet = Image.fromarray(strip)
            sheet.thumbnail((2400, 1200))
            sheet.save(os.path.join(report_dir, "measure_preview", f"{entry['id']}.jpg"), quality=88)

        needed = scores if args.add_original else member_scores
        if any(s is None for s in needed):
            unmeasured.append(entry["id"])
            continue
        if args.out:
            new = dict(entry, members=[dict(m, level=round(s, 5)) for m, s in zip(entry["members"], member_scores)])
            if args.add_original and entry.get("original"):
                new["members"].append({"image": entry["original"], "level": round(scores[-1], 5), "role": "real"})
            out_entries.append(rebase_entry(new, src_dir, os.path.dirname(os.path.abspath(args.out))))

    os.makedirs(report_dir, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(report_dir, "measurements.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # does the measure agree with the given levels?
    by_tuple = {}
    for r in rows:
        if r["score"] != "":
            by_tuple.setdefault(r["id"], []).append(r)
    n_members = {e["id"]: len(e["members"]) for e in entries}
    agree = total = 0
    for tid, rs in by_tuple.items():
        ranked = sorted((r for r in rs if r["level"] != ""), key=lambda r: r["level"])
        if len(ranked) == n_members[tid] >= 2:  # every member measured
            total += 1
            agree += all(a["score"] < b["score"] for a, b in zip(ranked, ranked[1:]))
    levels = [r["level"] for r in rows if r["level"] != "" and r["score"] != ""]
    scores = [r["score"] for r in rows if r["level"] != "" and r["score"] != ""]
    print(f"measure orders {agree}/{total} tuples like their levels; Spearman(level, score) over members = "
          f"{spearman(levels, scores) if levels else None}")
    between = [r for r in rows if r["role"] == "original" and r["score"] != ""]
    if between:
        inside = 0
        for r in between:
            ms = [x["score"] for x in by_tuple.get(r["id"], []) if x["level"] != ""]
            inside += bool(ms) and min(ms) <= r["score"] <= max(ms)
        print(f"original lies between its members in {inside}/{len(between)} tuples")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e) + "\n" for e in out_entries)
        print(f"wrote {len(out_entries)} tuples with measured levels to {args.out}")
    if unmeasured:
        print(f"not measurable (left out of --out): {unmeasured}")
    print(f"report: {os.path.join(report_dir, 'measurements.csv')}")
    return rows


if __name__ == "__main__":
    main()
