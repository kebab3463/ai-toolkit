"""Measure every image in a folder (body width) and optionally pick evenly spaced ones per group.

Made for interpolation sequences, e.g. video frames morphing a tuple's top
member into its bottom one, named <group>_<index>.png (the group is the tuple
id). Each image gets the body-width score (body_measure.py). Scores are
smoothed per group along the index (low-order polynomial; the measure's
per-image noise is ~0.03), and --pick K chooses K frames whose smoothed
scores are evenly spaced strictly between the group's two ends.

The ends are the group's first and last frame, or, with --top / --bottom,
the tuple's actual members (<dir>/<group>.png, e.g. the pass-through and thin
edits), so picks fall between the images they will sit between in a tuple.

Outputs in --out (default: <folder>_measure next to the folder):
    measurements.csv      every image: group, index, score, widths, why unmeasured
                          (also a cache: re-runs measure only new images)
    picks.csv             with --pick: group, rank, target, picked file, scores
    groups.csv            per group: ends, span, noise, monotonicity, flags
    preview/<group>.png   with --preview: score vs index with picks, thumbnails
    export/mid<k>/<group>.png  with --export: picked frames as member folders
                          (rank 1 nearest the top), the layout paste_back.py reads

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/measure_images.py datasets/weight_slider_tuples/mmh3 \\
        --top datasets/weight_slider_tuples/heavy --bottom datasets/weight_slider_tuples/thin \\
        --pick 2 --preview --export
"""

import argparse
import csv
import os
import re
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from extensions.slider_tuple.body_measure import LEVELS, measure_body  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_PATTERN = r"^(?P<group>.+)_(?P<index>\d+)$"
FIELDS = ["file", "group", "index", "role", "score", "torso_px"] + [f"width_{k}" for k in LEVELS] + ["reason"]


def group_files(folder, pattern=DEFAULT_PATTERN):
    """{group: [(index, path)] sorted by index}; names not matching the pattern are their own group."""
    regex = re.compile(pattern)
    groups = {}
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        match = regex.match(stem)
        group, index = (match.group("group"), int(match.group("index"))) if match else (stem, 0)
        groups.setdefault(group, []).append((index, os.path.join(folder, name)))
    return {g: sorted(v) for g, v in groups.items()}


def find_image(folder, stem):
    for ext in IMAGE_EXTS:
        path = os.path.join(folder, stem + ext)
        if os.path.exists(path):
            return path
    return None


def smooth(indices, scores, degree=3):
    """Polynomial fit of score vs index (degree capped by the number of points);
    returns (smoothed values at ``indices``, residual standard deviation)."""
    x, y = np.asarray(indices, float), np.asarray(scores, float)
    deg = int(min(degree, len(x) - 1))
    if deg < 1:
        return y.copy(), 0.0
    center, scale = x.mean(), max(np.ptp(x), 1e-9)
    coef = np.polyfit((x - center) / scale, y, deg)
    fitted = np.polyval(coef, (x - center) / scale)
    dof = max(len(x) - deg - 1, 1)
    return fitted, float(np.sqrt(((y - fitted) ** 2).sum() / dof))


def pick(smoothed, top, bottom, k):
    """Positions of k distinct entries whose values are closest to k evenly spaced
    targets strictly between ``top`` and ``bottom`` (rank 1 nearest the top)."""
    targets = [top + (bottom - top) * j / (k + 1) for j in range(1, k + 1)]
    chosen = []
    for target in targets:
        order = np.argsort(np.abs(np.asarray(smoothed) - target))
        chosen.append(next(int(i) for i in order if int(i) not in chosen))
    return targets, chosen


def spearman(x, y):
    from scipy.stats import spearmanr

    if len(x) < 3 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).correlation)


def load_cache(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def measure_paths(paths, cache, models, min_score, rel):
    """Rows for ``paths`` (path, group, index, role), measuring only those not in the cache."""
    from PIL import Image

    rows, new = [], 0
    for path, group, index, role in paths:
        key = rel(path)
        if key in cache and cache[key].get("role", "") == role:
            rows.append(cache[key])
            continue
        if models[0] is None:
            from extensions.slider_tuple.tuple_masks import load_mask_models
            models[:] = load_mask_models(models[2])
        img = Image.open(path).convert("RGB")
        m = measure_body(models[0](img), models[1](img), min_score=min_score)
        rows.append({"file": key, "group": group, "index": index, "role": role,
                     "score": "" if m.score is None else round(m.score, 5), "torso_px": round(m.torso_px, 1),
                     **{f"width_{k}": round(m.widths[k], 4) if k in m.widths else "" for k in LEVELS},
                     "reason": m.reason})
        new += 1
        if new % 25 == 0:
            print(f"measured {new} new images", flush=True)
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def preview(path, group, frames, smoothed, ends, picks, thumbs):
    """Score vs index (raw dots, smoothed line, targets, picks) above thumbnails of the ends and picks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_axes([0.07, 0.58, 0.9, 0.36])
    idx = [f["index"] for f in frames]
    ax.plot(idx, [f["score"] for f in frames], "o", color="0.55", ms=4, label="measured")
    ax.plot(idx, smoothed, "-", color="tab:blue", lw=2, label="smoothed")
    for name, value in ends.items():
        ax.axhline(value, color="tab:green" if name == "top" else "tab:red", ls="--", lw=1, label=name)
    for p in picks:
        ax.axhline(p["target"], color="0.8", lw=0.8)
        ax.plot([p["index"]], [p["smoothed"]], "s", color="tab:orange", ms=9)
    ax.set_title(group, fontsize=10)
    ax.set_xlabel("frame index")
    ax.set_ylabel("body width / torso")
    ax.legend(fontsize=8, loc="best")
    n = len(thumbs)
    for i, (label, image_path) in enumerate(thumbs):
        tax = fig.add_axes([0.02 + i * 0.96 / n, 0.02, 0.96 / n - 0.01, 0.48])
        tax.imshow(Image.open(image_path).convert("RGB"))
        tax.set_title(label, fontsize=8)
        tax.axis("off")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv=None, models=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--out", help="output folder (default: <folder>_measure)")
    p.add_argument("--pattern", default=DEFAULT_PATTERN, help="regex on file stems with named groups 'group', 'index'")
    p.add_argument("--top", help="folder with each group's top member, <group>.png (default: first frame)")
    p.add_argument("--bottom", help="folder with each group's bottom member (default: last frame)")
    p.add_argument("--pick", type=int, default=0, help="frames to pick per group, evenly spaced between the ends")
    p.add_argument("--degree", type=int, default=3, help="smoothing polynomial degree")
    p.add_argument("--kp-score", type=float, default=1.0)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--export", nargs="?", const="", default=None,
                   help="copy picks to <dir>/mid<k>/<group>.png (default dir: <out>/export)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    if bool(args.top) != bool(args.bottom):
        p.error("--top and --bottom go together")

    folder = os.path.normpath(args.folder)
    out = args.out or folder + "_measure"
    os.makedirs(out, exist_ok=True)
    rel = lambda path: (os.path.relpath(path, REPO_ROOT) if os.path.abspath(path).startswith(REPO_ROOT + os.sep)
                        else os.path.abspath(path))
    groups = group_files(folder, args.pattern)
    todo = [(path, g, i, "frame") for g, files in groups.items() for i, path in files]
    for role, ref_dir in (("top", args.top), ("bottom", args.bottom)):
        if ref_dir:
            for g in groups:
                path = find_image(ref_dir, g)
                if path is None:
                    print(f"{g}: no {role} image in {ref_dir}")
                else:
                    todo.append((path, g, "", role))
    cache_path = os.path.join(out, "measurements.csv")
    model_slot = [None, None, args.device] if models is None else [models[0], models[1], args.device]
    rows = measure_paths(todo, load_cache(cache_path), model_slot, args.kp_score, rel)
    write_csv(cache_path, rows, FIELDS)
    print(f"{len(rows)} images, {sum(r['score'] == '' for r in rows)} not measurable -> {cache_path}")

    group_rows, pick_rows = [], []
    for g, files in groups.items():
        frames = [dict(r, index=int(r["index"]), score=float(r["score"]))
                  for r in rows if r["group"] == g and r["role"] == "frame" and r["score"] != ""]
        refs = {r["role"]: float(r["score"]) for r in rows if r["group"] == g and r["role"] in ("top", "bottom")
                and r["score"] != ""}
        info = {"group": g, "frames": len(files), "measured": len(frames)}
        if len(frames) < 3:
            group_rows.append(dict(info, flags="too few measurable frames"))
            continue
        frames.sort(key=lambda f: f["index"])
        smoothed, noise = smooth([f["index"] for f in frames], [f["score"] for f in frames], args.degree)
        ends = ({"top": refs["top"], "bottom": refs["bottom"]} if {"top", "bottom"} <= refs.keys()
                else {"top": float(smoothed[0]), "bottom": float(smoothed[-1])})
        span = ends["top"] - ends["bottom"]
        rho = spearman([f["index"] for f in frames], [f["score"] for f in frames])
        flags = []
        if abs(span) < 3 * noise:
            flags.append(f"span {span:.3f} < 3 x noise {noise:.3f}: picks are within measurement noise")
        if rho is not None and abs(rho) < 0.8:
            flags.append(f"scores not monotone along the frames (Spearman {rho:.2f})")
        lo, hi = sorted(ends.values())
        if smoothed.min() > lo + 0.25 * (hi - lo) or smoothed.max() < hi - 0.25 * (hi - lo):
            flags.append("frames cover less than the range between the ends")
        group_rows.append(dict(info, top=round(ends["top"], 4), bottom=round(ends["bottom"], 4),
                               span=round(span, 4), noise=round(noise, 4),
                               spearman=None if rho is None else round(rho, 3), flags="; ".join(flags)))
        picks = []
        if args.pick:
            targets, chosen = pick(smoothed, ends["top"], ends["bottom"], args.pick)
            for rank, (target, i) in enumerate(zip(targets, chosen), 1):
                f = frames[i]
                picks.append({"group": g, "rank": rank, "target": round(target, 4), "file": f["file"],
                              "index": f["index"], "score": f["score"], "smoothed": round(float(smoothed[i]), 4),
                              "error": round(float(smoothed[i]) - target, 4)})
            pick_rows += picks
        if args.preview:
            thumbs = []
            if args.top and find_image(args.top, g):
                thumbs.append((f"top {ends['top']:.3f}", find_image(args.top, g)))
            thumbs += [(f"#{q['index']} {q['smoothed']:.3f}", os.path.join(REPO_ROOT, q["file"])
                        if not os.path.isabs(q["file"]) else q["file"]) for q in picks]
            if args.bottom and find_image(args.bottom, g):
                thumbs.append((f"bottom {ends['bottom']:.3f}", find_image(args.bottom, g)))
            if not thumbs:
                thumbs = [(f"#{frames[0]['index']}", files[0][1]), (f"#{frames[-1]['index']}", files[-1][1])]
            preview(os.path.join(out, "preview", f"{g}.png"), g, frames, smoothed, ends, picks, thumbs)
        status = f"span {span:+.3f}, noise {noise:.3f}" + (f" | {'; '.join(flags)}" if flags else "")
        chosen_txt = " picks " + ", ".join(f"#{q['index']} ({q['smoothed']:.3f})" for q in picks) if picks else ""
        print(f"{g}: {len(frames)} frames, {status}{chosen_txt}")

    write_csv(os.path.join(out, "groups.csv"), group_rows,
              ["group", "frames", "measured", "top", "bottom", "span", "noise", "spearman", "flags"])
    if args.pick:
        write_csv(os.path.join(out, "picks.csv"), pick_rows,
                  ["group", "rank", "target", "file", "index", "score", "smoothed", "error"])
        if args.export is not None:
            export = args.export or os.path.join(out, "export")
            for q in pick_rows:
                src = q["file"] if os.path.isabs(q["file"]) else os.path.join(REPO_ROOT, q["file"])
                dst = os.path.join(export, f"mid{q['rank']}", q["group"] + os.path.splitext(src)[1])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
            print(f"exported {len(pick_rows)} picks to {export}/mid<k>/")
    flagged = [r["group"] for r in group_rows if r.get("flags")]
    print(f"{len(group_rows)} groups, {len(flagged)} flagged (see groups.csv)" + (f": {flagged}" if flagged else ""))
    return group_rows, pick_rows


if __name__ == "__main__":
    main()
