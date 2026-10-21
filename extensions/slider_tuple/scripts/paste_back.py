"""Paste the original's head and background into edited images; write a tuple dataset.

Input: a folder with one subfolder per member level plus the originals, files
matched by name, e.g.

    weight_slider_data/heavy/<name>.png   weight_slider_data/thin/<name>.png
    weight_slider_data/orig/<name>.png

For every name, each member is registered to the original, gets the
original's head (aligned onto the member's head; see paste_back.py) and
background, and keeps its own body. Everything is written in the original's
frame:

    <out>/<member>/<name>.png   pasted members
    <out>/original/<name>.png   the untouched original ("original" in the manifest)
    <out>/masks/<name>.png      one loss mask per tuple (body union 1, head 0, background)
    <out>/tuples.jsonl          manifest (levels from --member name=level)
    <out>/report.csv            per member: registration, head fit, whether the head was pasted
    <out>/preview/<name>.jpg    raw vs pasted, the heavy-thin difference, and face close-ups

Images already written are reused unless --overwrite, so re-running with new
captions only rewrites the manifest. Captions: <captions>/<name>.txt, or
<captions>/<name>.json (the string under --caption-key, default caption_long),
else --default-caption (counted and reported).

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/paste_back.py datasets/weight_slider_data \\
        datasets/weight_slider_tuples --member heavy=1 --member thin=-1 --original orig --preview
"""

import argparse
import csv
import json
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from extensions.slider_tuple.paste_back import (  # noqa: E402
    bbox, feather, max_shift, paste_member, register_member, resample,
)
from extensions.slider_tuple import captions  # noqa: E402
from extensions.slider_tuple.tuple_masks import compose, head_ellipse, load_mask_models, to_png_array, union  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="folder with the member subfolders and the originals")
    p.add_argument("out")
    p.add_argument("--member", action="append", required=True, metavar="NAME=LEVEL",
                   help="member subfolder and its level, e.g. heavy=1 (repeat per member)")
    p.add_argument("--original", default="orig", help="subfolder with the originals")
    p.add_argument("--captions", help="folder of <name>.txt or <name>.json captions")
    p.add_argument("--caption-key", default="caption_long", help="field to read from .json captions")
    p.add_argument("--default-caption", default="a photo of a woman")
    p.add_argument("--background", type=float, default=0.2, help="loss mask weight outside body and head")
    p.add_argument("--dilate", type=int, default=0, help="body dilation px before blending (0 = 2%% of the short side)")
    p.add_argument("--seam", type=float, default=0, help="seam blur px (0 = 0.4%% of the short side)")
    p.add_argument("--head-margin", type=float, default=0.15)
    p.add_argument("--kp-score", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    members = []
    for spec in args.member:
        name, sep, level = spec.partition("=")
        if not sep:
            p.error(f"--member must be NAME=LEVEL, got {spec!r}")
        members.append((name, float(level)))
    args.members = members
    return args


def find(folder, stem):
    for ext in IMAGE_EXTS:
        path = os.path.join(folder, stem + ext)
        if os.path.exists(path):
            return path
    return None


def read_caption(folder, stem, key="caption_long"):
    try:
        return captions.read_caption(folder, stem, key)
    except ValueError as e:
        raise SystemExit(f"{e}; see --caption-key")


def head_mask_for(keypoints_fn, image, size, margin, min_score):
    kps = keypoints_fn(Image.fromarray(image))
    return None if kps is None else head_ellipse(kps, size, min_score=min_score, margin=margin)


def process(stem, args, matte_fn, keypoints_fn):
    """Pasted members (orig frame), tuple loss mask, per-member report rows, preview data."""
    original = np.asarray(Image.open(find(os.path.join(args.root, args.original), stem)).convert("RGB"))
    h, w = original.shape[:2]
    short = min(h, w)
    # the seam sits this far outside the bodies, so the blend band only ever sees background
    dilate = args.dilate or max(4, round(0.02 * short))
    seam = args.seam or max(2.0, 0.004 * short)

    orig_body = matte_fn(Image.fromarray(original))
    orig_head = head_mask_for(keypoints_fn, original, (w, h), args.head_margin, args.kp_score)
    orig_people = np.maximum(orig_body, 0 if orig_head is None else orig_head)
    orig_outside = 1.0 - feather(orig_people, dilate_px=3 * dilate)

    registered, bodies, rows = {}, {}, []
    for name, _ in args.members:
        member = np.asarray(Image.open(find(os.path.join(args.root, name), stem)).convert("RGB"))
        mh, mw = member.shape[:2]
        member_body = matte_fn(Image.fromarray(member))
        member_outside = 1.0 - feather(member_body, dilate_px=3 * dilate)
        alignment = register_member(original, member, orig_outside, member_outside)
        registered[name] = resample(member, alignment.matrix, (w, h))
        bodies[name] = resample(member_body, alignment.matrix, (w, h))
        rows.append({"name": stem, "member": name, "registration": alignment.method,
                     "registration_max_shift_px": round(max_shift(alignment.matrix, (mw, mh)), 2),
                     "registration_score": round(alignment.score, 4)})
    body_union = union([orig_body] + list(bodies.values()))  # for the loss mask

    pasted, heads = {}, []
    for (name, _), row in zip(args.members, rows):
        member_head = head_mask_for(keypoints_fn, registered[name], (w, h), args.head_margin, args.kp_score)
        candidates = [m for m in (member_head, orig_head) if m is not None]
        head = union(candidates) if candidates else np.zeros((h, w), np.float32)
        heads.append(head)
        # each member keeps only its own body (plus the original's), so no other
        # member's outline is left behind in its background
        keep = np.maximum(bodies[name], orig_body)
        result = paste_member(original, registered[name], keep, head, dilate_px=dilate, seam_px=seam)
        pasted[name] = result.image
        fit = result.head_alignment
        center = None if bbox(head) is None else ((bbox(head)[0] + bbox(head)[2]) / 2, (bbox(head)[1] + bbox(head)[3]) / 2)
        row.update({
            "head_found": bool(candidates),
            "head_pasted": fit is not None,
            "head_method": "" if fit is None else fit.method,
            "head_ncc": "" if fit is None else round(fit.score, 4),
            "head_shift_px": "" if fit is None or center is None else
            "%.2f,%.2f" % fit.shift_at(*center),
        })
    loss_mask = compose(body_union, union(heads), background=args.background)
    return original, registered, pasted, loss_mask, union(heads), rows


def preview(path, stem, original, registered, pasted, head):
    """Row 1: each member raw (registered) and pasted, plus |pasted_a - pasted_b| x4.
    Row 2: face close-ups: original, then each member raw and pasted."""
    height = 360
    fit = lambda a: Image.fromarray(a).resize((round(a.shape[1] * height / a.shape[0]), height))
    names = list(pasted)
    row1 = [fit(registered[n]) for n in names] + [fit(pasted[n]) for n in names]
    labels1 = [f"{n} raw" for n in names] + [f"{n} pasted" for n in names]
    if len(names) >= 2:
        diff = np.abs(pasted[names[0]].astype(np.float32) - pasted[names[1]].astype(np.float32)).mean(2)
        row1.append(fit(np.clip(diff * 4, 0, 255).astype(np.uint8)[..., None].repeat(3, 2)))
        labels1.append(f"|{names[0]} - {names[1]}| x4")
    row2, labels2 = [], []
    box = bbox(head, pad=0.4)
    if box is not None:
        x0, y0, x1, y1 = box
        face = lambda a: Image.fromarray(a[y0:y1, x0:x1]).resize((256, round(256 * (y1 - y0) / max(1, x1 - x0))))
        row2 = [face(original)] + [face(img) for n in names for img in (registered[n], pasted[n])]
        labels2 = ["original"] + [f"{n} {kind}" for n in names for kind in ("raw", "pasted")]
    rows = [(row1, labels1)] + ([(row2, labels2)] if row2 else [])
    width = max(sum(i.width for i in r) for r, _ in rows)
    sheet = Image.new("RGB", (width, sum(max(i.height for i in r) + 16 for r, _ in rows)), "black")
    draw, y = ImageDraw.Draw(sheet), 0
    for images, labels in rows:
        x = 0
        for img, label in zip(images, labels):
            sheet.paste(img, (x, y + 16))
            draw.text((x + 4, y + 2), label, fill="yellow")
            x += img.width
        y += max(i.height for i in images) + 16
    draw.text((width - 200, 2), stem[:30], fill="white")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path, quality=90)


def main(argv=None, models=None):
    args = parse_args(argv)
    orig_dir = os.path.join(args.root, args.original)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(orig_dir) if f.lower().endswith(IMAGE_EXTS))
    complete = [s for s in stems if all(find(os.path.join(args.root, n), s) for n, _ in args.members)]
    skipped = sorted(set(stems) - set(complete))
    out = os.path.abspath(args.out)
    for sub in [n for n, _ in args.members] + ["original", "masks"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    matte_fn = keypoints_fn = None
    report, lines, uncaptioned = [], [], []
    for i, stem in enumerate(complete, 1):
        outputs = [os.path.join(out, n, stem + ".png") for n, _ in args.members] + [os.path.join(out, "masks", stem + ".png")]
        if args.overwrite or not all(os.path.exists(p) for p in outputs):
            if matte_fn is None:
                matte_fn, keypoints_fn = models or load_mask_models(args.device)
            original, registered, pasted, loss_mask, head, rows = process(stem, args, matte_fn, keypoints_fn)
            for name, image in pasted.items():
                Image.fromarray(image).save(os.path.join(out, name, stem + ".png"))
            Image.fromarray(to_png_array(loss_mask)).save(os.path.join(out, "masks", stem + ".png"))
            if args.preview:
                preview(os.path.join(out, "preview", stem + ".jpg"), stem, original, registered, pasted, head)
            report += rows
            status = ", ".join(f"{r['member']}: head {'pasted' if r['head_pasted'] else 'KEPT'}" for r in rows)
            print(f"[{i}/{len(complete)}] {stem}: {status}", flush=True)
        src = find(orig_dir, stem)
        dst = os.path.join(out, "original", os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        caption = read_caption(args.captions, stem, args.caption_key) if args.captions else None
        if caption is None:
            caption = args.default_caption
            uncaptioned.append(stem)
        lines.append(json.dumps({
            "id": stem,
            "caption": caption,
            "original": os.path.relpath(dst, out),
            "mask": f"masks/{stem}.png",
            "members": [{"image": f"{n}/{stem}.png", "level": level, "role": "edit"} for n, level in args.members],
        }))

    with open(os.path.join(out, "tuples.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if report:
        with open(os.path.join(out, "report.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)
    kept = [f"{r['name']}/{r['member']}" for r in report if not r["head_pasted"]]
    print(f"wrote {len(lines)} tuples to {os.path.join(out, 'tuples.jsonl')}"
          + (f"; skipped (missing a member): {skipped}" if skipped else "")
          + (f"; heads NOT pasted (kept the edit's own): {kept}" if kept else "")
          + (f"; {len(uncaptioned)} use --default-caption: {uncaptioned}" if uncaptioned else ""))
    return report


if __name__ == "__main__":
    main()
