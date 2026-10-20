"""Linearity sweep for a slider LoRA on Krea 2 (F6).

Renders held-out prompts x seeds x multipliers with the LoRA attached as
inference hooks (strength changes per image, no reload), then measures every
image and writes <out>/sweep.csv, <out>/summary.json, <out>/summary.txt and a
contact sheet per prompt (<out>/sheets/). See sweep.py for the metrics.

Defaults are the deployment path, Krea 2 Turbo: 8 steps, CFG off, fixed
schedule mu 1.15 (model_kwargs.schedule_mu). ``--raw`` switches to Raw-style
sampling (25 steps, CFG 4, resolution-dependent mu) for a Raw checkpoint.

Attribute measurement:
  --measure body     built-in body width in torso lengths (body_measure.py; weight sliders)
Callables (``module:function``, imported normally, so put them on PYTHONPATH):
  --measure  PIL.Image -> float or None (not measurable)   the attribute measurement
  --face     PIL.Image -> 1-D array or None (no face)   optional identity embedding
Built-in side effects: background LPIPS outside the person mattes and face
LPIPS on the head crops (Sapiens2 matting + Keypoint R-CNN heads).

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/slider_sweep.py \\
        --model /path/to/turbo_checkpoint_folder --lora output/run/run.safetensors \\
        --prompts held_out_prompts.txt --measure mymetrics:body_score --out output/sweeps/run
    # acceptance check: same LoRA with every delta zeroed -> slope ~0, LPIPS ~0, face sim ~1
    ... --zero-lora --out output/sweeps/run_zero
"""

import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO_ROOT, ".env"))

from extensions.slider_tuple.sweep import (  # noqa: E402
    DEFAULT_DIAGNOSTIC,
    DEFAULT_MULTIPLIERS,
    SweepRow,
    contact_sheet,
    cosine,
    head_crop_box,
    load_callable,
    masked_lpips,
    summarize,
    write_csv,
)


def floats(text):
    return [float(x) for x in text.split(",") if x.strip()] if text else []


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="krea/Krea-2-Turbo",
                   help="Krea 2 checkpoint: hub id, folder, .safetensors, or a ComfyUI int8_convrot .safetensors")
    p.add_argument("--lora", help="trained slider LoRA (.safetensors); omit to sweep the base model only")
    p.add_argument("--zero-lora", action="store_true", help="zero every LoRA delta after loading (acceptance check)")
    p.add_argument("--prompts", required=True, help="text file, one prompt per line, or a tuples.jsonl (captions)")
    p.add_argument("--seeds", default="1000,1001,1002")
    p.add_argument("--multipliers", default=",".join(str(m) for m in DEFAULT_MULTIPLIERS))
    p.add_argument("--diagnostic", default=",".join(str(m) for m in DEFAULT_DIAGNOSTIC))
    p.add_argument("--raw", action="store_true", help="Raw sampling: 25 steps, CFG 4, dynamic mu")
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg", type=float)
    p.add_argument("--mu", type=float)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--qtype", default="convrot8", help="DiT quantization ('none' for bf16)")
    p.add_argument("--measure", help="'body' (built-in body width, see body_measure.py) or module:function, "
                                     "PIL image -> float or None")
    p.add_argument("--face", help="module:function, PIL image -> embedding or None")
    p.add_argument("--no-lpips", action="store_true", help="skip background and face LPIPS (and their Sapiens2 / keypoint models)")
    p.add_argument("--lpips-res", type=int, default=512, help="images are resized to this for LPIPS")
    p.add_argument("--margin", type=float, default=0.05, help="flag side effects at m=-1 worse than at +1 by this")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.raw:
        args.steps = args.steps or 25
        args.cfg = 4.0 if args.cfg is None else args.cfg
    else:
        args.steps = args.steps or 8
        args.cfg = 1.0 if args.cfg is None else args.cfg  # Krea 2 CFG is 0-normalized: 1 = off
        args.mu = 1.15 if args.mu is None else args.mu
    if args.zero_lora and not args.lora:
        p.error("--zero-lora needs --lora")
    return args


def read_prompts(path):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line)["caption"] for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def image_path(out, prompt_idx, seed, m):
    return os.path.join(out, "images", f"p{prompt_idx:03d}_s{seed}_m{m:+.2f}.png")


def render(args, prompts, seeds, multipliers):
    """Render every missing image; returns rows with paths."""
    import torch

    from extensions.slider_tuple.turbo import is_comfy_quantized, load_krea2_dit
    from toolkit.config_modules import GenerateImageConfig, ModelConfig
    from toolkit.inference_lora import InferenceLoRA
    from toolkit.util.get_model import get_model_class

    rows = [SweepRow(prompt_idx=i, prompt=p, seed=s, m=m, path=image_path(args.out, i, s, m))
            for i, p in enumerate(prompts) for s in seeds for m in multipliers]
    todo = [r for r in rows if not os.path.exists(r.path)]
    print(f"{len(rows)} images, {len(rows) - len(todo)} already rendered")
    if not todo:
        return rows

    model_kwargs = {} if args.mu is None else {"schedule_mu": args.mu}
    quantize = args.qtype != "none"
    model_config = ModelConfig(
        name_or_path=args.model, arch="krea2", dtype="bf16", model_kwargs=model_kwargs,
        quantize=quantize, qtype=args.qtype if quantize else "qfloat8", quantize_te=True, qtype_te="qfloat8",
    )
    model_class = get_model_class(model_config)
    sd = model_class(device=args.device, model_config=model_config, dtype="bf16",
                     noise_scheduler=model_class.get_train_scheduler())
    if is_comfy_quantized(args.model):
        # e.g. a ComfyUI int8_convrot Turbo: strip its key prefix, keep the layers pre-quantized
        sd._load_transformer = lambda: load_krea2_dit(args.model, sd.torch_dtype)
    sd.load_model()

    lora = None
    if args.lora:
        lora = InferenceLoRA(args.lora).load(sd, status_fn=print)
        print(lora.report_line())
        if not lora.entries:
            raise SystemExit(f"no LoRA modules matched in {args.lora}")
        if args.zero_lora:
            for e in lora.entries:
                for name in ("A", "B", "diff", "diff_b"):
                    t = getattr(e, name)
                    if t is not None:
                        t.zero_()
        lora.attach()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    with torch.no_grad():
        for n, r in enumerate(todo, 1):
            if lora is not None:
                lora.strength = r.m
            print(f"[{n}/{len(todo)}] prompt {r.prompt_idx} seed {r.seed} m {r.m:+g}", flush=True)
            sd.generate_images([GenerateImageConfig(
                prompt=r.prompt, width=args.width, height=args.height, num_inference_steps=args.steps,
                guidance_scale=args.cfg, seed=r.seed, output_path=r.path,
            )], sampler="flowmatch")
    if lora is not None:
        lora.detach()
    del sd
    torch.cuda.empty_cache()
    return rows


def measure(args, rows, models=None):
    """Fill measure / lpips_bg / face_lpips / face_sim; the m=0 image is the reference per prompt and seed.

    ``models``: optional (lpips_fn, matte_fn, keypoints_fn) for tests; lpips_fn
    takes two (1, 3, H, W) tensors in [-1, 1] and returns a (1, 1, H, W) map
    (None: no LPIPS)."""
    import numpy as np
    import torch
    from PIL import Image

    from extensions.slider_tuple.body_measure import measure_body
    from extensions.slider_tuple.tuple_masks import head_ellipse

    body = args.measure == "body"  # built in: body width from the same matte / keypoints
    measure_fn = load_callable(args.measure) if args.measure and not body else None
    face_fn = load_callable(args.face) if args.face else None
    refs = {(r.prompt_idx, r.seed): r for r in rows if math.isclose(r.m, 0.0)}
    if len(refs) != len({(r.prompt_idx, r.seed) for r in rows}):
        raise SystemExit("every prompt/seed needs an m=0 render as the side-effect reference")

    lpips_fn = matte_fn = keypoints_fn = None
    if models is not None:
        lpips_fn, matte_fn, keypoints_fn = models
    elif not args.no_lpips or body:
        from extensions.slider_tuple.tuple_masks import load_mask_models
        if not args.no_lpips:
            import lpips
            net = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(args.device).eval()
            lpips_fn = lambda a, b: net(a.to(args.device), b.to(args.device))
        matte_fn, keypoints_fn = load_mask_models(args.device)

    size = (args.lpips_res, args.lpips_res)
    to_tensor = lambda img: torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    cache = {}

    def load(path):
        if path not in cache:
            img = Image.open(path).convert("RGB")
            entry = {"img": img}
            if lpips_fn is not None or body:
                matte, kps = matte_fn(img), keypoints_fn(img)
                if body:
                    entry["body"] = measure_body(matte, kps).score
                if lpips_fn is not None:
                    entry["tensor"] = to_tensor(img.resize(size, Image.BICUBIC))
                    small = Image.fromarray((np.clip(matte, 0, 1) * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
                    entry["person"] = np.asarray(small, dtype=np.float32) / 255.0
                    entry["head"] = None if kps is None else head_ellipse(kps, img.size, margin=0.15)
            if face_fn is not None:
                entry["face"] = face_fn(img)
            cache[path] = entry
        return cache[path]

    with torch.no_grad():
        for n, r in enumerate(rows, 1):
            cur, ref = load(r.path), load(refs[(r.prompt_idx, r.seed)].path)
            value = cur["body"] if body else measure_fn(cur["img"]) if measure_fn is not None else None
            r.measure = None if value is None or not math.isfinite(float(value)) else float(value)
            if lpips_fn is not None:
                dist = lpips_fn(cur["tensor"], ref["tensor"])[0, 0].float().cpu().numpy()
                r.lpips_bg = masked_lpips(dist, cur["person"], ref["person"])
                heads = [h for h in (cur["head"], ref["head"]) if h is not None]
                box = head_crop_box(heads) if heads else None
                if box is not None:
                    crop = lambda img: to_tensor(img.crop(box).resize((256, 256), Image.BICUBIC))
                    r.face_lpips = float(lpips_fn(crop(cur["img"]), crop(ref["img"])).mean())
            if face_fn is not None:
                r.face_sim = cosine(cur["face"], ref["face"])
            if n % 20 == 0:
                print(f"measured {n}/{len(rows)}", flush=True)
    unmeasured = sum(r.measure is None for r in rows) if (body or measure_fn) else 0
    if unmeasured:
        print(f"{unmeasured}/{len(rows)} images could not be measured (left out of the fit)")
    return rows


def main():
    args = parse_args()
    prompts = read_prompts(args.prompts)
    seeds = [int(s) for s in floats(args.seeds)]
    multipliers = sorted(set(floats(args.multipliers) + floats(args.diagnostic)))
    if 0.0 not in multipliers:
        multipliers = sorted(multipliers + [0.0])
    if not args.lora:
        multipliers = [0.0]

    rows = render(args, prompts, seeds, multipliers)
    rows = measure(args, rows)
    write_csv(rows, os.path.join(args.out, "sweep.csv"))
    for i in range(len(prompts)):
        contact_sheet([r for r in rows if r.prompt_idx == i], os.path.join(args.out, "sheets", f"prompt_{i:03d}.png"))

    summary = summarize(rows, margin=args.margin)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "positive": summary.positive, "diagnostic": summary.diagnostic,
                   "flags": summary.flags}, f, indent=2)
    text = "\n".join(summary.lines())
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
