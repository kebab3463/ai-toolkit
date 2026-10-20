"""Render synthetic before/after pairs with Krea 2 for slider plumbing checks.

Each pair uses one prompt template and one seed; only the attribute words differ
between the two images, so composition stays roughly similar. This is enough to
check that paired training runs end to end; it is NOT training data (identity and
framing are not held fixed the way real tuples must be).

Output layout matches extensions/slider_tuple/configs/dev/krea2_polarity_smoke.yaml:

    <out>/after/pair_NN.png + pair_NN.txt   (positive attribute, shared caption)
    <out>/before/pair_NN.png                (negative attribute)

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/dev/make_smoke_pairs.py
    .venv/bin/python extensions/slider_tuple/scripts/dev/make_smoke_pairs.py --model krea/Krea-2-Turbo --steps 9 --cfg 1
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO_ROOT, ".env"))

# (subject, scene) per pair; the attribute words go in front of the subject
SUBJECTS = [
    ("man in a grey t-shirt and jeans", "in a sunny park"),
    ("woman in a blue sundress", "on a city sidewalk"),
    ("man in a white button-up shirt and chinos", "in a bright office"),
    ("woman in a black tank top and leggings", "in a gym"),
    ("man in a green hoodie and sweatpants", "in a living room"),
    ("woman in a red sweater and jeans", "on a beach"),
]
TEMPLATE = "full-body photo of a {attr}{subject} standing {scene}, facing the camera, natural light"
POSITIVE = "heavyset "
NEGATIVE = "slim "


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "datasets", "slider_smoke"))
    p.add_argument("--num-pairs", type=int, default=5)
    p.add_argument("--model", default="krea/Krea-2-Raw")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1000, help="pair i uses seed + i")
    p.add_argument("--no-quantize", action="store_true", help="load the DiT and text encoder in bf16")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.num_pairs <= len(SUBJECTS):
        raise SystemExit(f"--num-pairs must be between 1 and {len(SUBJECTS)}")

    import torch

    from toolkit.config_modules import GenerateImageConfig, ModelConfig
    from toolkit.util.get_model import get_model_class

    after_dir = os.path.join(args.out, "after")
    before_dir = os.path.join(args.out, "before")
    os.makedirs(after_dir, exist_ok=True)
    os.makedirs(before_dir, exist_ok=True)

    quantize = not args.no_quantize
    model_config = ModelConfig(
        name_or_path=args.model,
        arch="krea2",
        # model dtype also sets the VAE dtype; leaving it at the float16 default
        # makes Krea 2's decode fail (bf16 latents into an fp16 VAE)
        dtype="bf16",
        quantize=quantize,
        qtype="qfloat8",
        quantize_te=quantize,
        qtype_te="qfloat8",
    )
    model_class = get_model_class(model_config)
    sd = model_class(
        device=args.device,
        model_config=model_config,
        dtype="bf16",
        noise_scheduler=model_class.get_train_scheduler(),
    )
    sd.load_model()

    image_configs = []
    for i, (subject, scene) in enumerate(SUBJECTS[: args.num_pairs]):
        stem = f"pair_{i + 1:02d}"
        caption = TEMPLATE.format(attr="", subject=subject, scene=scene)
        with open(os.path.join(after_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(caption + "\n")
        for attr, folder in ((POSITIVE, after_dir), (NEGATIVE, before_dir)):
            image_configs.append(
                GenerateImageConfig(
                    prompt=TEMPLATE.format(attr=attr, subject=subject, scene=scene),
                    width=args.size,
                    height=args.size,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    seed=args.seed + i,
                    output_path=os.path.join(folder, stem + ".png"),
                )
            )

    with torch.no_grad():
        sd.generate_images(image_configs, sampler="flowmatch")
    print(f"Wrote {args.num_pairs} pairs to {args.out}")


if __name__ == "__main__":
    main()
