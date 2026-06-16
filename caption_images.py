"""
Caption images using an LMStudio vision model.
Saves each caption as a .txt file with the same name as the image.

Usage:
    # Caption only
    python caption_images.py /path/to/images --model "qwen3-vl-32b-instruct-nvfp4"

    # Caption + normalize in one pass
    python caption_images.py /path/to/images --canonical "woman with long auburn hair and green eyes, slender build"

    # Normalize previously captioned files only (no re-captioning)
    python caption_images.py /path/to/images --canonical "..." --postprocess-only
"""

import argparse
import base64
import io
import sys
from pathlib import Path

from openai import OpenAI

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

MIME_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".gif":  "image/gif",
    ".bmp":  "image/bmp",
}

SYSTEM_PROMPT = """\
You are captioning images to create training data for a character LoRA on Z-Image Base, \
a natural-language image generation model. Your captions will be used directly as training \
prompts. They must be objective, concrete, and match the vocabulary the model was trained on.

Write each caption as a single cohesive paragraph. No bullet points, tags, numbered lists, \
or preamble. Output the caption and nothing else.

Structure your paragraph in this order:

1. Shot type and framing — open with the composition: "A full-body shot of...", \
"A close-up portrait of...", "A medium shot from the waist up of...". Note the angle \
if not straight-on (e.g., "from a low angle", "slight three-quarter view", "profile").
2. Subject appearance — hair color, length, and style; facial features and expression; \
build and posture.
3. Clothing and accessories — specific colors, materials, layering, and coverage. \
Be concrete: "charcoal wool overcoat over a white collared shirt" not "nice outfit".
4. Pose and action — what the subject is doing and how they are positioned.
5. Background and setting — environment, location, and key scene elements with spatial \
depth (foreground, midground, background where relevant).
6. Lighting — be precise and use photographic language: "soft diffused overcast daylight \
from the left", "warm tungsten rim light from behind", "harsh direct midday sun overhead", \
"flat studio softbox fill", "cool blue ambient window light".
7. Visual medium and style — state clearly and explicitly: photorealistic photograph, \
digital illustration, oil painting, 3D render, cel-shaded anime artwork, ink illustration, \
etc. For photographs, describe lens character: "35mm perspective, shallow depth of field", \
"telephoto compression, blurred background", "wide-angle slight distortion", \
"analog film grain and muted color grade".
8. Mood — one brief concrete phrase only if it is unmistakably established in the image. \
Skip this entirely if ambiguous.

Rules:
- State facts only. Never use "appears to be", "possibly", "seems like", \
"it looks as though", "the image shows", or "I can see".
- Never use quality meta-tags like "8K", "high resolution", "masterpiece", \
"photorealistic render", or "detailed". Let the concrete description carry the quality signal.
- Use photographic and cinematic vocabulary throughout — name the lighting setup, \
shot type, and lens character.
- Describe only what is visible. If something is partially obscured or out of frame, \
describe only the visible portion.
- Target 90–150 words. Be thorough without repeating information or padding.\
"""


def encode_image(path: Path, max_size: int | None = None) -> tuple[str, str]:
    mime = MIME_TYPES.get(path.suffix.lower(), "image/jpeg")

    if max_size and HAS_PIL:
        img = Image.open(path)
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            buf = io.BytesIO()
            fmt = "JPEG" if mime == "image/jpeg" else "PNG"
            img.save(buf, format=fmt)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return b64, mime

    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return b64, mime


def caption_image(client: OpenAI, model: str, path: Path, max_size: int | None = None) -> str:
    b64, mime = encode_image(path, max_size)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                ],
            },
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


NORMALIZE_SYSTEM_PROMPT = """\
You are a caption editor. You will be given an image caption and a canonical character \
description. Rewrite the caption, replacing every reference to the subject's physical \
appearance — hair, face, eyes, skin tone, build, body type — with the provided canonical \
description verbatim. Do not paraphrase the canonical description.

Preserve everything else exactly: shot type, framing, angle, clothing, accessories, pose, \
action, background, setting, lighting, medium, style, lens character, and mood. Do not add, \
remove, or reorder any of these elements. Do not alter their wording.

Output only the rewritten caption. No preamble, no explanation.\
"""


def normalize_caption(client: OpenAI, model: str, caption: str, canonical: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": NORMALIZE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Canonical description: {canonical}\n\n"
                    f"Caption: {caption}"
                ),
            },
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Caption images with an LMStudio vision model."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing images to caption.",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:1234",
        help="LMStudio server base URL (default: http://localhost:1234).",
    )
    parser.add_argument(
        "--model",
        default="local-model",
        help='Model identifier shown in LMStudio (default: "local-model").',
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have a .txt caption file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds (default: 300). Increase if your GPU is slow.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        metavar="PIXELS",
        help="Resize images so the longest edge is at most PIXELS before sending. "
             "Requires Pillow. Speeds up inference significantly on large images "
             "(e.g. --max-size 1024).",
    )
    parser.add_argument(
        "--canonical",
        default=None,
        metavar="TEXT",
        help="Canonical character description to normalize all captions to after captioning. "
             "Replaces per-image appearance descriptions with this fixed string.",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Skip captioning entirely and only run normalization on existing .txt files. "
             "Requires --canonical.",
    )
    args = parser.parse_args()

    if args.max_size and not HAS_PIL:
        print("Warning: --max-size requires Pillow. Install with: pip install pillow")
        print("Continuing without resizing.\n")

    if args.postprocess_only and not args.canonical:
        print("Error: --postprocess-only requires --canonical.", file=sys.stderr)
        sys.exit(1)

    if not args.folder.is_dir():
        print(f"Error: '{args.folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    images = sorted(
        p for p in args.folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print("No images found.")
        sys.exit(0)

    client = OpenAI(base_url=f"{args.host}/v1", api_key="lm-studio", timeout=args.timeout)

    # ── Captioning pass ───────────────────────────────────────────────────────
    if not args.postprocess_only:
        total = len(images)
        resize_note = f", resizing to {args.max_size}px" if (args.max_size and HAS_PIL) else ""
        print(f"Found {total} image(s). Model: {args.model}, timeout: {args.timeout:.0f}s{resize_note}\n")

        ok = skipped = failed = 0

        for i, img_path in enumerate(images, 1):
            txt_path = img_path.with_suffix(".txt")
            prefix = f"[{i}/{total}] {img_path.name}"

            if args.skip_existing and txt_path.exists():
                print(f"{prefix} — skipped (already captioned)")
                skipped += 1
                continue

            print(f"{prefix} ...", end=" ", flush=True)
            try:
                caption = caption_image(client, args.model, img_path, args.max_size)
                txt_path.write_text(caption, encoding="utf-8")
                print("done")
                ok += 1
            except Exception as exc:
                print(f"FAILED: {exc}")
                failed += 1

        print(f"\nCaptioning finished. {ok} captioned, {skipped} skipped, {failed} failed.")

    # ── Normalization pass ────────────────────────────────────────────────────
    if args.canonical:
        txt_files = sorted(
            img.with_suffix(".txt") for img in images
            if img.with_suffix(".txt").exists()
        )

        if not txt_files:
            print("\nNo .txt files found to normalize.")
        else:
            print(f"\nNormalizing {len(txt_files)} caption(s) to canonical description...")
            print(f"  → \"{args.canonical}\"\n")

            ok = failed = 0
            for i, txt_path in enumerate(txt_files, 1):
                prefix = f"[{i}/{len(txt_files)}] {txt_path.name}"
                print(f"{prefix} ...", end=" ", flush=True)
                try:
                    original = txt_path.read_text(encoding="utf-8")
                    normalized = normalize_caption(client, args.model, original, args.canonical)
                    txt_path.write_text(normalized, encoding="utf-8")
                    print("done")
                    ok += 1
                except Exception as exc:
                    print(f"FAILED: {exc}")
                    failed += 1

            print(f"\nNormalization finished. {ok} updated, {failed} failed.")


if __name__ == "__main__":
    main()
