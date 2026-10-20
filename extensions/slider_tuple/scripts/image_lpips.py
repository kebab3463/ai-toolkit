"""LPIPS (AlexNet) between a reference image and one or more others.

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/image_lpips.py ref.png a.png b.png ...
"""

import sys

import numpy as np
import torch
from PIL import Image


def to_tensor(path, size):
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        raise SystemExit(__doc__)
    import lpips

    net = lpips.LPIPS(net="alex", verbose=False).eval()
    size = Image.open(argv[0]).size
    ref = to_tensor(argv[0], size)
    with torch.no_grad():
        for path in argv[1:]:
            other = to_tensor(path, size)
            print(f"{float(net(ref, other)):.5f}  max|diff| {float((ref - other).abs().max()) * 127.5:5.1f}  {path}")


if __name__ == "__main__":
    main()
