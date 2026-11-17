import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="Base dataset directory")
    parser.add_argument(
        "--masks_subdir",
        type=str,
        default="masks",
        help="Subdirectory within the dataset containing the normal mask images",
        required=False,
    )
    parser.add_argument(
        "--exclude_masks_subdir",
        type=str,
        default="exclude_masks",
        help="Subdirectory within the dataset containing the exclude mask images",
        required=False,
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    masks_dir = dataset_dir / args.masks_subdir
    if args.masks_subdir == "masks":
        shutil.move(masks_dir, dataset_dir / "masks_pre")
        masks_dir = dataset_dir / "masks_pre"
    exclude_masks_dir = dataset_dir / args.exclude_masks_subdir
    output_dir = dataset_dir / "masks"
    output_dir.mkdir(exist_ok=True)

    for mask_path in masks_dir.glob("*.png"):
        exclude_mask_path = exclude_masks_dir / mask_path.name
        output_mask_path = output_dir / mask_path.name
        mask = np.array(Image.open(mask_path))
        exclude_mask = np.array(Image.open(exclude_mask_path).convert("L"))
        index_mask = (
            np.any(exclude_mask > 0, axis=-1)
            if exclude_mask.ndim == 3
            else exclude_mask > 0
        )
        mask[index_mask] = 0
        Image.fromarray(mask).save(output_mask_path)


if __name__ == "__main__":
    main()
