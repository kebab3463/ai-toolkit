from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "dataset_dir", type=Path, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--pos",
        type=str,
        default="body_masks/heavy",
        help="Subdirectory of the dataset directory for the positive images",
    )
    parser.add_argument(
        "--neg",
        type=str,
        default="body_masks/thin",
        help="Subdirectory of the dataset directory for the negative images",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="body_masks/polygon",
        help="Subdirectory of the dataset directory for the measurement region masks",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    pos_dir = dataset_dir / args.pos
    neg_dir = dataset_dir / args.neg
    regions_dir = dataset_dir / args.regions
    stems = [file.stem for file in pos_dir.glob("*")]
    results = {}
    for stem in stems:
        pos_mask_path = pos_dir / f"{stem}.png"
        neg_mask_path = neg_dir / f"{stem}.png"
        region_mask_path = regions_dir / f"{stem}.png"
        pos_mask = np.array(Image.open(pos_mask_path).convert("L"))
        neg_mask = np.array(Image.open(neg_mask_path).convert("L"))
        region_mask = np.array(Image.open(region_mask_path).convert("L"))
        results[stem] = [
            np.count_nonzero(np.logical_and(neg_mask, region_mask)),
            np.count_nonzero(np.logical_and(pos_mask, region_mask)),
        ]
    results_csv = "stem,neg,pos\n"
    for stem, counts in results.items():
        results_csv += f"{stem},{counts[0]},{counts[1]}\n"
    with open(dataset_dir / "measurements.csv", "w") as f:
        f.write(results_csv)


if __name__ == "__main__":
    main()
