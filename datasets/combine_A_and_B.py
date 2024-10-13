import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def image_write(paths: Tuple[Path, Path, Path]) -> None:
    path_A, path_B, path_AB = paths
    try:
        im_A = cv2.imread(str(path_A), cv2.IMREAD_COLOR)
        im_B = cv2.imread(str(path_B), cv2.IMREAD_COLOR)

        if im_A is None or im_B is None:
            print(f"Error reading images: {path_A} or {path_B}")
            return

        min_height = min(im_A.shape[0], im_B.shape[0])
        im_A = im_A[:min_height, :]
        im_B = im_B[:min_height, :]

        im_AB = np.concatenate([im_A, im_B], axis=1)
        cv2.imwrite(str(path_AB), im_AB)
        print(f"Successfully wrote: {path_AB}")
    except Exception as e:
        print(f"Error processing images {path_A} and {path_B}: {str(e)}")


def process_split(
    fold_A: Path, fold_B: Path, fold_AB: Path, num_imgs: int, use_AB: bool
) -> None:
    print(f"Processing: fold_A={fold_A}, fold_B={fold_B}, fold_AB={fold_AB}")

    fold_AB.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {fold_AB}")

    img_list = list(fold_A.glob("*_A.*" if use_AB else "*"))
    num_imgs = min(num_imgs, len(img_list))
    print(f"Found {len(img_list)} images, processing {num_imgs}")

    tasks = []
    for img_path in img_list[:num_imgs]:
        name_A = img_path.name
        path_A = img_path
        name_B = name_A.replace("_A.", "_B.") if use_AB else name_A
        path_B = fold_B / name_B

        if path_A.is_file() and path_B.is_file():
            name_AB = name_A.replace("_A.", ".") if use_AB else name_A
            path_AB = fold_AB / name_AB
            tasks.append((path_A, path_B, path_AB))
        else:
            print(f"Skipping {name_A}: A={path_A.is_file()}, B={path_B.is_file()}")

    print(f"Created {len(tasks)} tasks")

    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(image_write, tasks),
                total=len(tasks),
                desc="Processing images",
            )
        )


def main(args: argparse.Namespace) -> None:
    fold_A = Path(args.fold_A)
    fold_B = Path(args.fold_B)
    fold_AB = Path(args.fold_AB)

    print(f"Input A: {fold_A}")
    print(f"Input B: {fold_B}")
    print(f"Output AB: {fold_AB}")

    if not fold_A.is_dir() or not fold_B.is_dir():
        print(
            f"Error: Input directories do not exist. A: {fold_A.is_dir()}, B: {fold_B.is_dir()}"
        )
        return

    process_split(fold_A, fold_B, fold_AB, args.num_imgs, args.use_AB)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create image pairs")
    parser.add_argument(
        "--fold_A", help="Input directory for image A", type=str, required=True
    )
    parser.add_argument(
        "--fold_B", help="Input directory for image B", type=str, required=True
    )
    parser.add_argument("--fold_AB", help="Output directory", type=str, required=True)
    parser.add_argument(
        "--num_imgs", help="Number of images to process", type=int, default=1000000
    )
    parser.add_argument(
        "--use_AB", help="If true: (0001_A, 0001_B) to (0001_AB)", action="store_true"
    )

    args = parser.parse_args()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    main(args)
