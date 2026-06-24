#!/usr/bin/env python3
"""Small pipeline: sample 30 images into data/part1, preprocess to part2, segment to part3.

Usage:
  python scripts/run_pipeline.py [--source SRC] [--part1 PART1] [--part2 PART2] [--part3 PART3] [--n N]

This script intentionally keeps changes local (does not delete other sample data).
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
import sys

# Ensure repo root is on sys.path so we can import `src` package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import preprocessing as preproc
from src.segmentation import segmentation as seg


def select_images(source_dir: Path, dst_dir: Path, n: int = 30, seed: int | None = None) -> int:
    source_dir = Path(source_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [p for p in sorted(source_dir.iterdir()) if p.suffix.lower() in exts and p.is_file()]
    if not images:
        raise RuntimeError(f"no images found in {source_dir}")

    if seed is not None:
        random.seed(seed)

    chosen = images if len(images) <= n else random.sample(images, n)

    for p in chosen:
        shutil.copy2(p, dst_dir / p.name)

    print(f"Copied {len(chosen)} images from {source_dir} -> {dst_dir}")
    return len(chosen)


def run_preprocess(input_dir: Path, output_dir: Path, **kwargs) -> None:
    preproc.process_path(str(input_dir), str(output_dir), **kwargs)


def run_segmentation(input_dir: Path, output_dir: Path, method: str = "connected_components") -> None:
    import cv2

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts and p.is_file()])
    if not files:
        print(f"No images found in {input_dir} for segmentation")
        return

    for img_path in files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"failed to read {img_path}; skipping")
            continue

        digits = seg.segment_and_preprocess(img, method=method)
        base = img_path.stem
        for idx, d in enumerate(digits, 1):
            out_path = output_dir / f"{base}_digit{idx}.png"
            cv2.imwrite(str(out_path), (d * 255).astype("uint8"))

    print(f"Segmentation finished. outputs written to {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build small 3-stage pipeline: part1->part2->part3")
    p.add_argument("--source", default=os.path.join("data", "sample", "part1", "image"), help="source images to sample from")
    p.add_argument("--part1", default=os.path.join("data", "part1", "image"), help="destination for part1 images")
    p.add_argument("--part2", default=os.path.join("data", "part2"), help="preprocess output directory")
    p.add_argument("--part3", default=os.path.join("data", "part3"), help="segmentation output directory")
    p.add_argument("--n", type=int, default=30, help="number of images to sample for part1")
    p.add_argument("--seed", type=int, default=None, help="random seed for sampling")
    p.add_argument("--seg-method", choices=("connected_components", "projection"), default="connected_components")
    p.add_argument("--no-normalize", action="store_true", help="disable normalization in preprocessing (save uint8 images)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    src_dir = Path(args.source)
    part1_dir = Path(args.part1)
    part2_dir = Path(args.part2)
    part3_dir = Path(args.part3)

    # Step 1: prepare part1 (sample N images)
    select_images(src_dir, part1_dir, n=args.n, seed=args.seed)

    # Step 2: preprocess part1 -> part2
    preproc_opts = dict(
        size=(28, 28),
        method="otsu",
        blur_ksize=5,
        adaptive_params=(15, 7),
        thresh=128,
        invert=False,
        normalize=not args.no_normalize,
        margin=4,
        save_steps=None,
    )
    run_preprocess(part1_dir, part2_dir, **preproc_opts)

    # Step 3: segmentation part2 -> part3
    run_segmentation(part2_dir, part3_dir, method=args.seg_method)


if __name__ == "__main__":
    main()
