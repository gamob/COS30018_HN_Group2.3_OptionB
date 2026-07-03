#!/usr/bin/env python3
"""Interactive script to cherry-pick good images for part1.

Displays each image in data/part1/image and lets you mark it as good/bad.
Bad images are replaced with random good ones from data/sample/part1/image.

Usage:
  python scripts/cherry_pick_part1.py

Controls:
  - Press 'g' or 'y' to mark as GOOD
  - Press 'b' or 'n' to mark as BAD (will be replaced)
  - Press 's' to SKIP without marking
  - Press 'q' to QUIT
"""
from __future__ import annotations

import cv2
import os
import random
import shutil
from pathlib import Path


def get_image_files(folder: Path) -> list[Path]:
    """Get all image files from folder."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()])


def cherry_pick(part1_image_dir: Path, sample_image_dir: Path, seed: int | None = None) -> None:
    """Interactive cherry-picker."""
    if seed is not None:
        random.seed(seed)

    part1_files = get_image_files(part1_image_dir)
    if not part1_files:
        print(f"No images found in {part1_image_dir}")
        return

    sample_files = get_image_files(sample_image_dir)
    if not sample_files:
        print(f"No replacement images found in {sample_image_dir}")
        return

    print(f"Found {len(part1_files)} images in part1")
    print(f"Found {len(sample_files)} replacement candidates in sample folder\n")
    print("Controls:")
    print("  g / y = GOOD, keep it")
    print("  b / n = BAD, replace it")
    print("  s = SKIP (no mark)")
    print("  q = QUIT\n")

    good_count = 0
    bad_count = 0
    skipped_count = 0

    for idx, img_path in enumerate(part1_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{idx}/{len(part1_files)}] SKIP: could not read {img_path.name}")
            skipped_count += 1
            continue

        # Resize for display
        h, w = img.shape[:2]
        scale = min(800 / w, 600 / h)
        display_w, display_h = int(w * scale), int(h * scale)
        display_img = cv2.resize(img, (display_w, display_h))

        window_name = f"Cherry Pick [{idx}/{len(part1_files)}] {img_path.name}"
        cv2.imshow(window_name, display_img)
        print(f"[{idx}/{len(part1_files)}] {img_path.name}")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('g') or key == ord('y'):
                print("  -> GOOD (kept)")
                good_count += 1
                break
            elif key == ord('b') or key == ord('n'):
                replacement = random.choice(sample_files)
                shutil.copy2(replacement, img_path)
                print(f"  -> BAD (replaced with {replacement.name})")
                bad_count += 1
                break
            elif key == ord('s'):
                print("  -> SKIP")
                skipped_count += 1
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                print(f"\nQuit early.")
                print(f"Good: {good_count}, Bad: {bad_count}, Skipped: {skipped_count}")
                return
            else:
                print("  Invalid key. Try again (g/b/s/q)")

        cv2.destroyAllWindows()

    print(f"\n--- Summary ---")
    print(f"Good: {good_count}")
    print(f"Bad (replaced): {bad_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total: {len(part1_files)}")


def main() -> None:
    part1_dir = Path("data/part1/image")
    sample_dir = Path("data/sample/part1/image")

    if not part1_dir.exists():
        print(f"Error: {part1_dir} does not exist")
        return
    if not sample_dir.exists():
        print(f"Error: {sample_dir} does not exist")
        return

    cherry_pick(part1_dir, sample_dir)


if __name__ == "__main__":
    main()
