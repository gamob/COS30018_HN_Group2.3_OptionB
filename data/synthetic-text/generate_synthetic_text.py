"""Generate handwritten alphanumeric strings by composing EMNIST glyphs.

The generator reads held-out glyphs from data/alphanumeric_emnist/test and
creates letter-only, number-only, and mixed strings with moderate handwritten
variation. Every mixed string contains at least one uppercase letter and one
digit.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import string
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CHARACTERS = string.digits + string.ascii_uppercase
DIGITS = string.digits
LETTERS = string.ascii_uppercase
CATEGORIES = ("letters", "numbers", "mixed")


@dataclass(frozen=True)
class ImageConfig:
    target_height_min: int
    target_height_max: int
    rotation_degrees: float
    baseline_jitter: int
    spacing_min: int
    spacing_max: int
    canvas_padding: int


CONFIG = ImageConfig(
    target_height_min=36,
    target_height_max=36,
    rotation_degrees=0.0,
    baseline_jitter=0,
    spacing_min=10,
    spacing_max=10,
    canvas_padding=10,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description="Compose EMNIST test glyphs into synthetic handwritten strings."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_root / "data" / "alphanumeric_emnist" / "test",
        help="Folder containing the 0-9 and A-Z EMNIST test class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Destination containing images/, labels.csv, and generation_config.json.",
    )
    parser.add_argument(
        "--count-per-category",
        "--count-per-group",
        "--count-per-difficulty",
        dest="count_per_category",
        type=int,
        default=300,
        help=(
            "Images to create for each of letters, numbers, and mixed. "
            "The old count option names remain as aliases."
        ),
    )
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing generated labels.csv and image names.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.count_per_category <= 0:
        raise ValueError("--count-per-category must be greater than zero")
    if args.min_length < 2:
        raise ValueError("--min-length must be at least 2 to include a letter and a digit")
    if args.max_length < args.min_length:
        raise ValueError("--max-length must be greater than or equal to --min-length")


def index_glyphs(source_dir: Path) -> dict[str, list[Path]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"EMNIST source directory not found: {source_dir}")

    glyphs: dict[str, list[Path]] = {}
    for character in CHARACTERS:
        class_dir = source_dir / character
        paths = sorted(class_dir.glob("*.png")) if class_dir.is_dir() else []
        if not paths:
            raise FileNotFoundError(f"No PNG glyphs found for class {character!r} in {class_dir}")
        glyphs[character] = paths
    return glyphs


def make_text(
    rng: random.Random,
    min_length: int,
    max_length: int,
    category: str,
) -> str:
    """Create a letter-only, number-only, or mixed alphanumeric string."""
    length = rng.randint(min_length, max_length)
    if category == "letters":
        return "".join(rng.choice(LETTERS) for _ in range(length))
    if category == "numbers":
        return "".join(rng.choice(DIGITS) for _ in range(length))
    if category == "mixed":
        characters = [rng.choice(LETTERS), rng.choice(DIGITS)]
        characters.extend(rng.choice(CHARACTERS) for _ in range(length - 2))
        rng.shuffle(characters)
        return "".join(characters)
    raise ValueError(f"Unknown text category: {category}")


def crop_foreground(image: Image.Image) -> Image.Image:
    """Crop black padding while retaining faint anti-aliased foreground pixels."""
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    foreground = grayscale > 8
    if not foreground.any():
        raise ValueError("Encountered a blank EMNIST glyph")
    rows, columns = np.where(foreground)
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    return image.convert("L").crop((left, top, right, bottom))


def transform_glyph(
    glyph_path: Path,
    config: ImageConfig,
    rng: random.Random,
) -> Image.Image:
    with Image.open(glyph_path) as source:
        glyph = crop_foreground(source)

    target_height = rng.randint(config.target_height_min, config.target_height_max)
    scale = target_height / glyph.height
    target_width = max(1, round(glyph.width * scale))
    glyph = glyph.resize((target_width, target_height), Image.Resampling.LANCZOS)

    if config.rotation_degrees == 0:
        return glyph

    angle = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
    return glyph.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=0,
    )


def compose_text(
    text: str,
    glyph_index: dict[str, list[Path]],
    rng: random.Random,
) -> Image.Image:
    config = CONFIG
    glyphs = [
        transform_glyph(rng.choice(glyph_index[character]), config, rng)
        for character in text
    ]
    spacings = [
        rng.randint(config.spacing_min, config.spacing_max)
        for _ in range(max(0, len(glyphs) - 1))
    ]
    baseline_offsets = [
        rng.randint(-config.baseline_jitter, config.baseline_jitter)
        for _ in glyphs
    ]

    content_width = sum(glyph.width for glyph in glyphs) + sum(spacings)
    content_width = max(content_width, max(glyph.width for glyph in glyphs))
    tallest = max(glyph.height for glyph in glyphs)
    canvas_width = content_width + 2 * config.canvas_padding
    canvas_height = tallest + 2 * (config.canvas_padding + config.baseline_jitter)
    canvas = Image.new("L", (canvas_width, canvas_height), color=0)

    x = config.canvas_padding
    baseline_y = config.canvas_padding + config.baseline_jitter + tallest
    for index, (glyph, offset) in enumerate(zip(glyphs, baseline_offsets)):
        y = baseline_y - glyph.height + offset
        # Spacing is always positive, so direct pasting preserves the original
        # stroke intensity without requiring an alpha-like grayscale mask.
        canvas.paste(glyph, (x, y))
        if index < len(spacings):
            x += glyph.width + spacings[index]

    return canvas


def build_preview(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Create a compact contact sheet for quick visual inspection."""
    selected: list[dict[str, str]] = []
    for category in CATEGORIES:
        matches = [row for row in rows if row["category"] == category]
        selected.extend(matches[:4])

    cell_width, cell_height = 300, 92
    sheet_rows = max(1, (len(selected) + 1) // 2)
    sheet = Image.new("L", (cell_width * 2, cell_height * sheet_rows), color=24)
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        column, grid_row = index % 2, index // 2
        x0, y0 = column * cell_width, grid_row * cell_height
        image_path = output_dir / row["filename"]
        with Image.open(image_path) as sample:
            sample = sample.convert("L")
            max_width, max_height = cell_width - 16, cell_height - 28
            scale = min(max_width / sample.width, max_height / sample.height, 1.0)
            if scale < 1.0:
                sample = sample.resize(
                    (max(1, round(sample.width * scale)), max(1, round(sample.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            sheet.paste(sample, (x0 + 8, y0 + 22))
        draw.text(
            (x0 + 8, y0 + 5),
            f'{row["category"]}: {row["text"]}',
            fill=230,
        )
    sheet.save(output_dir / "preview.png", optimize=True)


def clear_generated_images(image_root: Path) -> None:
    """Remove prior generated PNGs and their now-empty folders."""
    if not image_root.is_dir():
        return
    for image_path in image_root.rglob("*.png"):
        image_path.unlink()
    directories = sorted(
        (path for path in image_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass


def generate(args: argparse.Namespace) -> None:
    validate_args(args)
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "labels.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Generated dataset already exists at {manifest_path}. "
            "Pass --overwrite to regenerate the named output files."
        )

    glyph_index = index_glyphs(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = output_dir / "images"
    if args.overwrite:
        clear_generated_images(image_root)
    for category in CATEGORIES:
        (image_root / category).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for category in CATEGORIES:
        used_texts: set[str] = set()
        # Separate deterministic streams keep one category stable if the size
        # of a different category changes in a later generation run.
        rng = random.Random(f"{args.seed}:{category}")
        for index in range(1, args.count_per_category + 1):
            text = make_text(rng, args.min_length, args.max_length, category)
            while text in used_texts:
                text = make_text(rng, args.min_length, args.max_length, category)
            used_texts.add(text)

            filename = f"{category}_{index:06d}.png"
            relative_path = Path("images") / category / filename
            image = compose_text(text, glyph_index, rng)
            image.save(output_dir / relative_path, optimize=True)
            rows.append(
                {
                    "filename": relative_path.as_posix(),
                    "text": text,
                    "category": category,
                    "length": str(len(text)),
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("filename", "text", "category", "length"),
        )
        writer.writeheader()
        writer.writerows(rows)

    generation_config = {
        "source_dir": str(source_dir),
        "source_split": "test",
        "seed": args.seed,
        "count_per_category": args.count_per_category,
        "total_images": len(rows),
        "min_length": args.min_length,
        "max_length": args.max_length,
        "characters": CHARACTERS,
        "categories": list(CATEGORIES),
        "transform": asdict(CONFIG),
    }
    with (output_dir / "generation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(generation_config, handle, indent=2)
        handle.write("\n")

    build_preview(rows, output_dir)
    print(f"Generated {len(rows)} images in {output_dir / 'images'}")
    print(f"Labels: {manifest_path}")
    print(f"Preview: {output_dir / 'preview.png'}")


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
