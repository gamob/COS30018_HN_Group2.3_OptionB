# Synthetic handwritten text dataset

This folder contains synthetic alphanumeric strings composed from individual
handwritten glyphs in `data/alphanumeric_emnist/test`.

Generated strings are divided into three categories:

- `letters`: uppercase `A-Z` only;
- `numbers`: digits `0-9` only;
- `mixed`: uppercase `A-Z` plus digits `0-9`, with at least one of each.

Each generated string:

- contains 3-8 characters by default;
- has an exact label in `labels.csv`.

## Appearance

All images use an intentionally easy-to-read configuration: every character
has the same height, all characters share a straight baseline, no rotation is
applied, and a fixed 10-pixel gap separates adjacent characters. There is no
separate difficulty level.

The images use white handwritten strokes on a black background, matching the
polarity of the source EMNIST files.

## Files

```text
synthetic-text/
|-- generate_synthetic_text.py
|-- generation_config.json
|-- labels.csv
|-- preview.png
`-- images/
    |-- letters/
    |-- numbers/
    `-- mixed/
```

`labels.csv` contains the relative image path, expected text, category,
and string length. `generation_config.json` records the seed and augmentation
settings used for the current generated dataset. `preview.png` is a small
contact sheet for visual inspection.

## Generate the dataset

Run from the project root:

```powershell
python data/synthetic-text/generate_synthetic_text.py
```

The default command uses seed `42` and creates 300 images for each category,
for 900 images in total.



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

## Copy command: generate a fresh 300-image batch per category

Run this command from the project root whenever a replacement batch is needed:

```powershell
python data/synthetic-text/generate_synthetic_text.py `
  --count-per-category 300 `
  --seed 45 `
  --overwrite
```

Increment or otherwise change `--seed` each time to produce a different batch
while retaining 300 images for each of `letters`, `numbers`, and `mixed`.

For a different dataset size:

```powershell
python data/synthetic-text/generate_synthetic_text.py `
  --count-per-category 2000 `
  --min-length 3 `
  --max-length 8 `
  --seed 42 `
  --overwrite
```

`--overwrite` replaces the generated PNG files under `images/` and updates the
manifest. Only the held-out EMNIST `test` split is used. The generator does not
modify the source dataset. Because these are composed synthetic strings,
evaluation results describe performance on this synthetic benchmark rather
than on natural handwritten phrases.

`scripts/test_ocr.py` discovers this `labels.csv` automatically when testing
the `images` folder or one of its subfolders. It can also be supplied explicitly
with `--labels-csv data/synthetic-text/labels.csv`.
