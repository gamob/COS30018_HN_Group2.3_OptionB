Sample datasets for quick testing

Structure produced by `scripts/create_sample_datasets.py --all`:

- part1/
  - image/                # word images (flat)
  - train.txt, valid.txt, test.txt   # manifests: "image/<filename> <label>"
  - prep_out/             # optional preprocessing output (created by runner)

- part2/
  - raw/                  # images for segmentation
  - processed/            # segmentation outputs (created by runner)

- part3/
  - train/<label>/*       # per-class training images
  - valid/<label>/*
  - test/<label>/*
  - train.txt, valid.txt, test.txt   # simple manifests

How to use:

1. Create the sample data (already ran in this workspace):
   python scripts/create_sample_datasets.py --all --force

2. Preprocess the word-level images and run segmentation on the sample:
   python scripts/run_sample_pipeline.py --part1 --part2 --force

3. Point your training/preprocessing scripts to the sample folders when testing.
