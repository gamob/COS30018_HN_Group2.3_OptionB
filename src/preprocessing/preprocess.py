"""Compatibility wrapper: run the preprocessing CLI using the shorter filename.

This allows running:
    python src/preprocessing/preprocess.py <input> <output> [options]

It simply executes the existing `preprocessing.py` as a script while preserving
`sys.argv` so the original CLI code is invoked.
"""
from __future__ import annotations

import runpy
import sys
import os


def main() -> None:
    script_path = os.path.join(os.path.dirname(__file__), "preprocessing.py")
    sys.argv = [script_path] + sys.argv[1:]
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
