from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from preprocessing.preprocessing import process_path
from preprocessing.preprocessing import preprocess_image_steps
from models.model import load_digit_cnn_model, predict_digit
from models.training.train_cnn import train_and_save_model


def run_preprocess(args: argparse.Namespace) -> None:
    process_path(
        args.input,
        args.output,
        size=tuple(args.size),
        method=args.method,
        blur_ksize=args.blur_ksize,
        adaptive_params=(args.adaptive_block_size, args.adaptive_C),
        thresh=args.thresh,
        invert=args.invert,
        normalize=args.normalize,
        margin=args.margin,
        save_steps=args.save_steps,
    )


def run_train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    train_and_save_model(output_dir, epochs=args.epochs)


def run_predict(args: argparse.Namespace) -> None:
    model = load_digit_cnn_model(args.model_path)
    steps = preprocess_image_steps(
        args.image_path,
        size=(28, 28),
        method=args.method,
        blur_ksize=args.blur_ksize,
        adaptive_params=(args.adaptive_block_size, args.adaptive_C),
        thresh=args.thresh,
        invert=args.invert,
        normalize=True,
        margin=args.margin,
    )
    prediction = predict_digit(model, steps["final"])
    print(f"Predicted digit: {prediction}")


def run_gui(_: argparse.Namespace) -> None:
    print("Launch the GUI with: streamlit run src/gui/app.py")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="HNRS project workflow runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess one or more images")
    preprocess_parser.add_argument("input", help="Input image file or folder")
    preprocess_parser.add_argument("output", help="Output directory")
    preprocess_parser.add_argument("--method", choices=("otsu", "simple", "adaptive"), default="otsu")
    preprocess_parser.add_argument("--size", type=int, nargs=2, default=(28, 28))
    preprocess_parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    preprocess_parser.add_argument("--invert", action="store_true")
    preprocess_parser.add_argument("--blur-ksize", type=int, default=5)
    preprocess_parser.add_argument("--adaptive-block-size", type=int, default=15)
    preprocess_parser.add_argument("--adaptive-C", type=int, default=7)
    preprocess_parser.add_argument("--thresh", type=int, default=128)
    preprocess_parser.add_argument("--margin", type=int, default=4)
    preprocess_parser.add_argument("--save-steps", default=None)
    preprocess_parser.set_defaults(func=run_preprocess)

    train_parser = subparsers.add_parser("train", help="Train the CNN model")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "models"))
    train_parser.set_defaults(func=run_train)

    predict_parser = subparsers.add_parser("predict", help="Predict a digit image using the saved model")
    predict_parser.add_argument("image_path", help="Image file to predict")
    predict_parser.add_argument("--model-path", default=None)
    predict_parser.add_argument("--method", choices=("otsu", "simple", "adaptive"), default="otsu")
    predict_parser.add_argument("--blur-ksize", type=int, default=5)
    predict_parser.add_argument("--adaptive-block-size", type=int, default=15)
    predict_parser.add_argument("--adaptive-C", type=int, default=7)
    predict_parser.add_argument("--thresh", type=int, default=128)
    predict_parser.add_argument("--invert", action="store_true")
    predict_parser.add_argument("--margin", type=int, default=4)
    predict_parser.set_defaults(func=run_predict)

    gui_parser = subparsers.add_parser("gui", help="Show GUI launch instructions")
    gui_parser.set_defaults(func=run_gui)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
