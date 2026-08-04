from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from src.models.model import load_letter_cnn_model
from src.models.training.train_cnn import _load_image_array


def load_validation_data(
    validation_dir: Path,
    max_per_class: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_names = sorted(path.name.upper() for path in validation_dir.iterdir() if path.is_dir())
    images: list[np.ndarray] = []
    labels: list[int] = []

    for class_index, class_name in enumerate(class_names):
        paths = sorted((validation_dir / class_name).glob("*.png"))
        if max_per_class > 0:
            paths = paths[:max_per_class]
        for path in paths:
            images.append(_load_image_array(path))
            labels.append(class_index)

    if not images:
        raise ValueError(f"No PNG validation images found in {validation_dir}")
    return np.stack(images), np.asarray(labels, dtype=np.int32), class_names


def evaluate_letter_model(
    model_path: Path,
    validation_dir: Path,
    output_dir: Path,
    max_per_class: int = 0,
    batch_size: int = 128,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_valid, y_true, class_names = load_validation_data(validation_dir, max_per_class)
    model = load_letter_cnn_model(model_path)

    probabilities = model.predict(x_valid, batch_size=batch_size, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    per_class = {}
    for index, name in enumerate(class_names):
        total = int(matrix[index].sum())
        correct = int(matrix[index, index])
        per_class[name] = {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
        }

    confusions = []
    for actual in range(len(class_names)):
        for predicted in range(len(class_names)):
            if actual != predicted and matrix[actual, predicted] > 0:
                confusions.append({
                    "actual": class_names[actual],
                    "predicted": class_names[predicted],
                    "count": int(matrix[actual, predicted]),
                })
    confusions.sort(key=lambda item: item["count"], reverse=True)

    metrics = {
        "model": str(model_path),
        "validation_directory": str(validation_dir),
        "sample_count": int(len(y_true)),
        "max_per_class": max_per_class,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mean_confidence": float(np.max(probabilities, axis=1).mean()),
        "per_class": per_class,
        "top_confusions": confusions[:20],
        "confusion_matrix": matrix.tolist(),
    }

    metrics_path = output_dir / "letter_cnn_evaluation.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title=f"Letter CNN confusion matrix (accuracy: {metrics['accuracy']:.1%})",
        xlabel="Predicted letter",
        ylabel="Actual letter",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "letter_cnn_confusion_matrix.png", dpi=160)
    plt.close(figure)
    return metrics


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Evaluate the saved letter CNN on EMNIST validation data.")
    parser.add_argument("--model-path", type=Path, default=project_root / "src/models/letter_cnn_model.h5")
    parser.add_argument("--validation-dir", type=Path, default=project_root / "data/letters/valid")
    parser.add_argument("--output-dir", type=Path, default=project_root / "documents/model-evaluation")
    parser.add_argument("--max-per-class", type=int, default=0, help="0 evaluates every validation image")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    metrics = evaluate_letter_model(
        args.model_path,
        args.validation_dir,
        args.output_dir,
        max_per_class=args.max_per_class,
        batch_size=args.batch_size,
    )
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean confidence: {metrics['mean_confidence']:.4f}")
    print("Top confusions:")
    for item in metrics["top_confusions"][:10]:
        print(f"  {item['actual']} -> {item['predicted']}: {item['count']}")


if __name__ == "__main__":
    main()
