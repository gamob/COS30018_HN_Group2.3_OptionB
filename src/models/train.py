"""Train a digit classifier on MNIST using PyTorch.

Usage example:
    python src/models/train.py --epochs 10 --batch-size 64 --output-dir models/

The script downloads MNIST to `~/.torch` by default and saves checkpoints
to `--output-dir`.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
except Exception:
    raise RuntimeError("PyTorch and torchvision are required. Install via pip or conda.")

from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

try:
    from .model import SimpleCNN
except Exception:  # when running as a script (no package context)
    from model import SimpleCNN


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += data.size(0)
        pbar.set_postfix(loss=running_loss / total, acc=100.0 * correct / total)

    return running_loss / total, 100.0 * correct / total


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device, collect_preds: bool = False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item() * data.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)
            if collect_preds:
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())

    avg_loss = running_loss / total if total else 0.0
    acc = 100.0 * correct / total if total else 0.0
    if collect_preds:
        return avg_loss, acc, np.concatenate(all_preds), np.concatenate(all_targets)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train a digit classifier on MNIST")
    parser.add_argument("--data-dir", default="data/mnist", help="Path to download/store MNIST")
    parser.add_argument("--output-dir", default="models", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detect if not set)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", optimizer.state_dict()))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} (start_epoch={start_epoch})")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        t1 = time.time()

        print(f"Epoch {epoch}/{args.epochs}  time={t1-t0:.1f}s  train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")

        # save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "args": vars(args)
        }, ckpt_path)

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.output_dir, "best.pth")
            torch.save(model.state_dict(), best_path)

    # final evaluation with confusion matrix
    _, _, preds, targets = evaluate(model, test_loader, criterion, device, collect_preds=True)
    cm = confusion_matrix(targets, preds)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(targets, preds, digits=4))


if __name__ == "__main__":
    main()
