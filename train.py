"""Train a steering-angle CNN on the Sully Chen labelled driving dataset.

Example:
  pip install -r requirements-train.txt
  python train.py --data data/driving_dataset --epochs 12

The dataset is NOT distributed with this repository.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.data import OUTPUT_SCALE_DEGREES, image_to_tensor, read_index
from src.network import SteeringNet


class RoadDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.image_path) as img:
            tensor = image_to_tensor(img)[0]
        return torch.from_numpy(tensor), torch.tensor(
            [sample.angle_degrees / OUTPUT_SCALE_DEGREES], dtype=torch.float32
        )


def chronological_splits(count: int, gap: int):
    """Keep temporally adjacent frames out of neighbouring splits."""
    if count < 60:
        raise ValueError("At least 60 frames are required; a real study needs far more")
    end_train = int(count * 0.70)
    end_valid = int(count * 0.85)
    gap = min(gap, max(0, (end_valid - end_train) // 3))
    train = range(0, end_train)
    valid = range(end_train + gap, end_valid)
    test = range(end_valid + gap, count)
    if min(map(len, (train, valid, test))) == 0:
        raise ValueError("Not enough samples for the requested split and gap")
    return train, valid, test


def score(model, loader, device):
    model.eval()
    abs_total = sq_total = count = 0.0
    with torch.inference_mode():
        for inputs, expected in loader:
            inputs, expected = inputs.to(device), expected.to(device)
            diff_degrees = (model(inputs) - expected) * OUTPUT_SCALE_DEGREES
            abs_total += diff_degrees.abs().sum().item()
            sq_total += diff_degrees.square().sum().item()
            count += expected.numel()
    return {"mae_degrees": round(abs_total / count, 3),
            "rmse_degrees": round((sq_total / count) ** 0.5, 3)}


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    samples = read_index(args.data)
    train_idx, valid_idx, test_idx = chronological_splits(len(samples), args.gap)
    dataset = RoadDataset(samples)
    def loader(idx, shuffle=False):
        return DataLoader(Subset(dataset, list(idx)), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=0)
    train_loader = loader(train_idx, shuffle=True)
    valid_loader = loader(valid_idx)
    test_loader = loader(test_idx)

    model = SteeringNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inputs)
            total += len(inputs)
        validation = score(model, valid_loader, device)
        print(f"Epoch {epoch:02d}/{args.epochs} | train MSE: {train_loss / total:.5f} "
              f"| validation MAE: {validation['mae_degrees']:.2f} deg")
        if validation["mae_degrees"] < best_loss:
            best_loss = validation["mae_degrees"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No successful training epochs")
    model.load_state_dict(best_state)
    model.eval()
    results = {
        "train_samples": len(train_idx), "validation_samples": len(valid_idx),
        "test_samples": len(test_idx), "gap_frames": min(args.gap, max(0, (int(len(samples) * .85) - int(len(samples) * .7)) // 3)),
        "validation": score(model, valid_loader, device),
        "held_out_test": score(model, test_loader, device),
        "model": "SteeringNet", "labels_unit": "degrees",
        "limitations": "Performance on one chronological split is not real-world driving validation.",
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "steering.pt")
    (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    if not args.skip_export:
        try:
            model_cpu = model.to("cpu")
            torch.onnx.export(model_cpu, torch.zeros(1, 3, 66, 200),
                              str(out / "steering.onnx"),
                              export_params=True, opset_version=17,
                              input_names=["road_image"], output_names=["normalized_angle"],
                              dynamo=False)
        except Exception as exc:
            raise RuntimeError("ONNX export failed. Ensure 'onnx' is installed from requirements-train.txt") from exc
        print(f"Exported deployable model to {out / 'steering.onnx'}")
    print("CAUTION: This model is for educational use only; never use it to steer a vehicle.")


def main():
    parser = argparse.ArgumentParser(description="Train original steering prediction CNN")
    parser.add_argument("--data", default="data/driving_dataset", help="Folder with data.txt and JPGs")
    parser.add_argument("--output", default="models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gap", type=int, default=30)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-export", action="store_true", help="Train without exporting ONNX (debug only)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
