import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

from classifier import CLASS_LABELS, IMG_SIZE, NUM_CHANNELS, ConvNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_transforms(train: bool) -> transforms.Compose:
    aug = []
    if train:
        aug.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    aug.extend(
        [
            transforms.Grayscale(num_output_channels=NUM_CHANNELS),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transforms.Compose(aug)

def compute_class_weights(labels: Iterable[int], num_classes: int) -> List[float]:
    counts = [0] * num_classes
    for lbl in labels:
        counts[int(lbl)] += 1
    total = sum(counts)
    weights = []
    for idx, cnt in enumerate(counts):
        weights.append((total / (num_classes * cnt)) if cnt > 0 else 0.0)
    return weights


class TransformSubset(Dataset):
    """Wrap a Subset to apply a per-split transform without duplicating indices."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset.dataset[self.subset.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def make_loaders(
    data_dir: Path,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, Sequence[str]]:
    train_transform = build_transforms(train=True)
    val_transform = build_transforms(train=False)
    full_ds = datasets.ImageFolder(root=str(data_dir))
    class_names = full_ds.classes
    num_classes = len(class_names)

    val_len = int(len(full_ds) * val_split)
    train_len = len(full_ds) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=generator)

    train_labels = [full_ds.targets[idx] for idx in train_ds.indices]
    class_weights = compute_class_weights(train_labels, num_classes)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds=TransformSubset(train_ds, transform=train_transform)
    val_ds =TransformSubset(val_ds, transform=val_transform)

    train_loader= DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, class_names


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion : nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def save_artifacts(
    model: nn.Module,
    class_names: Sequence[str],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "img_size": IMG_SIZE,
            "num_channels": NUM_CHANNELS,
            "arch": "ConvNet",
        },
        save_path,
    )
    labels_path = save_path.parent / "labels.json"
    with open(labels_path, "w") as fh:
        json.dump(list(class_names), fh)


def main():
    parser = argparse.ArgumentParser(description="Train ConvNet emotion classifier (PyTorch)")
    parser.add_argument("--data", type=Path, default=Path("kaggle/train"), help="Training data directory (ImageFolder layout)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--max-lr", type=float, default=3e-3, help="Peak learning rate for OneCycle")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for CrossEntropyLoss")
    parser.add_argument("--save", type=Path, default=Path("models/emotion_cnn.pt"), help="Where to save trained model")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = make_loaders(
        data_dir=args.data,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Classes: {class_names}")

    model = ConvNet(num_classes=len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_artifacts(model, class_names, args.save)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc:.4f}"
        )

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}. Model saved to {args.save}")


if __name__ == "__main__":
    main()
