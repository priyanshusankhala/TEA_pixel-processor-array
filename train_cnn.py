"""
Simple 3-layer CNN for touch/no-touch classification.
Architecture matches what SCAMP-5 can run (small kernels, few channels).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
import time


class TactileDataset(Dataset):
    """Load tactile marker images."""

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        for label, cls_name in enumerate(["no_contact", "contact"]):
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.exists(cls_dir):
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.endswith(".png"):
                        self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # Grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


class TactileCNN(nn.Module):
    """
    3-layer CNN designed to be SCAMP-5 compatible:
    - Small kernels (3x3)
    - Few channels (8, 16, 32)
    - Simple pooling
    - Single FC output
    """

    def __init__(self):
        super().__init__()
        # Layer 1: Conv 3x3, 1->8 channels, stride 2
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        # Layer 2: Conv 3x3, 8->16 channels, stride 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Layer 3: Conv 3x3, 16->32 channels, stride 2
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Global average pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 256->128, 8ch
        x = self.relu(self.bn2(self.conv2(x)))  # 128->64, 16ch
        x = self.relu(self.bn3(self.conv3(x)))  # 64->32, 32ch
        x = self.pool(x)  # 32->1x1, 32ch
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train():
    print("=" * 50)
    print("Training Baseline CNN (Float32)")
    print("=" * 50)
    print()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load data
    train_set = TactileDataset("dataset/train", transform=transform)
    test_set = TactileDataset("dataset/test", transform=transform)

    print(f"  Train samples: {len(train_set)}")
    print(f"  Test samples: {len(test_set)}")
    print()

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = TactileCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Architecture: Conv3x3(1→8) → Conv3x3(8→16) → Conv3x3(16→32) → GAP → FC(2)")
    print()

    # Training loop
    epochs = 25
    best_acc = 0
    history = {"train_loss": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        history["train_loss"].append(avg_loss)
        history["test_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "model_float32.pth")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.1f}%")

    print()
    print(f"  BEST TEST ACCURACY: {best_acc:.1f}%")
    print(f"  Model saved: model_float32.pth")

    # Save results
    results = {
        "model": "TactileCNN (3-layer, float32)",
        "parameters": total_params,
        "epochs": epochs,
        "best_accuracy": best_acc,
        "device": str(device),
        "history": history,
    }
    with open("results_float32.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("Next: python3 quantize_model.py")
    return model, best_acc


if __name__ == "__main__":
    train()
