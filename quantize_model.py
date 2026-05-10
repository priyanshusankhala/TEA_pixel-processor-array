"""
Quantize the trained CNN to binary/ternary weights.
- Binary: weights in {-1, +1}
- Ternary: weights in {-1, 0, +1}
- 2-bit: weights in {-1, -0.5, 0.5, 1} (approximate)

This simulates what SCAMP-5 can actually execute (no true multiply).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from train_cnn import TactileCNN, TactileDataset
import json
import copy


def binarize_weights(weight):
    """Binary: sign function → {-1, +1}"""
    return torch.sign(weight).clamp(min=-1, max=1)


def ternarize_weights(weight, threshold=0.05):
    """Ternary: {-1, 0, +1} with dead zone around zero."""
    ternary = torch.zeros_like(weight)
    ternary[weight > threshold] = 1.0
    ternary[weight < -threshold] = -1.0
    return ternary


def quantize_2bit(weight):
    """2-bit: map to {-1, -0.33, 0.33, 1}"""
    # Scale to [-1, 1]
    w_max = weight.abs().max()
    if w_max > 0:
        w_norm = weight / w_max
    else:
        return weight

    # Quantize to 4 levels
    boundaries = [-0.67, 0.0, 0.67]
    q = torch.zeros_like(w_norm)
    q[w_norm < boundaries[0]] = -1.0
    q[(w_norm >= boundaries[0]) & (w_norm < boundaries[1])] = -0.33
    q[(w_norm >= boundaries[1]) & (w_norm < boundaries[2])] = 0.33
    q[w_norm >= boundaries[2]] = 1.0

    return q * w_max

def quantize_3bit(weight):
    """3-bit: map to 8 levels between [-1, 1]"""
    w_max = weight.abs().max()
    if w_max > 0:
        w_norm = weight / w_max
    else:
        return weight

    # 8 levels — place on same device as weight
    levels = torch.tensor([-1.0, -0.71, -0.43, -0.14, 0.14, 0.43, 0.71, 1.0],
                          device=weight.device)

    # Vectorized approach (faster than for-loop)
    flat = w_norm.flatten()
    # Compute distances from each value to all levels
    distances = (flat.unsqueeze(1) - levels.unsqueeze(0)).abs()
    # Find nearest level for each value
    nearest_idx = distances.argmin(dim=1)
    flat = levels[nearest_idx]
    q = flat.reshape(w_norm.shape)

    return q * w_max


# def quantize_3bit(weight):
#     """3-bit: map to 8 levels between [-1, 1]"""
#     w_max = weight.abs().max()
#     if w_max > 0:
#         w_norm = weight / w_max
#     else:
#         return weight

#     # 8 levels: -1, -0.71, -0.43, -0.14, 0.14, 0.43, 0.71, 1
#     levels = torch.tensor([-1.0, -0.71, -0.43, -0.14, 0.14, 0.43, 0.71, 1.0])
#     q = torch.zeros_like(w_norm)
#     flat = w_norm.flatten()
#     for i in range(len(flat)):
#         idx = (levels - flat[i]).abs().argmin()
#         flat[i] = levels[idx]
#     q = flat.reshape(w_norm.shape)

#     return q * w_max


def evaluate(model, test_loader, device):
    """Evaluate model accuracy."""
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
    return 100 * correct / total

def fold_batchnorm(model):
    """
    Fold BatchNorm parameters into Conv weights.
    After folding, BN layers become identity (can be removed).
    
    For conv+BN: effective_weight = (gamma / sqrt(var + eps)) * W
                 effective_bias   = gamma * (bias - mean) / sqrt(var + eps) + beta
    """
    folded_model = copy.deepcopy(model)
    folded_model.eval()
    
    # Pairs of (conv_layer, bn_layer)
    pairs = [
        (folded_model.conv1, folded_model.bn1),
        (folded_model.conv2, folded_model.bn2),
        (folded_model.conv3, folded_model.bn3),
    ]
    
    for conv, bn in pairs:
        # BN parameters
        gamma = bn.weight.data          # scale
        beta = bn.bias.data             # shift
        mean = bn.running_mean          # running mean
        var = bn.running_var            # running variance
        eps = bn.eps
        
        # Compute folding factor
        std = torch.sqrt(var + eps)
        scale = gamma / std             # shape: [out_channels]
        
        # Fold into conv weights: new_W = scale * W
        # conv.weight shape: [out_ch, in_ch, kH, kW]
        conv.weight.data = conv.weight.data * scale.reshape(-1, 1, 1, 1)
        
        # Fold into bias: new_bias = scale * (old_bias - mean) + beta
        if conv.bias is not None:
            conv.bias.data = scale * (conv.bias.data - mean) + beta
        else:
            conv.bias = nn.Parameter(scale * (-mean) + beta)
        
        # Make BN an identity operation
        bn.weight.data.fill_(1.0)
        bn.bias.data.fill_(0.0)
        bn.running_mean.fill_(0.0)
        bn.running_var.fill_(1.0)
    
    return folded_model

def quantize_model(model, method="ternary"):
    """Apply quantization to all conv and fc layers."""
    q_model = copy.deepcopy(model)
    for name, param in q_model.named_parameters():
        if "weight" in name and ("conv" in name or "fc" in name):
            with torch.no_grad():
                if method == "binary":
                    param.copy_(binarize_weights(param.data))
                elif method == "ternary":
                    param.copy_(ternarize_weights(param.data))
                elif method == "2bit":
                    param.copy_(quantize_2bit(param.data))
                elif method == "3bit":
                    param.copy_(quantize_3bit(param.data))
    return q_model


def main():
    print("=" * 50)
    print("Model Quantization: Binary / Ternary / 2-bit / 3-bit")
    print("=" * 50)
    print()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load trained model
    model = TactileCNN().to(device)
    if not os.path.exists("model_float32.pth"):
        print("  ERROR: model_float32.pth not found. Run train_cnn.py first.")
        return
    model.load_state_dict(torch.load("model_float32.pth", map_location=device))
    model.eval()
    print("  Loaded: model_float32.pth")
    print()

    # Load test data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_set = TactileDataset("dataset/test", transform=transform)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    # Evaluate float32 baseline
    float_acc = evaluate(model, test_loader, device)
    print(f"  Float32 baseline accuracy: {float_acc:.1f}%")

    # *** KEY STEP: Fold BatchNorm into Conv weights ***
    model = fold_batchnorm(model).to(device)
    folded_acc = evaluate(model, test_loader, device)
    print(f"  After BN folding accuracy: {folded_acc:.1f}%  (should be ~same)")
    print()

    # Now quantize the BN-folded model
    results = {"float32": float_acc, "folded": folded_acc}
    methods = ["binary", "ternary", "2bit", "3bit"]

    print("  Quantization Results:")
    print("  " + "-" * 45)
    print(f"  {'Method':<12} {'Accuracy':<12} {'Drop':<10} {'Bits/Weight'}")
    print("  " + "-" * 45)
    print(f"  {'Float32':<12} {float_acc:<12.1f} {'---':<10} {'32'}")

    for method in methods:
        q_model = quantize_model(model, method).to(device)
        acc = evaluate(q_model, test_loader, device)
        drop = float_acc - acc
        bits = {"binary": "1", "ternary": "1.6", "2bit": "2", "3bit": "3"}[method]
        results[method] = acc
        print(f"  {method:<12} {acc:<12.1f} {drop:<10.1f} {bits}")

        torch.save(q_model.state_dict(), f"model_{method}.pth")

    print("  " + "-" * 45)
    print()

    with open("results_quantization.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved: results_quantization.json")
    print()
    print("  Key Takeaway for SCAMP-5:")
    print(f"    Ternary ({results.get('ternary', 0):.1f}%) uses only {{-1, 0, +1}}")
    print("    → No multiplication needed! Only add/subtract/skip")
    print("    → Directly implementable on SCAMP-5 ALU")
    print()
    print("Next: python3 scamp_simulator.py")


if __name__ == "__main__":
    main()
