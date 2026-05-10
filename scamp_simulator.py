"""
Simulate SCAMP-5 PPA execution of the tactile CNN.
Models the actual hardware constraints:
- Analog registers (7 per pixel, noisy)
- No true multiplication (only shift-and-add)
- SIMD execution (all pixels same instruction)
- Neighbour communication (NEWS)
- Binary/ternary weights only

Compares accuracy: PC float → PC quantized → SCAMP simulated
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from train_cnn import TactileCNN, TactileDataset
from quantize_model import ternarize_weights
import json
import copy


class SCAMPSimulator:
    """
    Simulates a 256x256 SCAMP-5 Pixel Processor Array.
    Each pixel has:
      - 7 analog registers (A-G): 8-bit with noise
      - 6 digital registers (flags): 1-bit
      - NEWS communication with 4 neighbours
    """

    ANALOG_NOISE_STD = 1.5  # ADC noise per operation (out of 255)
    MAX_ANALOG_VAL = 127  # Signed 8-bit range
    MIN_ANALOG_VAL = -128
    NUM_ANALOG_REGS = 7

    def __init__(self, height=256, width=256):
        self.h = height
        self.w = width
        self.registers = {}
        self.flags = {}
        self.op_count = 0

    def _add_noise(self, data):
        """Simulate analog register noise."""
        noise = np.random.normal(0, self.ANALOG_NOISE_STD, data.shape)
        return np.clip(data + noise, self.MIN_ANALOG_VAL, self.MAX_ANALOG_VAL)

    def load_image(self, image, reg='A'):
        """Load image into analog register (simulates photoreceptor read)."""
        # Quantize to 8-bit signed
        img = np.array(image, dtype=np.float32)
        img = (img - 128).clip(self.MIN_ANALOG_VAL, self.MAX_ANALOG_VAL)
        self.registers[reg] = self._add_noise(img)
        self.op_count += 1

    def copy_reg(self, src, dst):
        """Copy register: dst = src (with noise)."""
        self.registers[dst] = self._add_noise(self.registers[src].copy())
        self.op_count += 1

    def add_reg(self, src1, src2, dst):
        """Add: dst = src1 + src2 (with noise + saturation)."""
        result = self.registers[src1] + self.registers[src2]
        result = np.clip(result, self.MIN_ANALOG_VAL, self.MAX_ANALOG_VAL)
        self.registers[dst] = self._add_noise(result)
        self.op_count += 1

    def sub_reg(self, src1, src2, dst):
        """Subtract: dst = src1 - src2 (with noise + saturation)."""
        result = self.registers[src1] - self.registers[src2]
        result = np.clip(result, self.MIN_ANALOG_VAL, self.MAX_ANALOG_VAL)
        self.registers[dst] = self._add_noise(result)
        self.op_count += 1

    def div2(self, src, dst):
        """Divide by 2: dst = src / 2 (analog shift, lossy)."""
        self.registers[dst] = self._add_noise(self.registers[src] / 2.0)
        self.op_count += 1

    def shift_north(self, src, dst):
        """Shift image up (get south neighbour's value)."""
        data = self.registers[src]
        shifted = np.zeros_like(data)
        shifted[:-1, :] = data[1:, :]  # Shift up
        self.registers[dst] = self._add_noise(shifted)
        self.op_count += 1

    def shift_south(self, src, dst):
        """Shift image down."""
        data = self.registers[src]
        shifted = np.zeros_like(data)
        shifted[1:, :] = data[:-1, :]
        self.registers[dst] = self._add_noise(shifted)
        self.op_count += 1

    def shift_east(self, src, dst):
        """Shift image right."""
        data = self.registers[src]
        shifted = np.zeros_like(data)
        shifted[:, 1:] = data[:, :-1]
        self.registers[dst] = self._add_noise(shifted)
        self.op_count += 1

    def shift_west(self, src, dst):
        """Shift image left."""
        data = self.registers[src]
        shifted = np.zeros_like(data)
        shifted[:, :-1] = data[:, 1:]
        self.registers[dst] = self._add_noise(shifted)
        self.op_count += 1

    def relu(self, src, dst):
        """ReLU: max(0, x) using threshold + flag."""
        data = self.registers[src]
        self.registers[dst] = np.maximum(0, data)
        self.op_count += 1

    def avg_pool_2x2(self, src, dst):
        """Average pooling 2x2 → halves resolution."""
        data = self.registers[src]
        h, w = data.shape
        pooled = np.zeros((h // 2, w // 2))
        for i in range(0, h - 1, 2):
            for j in range(0, w - 1, 2):
                pooled[i // 2, j // 2] = (
                    data[i, j] + data[i+1, j] + data[i, j+1] + data[i+1, j+1]
                ) / 4.0
        # Pad back to original size (SCAMP operates at full resolution)
        result = np.zeros_like(data)
        result[:h//2, :w//2] = pooled
        self.registers[dst] = self._add_noise(result)
        self.op_count += 4  # 3 adds + 1 div

    def apply_ternary_conv3x3(self, input_reg, output_reg, kernel):
        """
        Apply 3x3 convolution with ternary weights {-1, 0, +1}.
        On SCAMP: +1 = add shifted, -1 = subtract shifted, 0 = skip.
        """
        # Initialize output to zero
        self.registers[output_reg] = np.zeros((self.h, self.w))

        # Map 3x3 kernel positions to shift operations
        shifts = [
            ((-1, -1), 'NW'), ((-1, 0), 'N'), ((-1, 1), 'NE'),
            ((0, -1), 'W'),   ((0, 0), 'C'),  ((0, 1), 'E'),
            ((1, -1), 'SW'),  ((1, 0), 'S'),   ((1, 1), 'SE'),
        ]

        for idx, ((dy, dx), name) in enumerate(shifts):
            ky, kx = idx // 3, idx % 3
            w_val = kernel[ky, kx]

            if w_val == 0:
                continue  # Skip — no operation needed

            # Shift input to align
            shifted = np.roll(np.roll(self.registers[input_reg], -dy, axis=0), -dx, axis=1)

            if w_val > 0:
                self.registers[output_reg] += shifted
            else:
                self.registers[output_reg] -= shifted

            self.op_count += 2  # shift + add/sub

        # Add noise for the accumulation
        self.registers[output_reg] = self._add_noise(
            np.clip(self.registers[output_reg], self.MIN_ANALOG_VAL, self.MAX_ANALOG_VAL)
        )

    def global_sum(self, src):
        """Global sum of all pixels (for readout)."""
        self.op_count += 1
        return float(np.sum(self.registers[src]))


def simulate_inference(scamp, image, model_weights):
    """
    Run one forward pass of the ternary CNN on SCAMP simulator.
    Simplified: 3 conv layers with ternary weights → global pooling → classify.
    """
    scamp.load_image(image, 'A')

    # Layer 1: 3x3 conv (ternary), stride simulated via pooling
    kernel1 = model_weights["conv1"]  # Shape: (8, 1, 3, 3) → use first output channel
    scamp.apply_ternary_conv3x3('A', 'B', kernel1[0, 0])
    scamp.relu('B', 'B')
    scamp.avg_pool_2x2('B', 'C')  # Simulate stride-2

    # Layer 2: 3x3 conv (ternary)
    kernel2 = model_weights["conv2"]
    scamp.apply_ternary_conv3x3('C', 'D', kernel2[0, 0])
    scamp.relu('D', 'D')
    scamp.avg_pool_2x2('D', 'E')

    # Layer 3: 3x3 conv (ternary)
    kernel3 = model_weights["conv3"]
    scamp.apply_ternary_conv3x3('E', 'F', kernel3[0, 0])
    scamp.relu('F', 'F')

    # Global average → single value readout
    score = scamp.global_sum('F')
    return 1 if score > 0 else 0


def main():
    print("=" * 50)
    print("SCAMP-5 Simulator: Ternary CNN Inference")
    print("=" * 50)
    print()

    # Load ternary model weights
    device = torch.device("cpu")
    model = TactileCNN().to(device)

    if os.path.exists("model_ternary.pth"):
        model.load_state_dict(torch.load("model_ternary.pth", map_location=device))
        print("  Loaded: model_ternary.pth")
    elif os.path.exists("model_float32.pth"):
        model.load_state_dict(torch.load("model_float32.pth", map_location=device))
        # Ternarize
        for name, param in model.named_parameters():
            if "weight" in name and ("conv" in name or "fc" in name):
                param.data = ternarize_weights(param.data)
        print("  Loaded float32 and ternarized")
    else:
        print("  ERROR: No model found. Run train_cnn.py first.")
        return

    # Extract ternary kernels
    weights = {}
    for name, param in model.named_parameters():
        if "conv" in name and "weight" in name:
            key = name.split(".")[0]
            weights[key] = param.detach().numpy()
    print(f"  Extracted kernels: {list(weights.keys())}")
    print()

    # Load test images directly (not through torch transforms)
    test_dir = "dataset/test"
    test_images = []
    test_labels = []

    for label, cls in enumerate(["no_contact", "contact"]):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir))[:50]:  # 50 per class for speed
            if fname.endswith(".png"):
                from PIL import Image
                img = np.array(Image.open(os.path.join(cls_dir, fname)).convert("L"))
                test_images.append(img)
                test_labels.append(label)

    print(f"  Test images: {len(test_images)}")
    print()

    # Run SCAMP simulation
    print("  Running SCAMP-5 simulation...")
    scamp = SCAMPSimulator(256, 256)
    correct = 0
    total = 0

    for i, (img, label) in enumerate(zip(test_images, test_labels)):
        pred = simulate_inference(scamp, img, weights)
        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(test_images)} processed... "
                  f"(running acc: {100*correct/total:.1f}%)")

    scamp_acc = 100 * correct / total
    ops_per_image = scamp.op_count / total

    print()
    print("  " + "=" * 40)
    print("  FINAL RESULTS COMPARISON")
    print("  " + "=" * 40)

    # Load previous results
    float_acc = 0
    ternary_acc = 0
    if os.path.exists("results_quantization.json"):
        with open("results_quantization.json") as f:
            prev = json.load(f)
            float_acc = prev.get("float32", 0)
            ternary_acc = prev.get("ternary", 0)

    print(f"  {'Method':<25} {'Accuracy':<12} {'Hardware'}")
    print(f"  {'-'*50}")
    print(f"  {'PC Float32 CNN':<25} {float_acc:<12.1f} {'GPU/CPU'}")
    print(f"  {'PC Ternary CNN':<25} {ternary_acc:<12.1f} {'CPU'}")
    print(f"  {'SCAMP-5 Simulated':<25} {scamp_acc:<12.1f} {'PPA (256x256)'}")
    print(f"  {'-'*50}")
    print()
    print(f"  SCAMP-5 ops per image: ~{int(ops_per_image)}")
    print(f"  At 10MHz clock: ~{ops_per_image/10e6*1000:.2f} ms per inference")
    print(f"  Theoretical FPS: ~{10e6/ops_per_image:.0f}")
    print()
    print("  Note: SCAMP accuracy is lower due to:")
    print("    - Analog register noise (simulated)")
    print("    - 8-bit saturation")
    print("    - Simplified single-channel processing")
    print("    - Only first output channel used per layer")

    # Save final results
    final = {
        "float32_accuracy": float_acc,
        "ternary_pc_accuracy": ternary_acc,
        "scamp_simulated_accuracy": scamp_acc,
        "scamp_ops_per_image": int(ops_per_image),
        "scamp_estimated_fps": int(10e6 / max(ops_per_image, 1)),
        "noise_std": SCAMPSimulator.ANALOG_NOISE_STD,
        "test_samples": total,
    }
    with open("results_final.json", "w") as f:
        json.dump(final, f, indent=2)
    print("  Saved: results_final.json")


if __name__ == "__main__":
    main()
