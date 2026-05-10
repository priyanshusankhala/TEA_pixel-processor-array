"""
Generate synthetic TacTip-style tactile sensor images.
- Class 0: No contact (markers at regular grid positions)
- Class 1: Contact (markers displaced by simulated force)

Mimics a real optical tactile sensor with dot markers on elastomer.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random
import json

# Configuration
IMG_SIZE = 256  # Matches SCAMP-5 resolution
MARKER_RADIUS = 3  # Pixel radius of eaçh dot marker
GRID_SPACING = 16  # Pixels between marker centers
NUM_TRAIN = 2000  # 1000 per class
NUM_TEST = 400  # 200 per class
OUTPUT_DIR = "dataset"


def create_marker_grid():
    """Create regular grid of marker positions (rest state)."""
    markers = []
    offset = GRID_SPACING // 2
    for y in range(offset, IMG_SIZE - offset, GRID_SPACING):
        for x in range(offset, IMG_SIZE - offset, GRID_SPACING):
            markers.append((x, y))
    return markers


def apply_contact_displacement(markers, contact_center, contact_radius, max_displacement):
    """
    Simulate contact: markers near contact center get displaced outward.
    This mimics how a real TacTip sensor deforms when pressed.
    """
    displaced = []
    cx, cy = contact_center
    for (mx, my) in markers:
        dx = mx - cx
        dy = my - cy
        dist = np.sqrt(dx**2 + dy**2)

        if dist < contact_radius:
            # Markers inside contact area get displaced outward
            if dist < 1:
                angle = random.uniform(0, 2 * np.pi)
                displacement = max_displacement
            else:
                angle = np.arctan2(dy, dx)
                # Displacement decreases with distance from center
                strength = 1.0 - (dist / contact_radius)
                displacement = max_displacement * strength

            new_x = mx + displacement * np.cos(angle) + random.gauss(0, 0.5)
            new_y = my + displacement * np.sin(angle) + random.gauss(0, 0.5)
            displaced.append((new_x, new_y))
        else:
            # Markers outside contact area: tiny random noise only
            displaced.append((mx + random.gauss(0, 0.3), my + random.gauss(0, 0.3)))

    return displaced


def render_image(markers, add_noise=True):
    """Render markers as white dots on dark background (like TacTip sensor)."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Dark background with slight gradient (mimics real sensor)
    base_level = random.randint(15, 30)
    img[:] = base_level

    # Draw each marker as a white/bright circle
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    for (x, y) in markers:
        # Marker brightness varies slightly (realistic)
        brightness = random.randint(200, 255)
        x0 = int(x) - MARKER_RADIUS
        y0 = int(y) - MARKER_RADIUS
        x1 = int(x) + MARKER_RADIUS
        y1 = int(y) + MARKER_RADIUS
        draw.ellipse([x0, y0, x1, y1], fill=brightness)

    img = np.array(pil_img)

    # Add sensor noise
    if add_noise:
        noise = np.random.normal(0, 3, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def generate_no_contact():
    """Generate a no-contact image (markers at rest with tiny noise)."""
    markers = create_marker_grid()
    # Add tiny positional noise (sensor imperfection)
    noisy_markers = [(x + random.gauss(0, 0.5), y + random.gauss(0, 0.5))
                     for (x, y) in markers]
    return render_image(noisy_markers)


def generate_contact():
    """Generate a contact image (markers displaced by random press)."""
    markers = create_marker_grid()

    # Random contact point and properties
    cx = random.randint(40, IMG_SIZE - 40)
    cy = random.randint(40, IMG_SIZE - 40)
    contact_radius = random.randint(25, 70)
    max_displacement = random.uniform(3.0, 8.0)

    displaced = apply_contact_displacement(
        markers, (cx, cy), contact_radius, max_displacement
    )
    return render_image(displaced)


def main():
    print("=" * 50)
    print("Generating Synthetic TacTip Marker Dataset")
    print("=" * 50)
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Marker grid spacing: {GRID_SPACING}px")
    print(f"  Train: {NUM_TRAIN} images ({NUM_TRAIN//2} per class)")
    print(f"  Test: {NUM_TEST} images ({NUM_TEST//2} per class)")
    print()

    # Create directories
    for split in ["train", "test"]:
        for cls in ["no_contact", "contact"]:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    # Generate training data
    print("Generating training data...")
    for i in range(NUM_TRAIN // 2):
        img = generate_no_contact()
        Image.fromarray(img).save(
            os.path.join(OUTPUT_DIR, "train", "no_contact", f"nc_{i:04d}.png")
        )
        img = generate_contact()
        Image.fromarray(img).save(
            os.path.join(OUTPUT_DIR, "train", "contact", f"c_{i:04d}.png")
        )
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{NUM_TRAIN//2} pairs done")

    # Generate test data
    print("Generating test data...")
    for i in range(NUM_TEST // 2):
        img = generate_no_contact()
        Image.fromarray(img).save(
            os.path.join(OUTPUT_DIR, "test", "no_contact", f"nc_{i:04d}.png")
        )
        img = generate_contact()
        Image.fromarray(img).save(
            os.path.join(OUTPUT_DIR, "test", "contact", f"c_{i:04d}.png")
        )

    # Save dataset info
    info = {
        "img_size": IMG_SIZE,
        "marker_radius": MARKER_RADIUS,
        "grid_spacing": GRID_SPACING,
        "num_markers": len(create_marker_grid()),
        "train_total": NUM_TRAIN,
        "test_total": NUM_TEST,
        "classes": ["no_contact", "contact"],
        "description": "Synthetic TacTip-style tactile sensor marker images"
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print()
    print("Dataset generated successfully!")
    print(f"  Location: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  Markers per image: {len(create_marker_grid())}")
    print(f"  Total images: {NUM_TRAIN + NUM_TEST}")
    print()
    print("Next: python3 train_cnn.py")


if __name__ == "__main__":
    main()
