"""Run the complete pipeline: generate → train → quantize → simulate."""
import subprocess
import sys

steps = [
    ("Step 1: Generate Dataset", "generate_dataset.py"),
    ("Step 2: Train Baseline CNN", "train_cnn.py"),
    ("Step 3: Quantize Model", "quantize_model.py"),
    ("Step 4: SCAMP-5 Simulation", "scamp_simulator.py"),
]

print("=" * 60)
print("TACTILE MARKER TRACKING — FULL PIPELINE")
print("=" * 60)
print()

for title, script in steps:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n  ERROR in {script}. Stopping.")
        sys.exit(1)

print("\n" + "=" * 60)
print("ALL STEPS COMPLETE")
print("=" * 60)
