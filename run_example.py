"""
Example script demonstrating the complete workflow:
1. Generate sample data
2. Train models
3. Make predictions
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        return False
    return True

def main():
    print("Property Valuation AVM - Complete Example Workflow")
    print("="*60)

    # Step 1: Create sample dataset
    if not run_command(
        "python scripts/prepare_dataset.py --create_sample --n_samples 1000 --split",
        "Step 1: Creating sample dataset with 1000 properties..."
    ):
        return

    # Step 2: Train tabular model
    if not run_command(
        "python ml/train_tabular.py --data_path data/processed/train.csv --output_dir models",
        "Step 2: Training tabular model..."
    ):
        return

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nTo train additional models (optional):")
    print("  - Image model: python ml/train_image.py --data_path data/processed/train.csv --output_dir models")
    print("  - Text model: python ml/train_text.py --data_path data/processed/train.csv --output_dir models")
    print("  - Stacker: python ml/train_stack.py --data_path data/processed/train.csv --model_dir models --output_dir models/stacker")

    print("\nTo start the API server:")
    print("  python app/main.py")
    print("\nOr with uvicorn:")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")

    print("\nAPI will be available at:")
    print("  - Swagger docs: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    print("  - Health check: http://localhost:8000/health")

if __name__ == "__main__":
    main()
