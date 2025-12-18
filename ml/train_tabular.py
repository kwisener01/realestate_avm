"""
Training script for tabular model
"""

import pandas as pd
import numpy as np
import argparse
import os
from tabular_model import TabularModel


def main(args):
    print("Loading data...")
    # Load processed data
    df = pd.read_csv(args.data_path)

    # Separate features and target
    y = df['price'].values
    X = df.drop(columns=['price', 'id', 'date'], errors='ignore')

    print(f"Training samples: {len(X)}")
    print(f"Features: {X.columns.tolist()}")

    # Initialize and train model
    print("\nTraining tabular model...")
    model = TabularModel()
    metrics = model.train(X, y, validation_split=args.val_split)

    # Print results
    print("\n" + "="*50)
    print("Training Results:")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'tabular_model.joblib')
    model.save(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train tabular model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data CSV')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save trained model')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split fraction')

    args = parser.parse_args()
    main(args)
