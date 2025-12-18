"""
Training script for image model
"""

import pandas as pd
import numpy as np
import argparse
import os
from image_model import ImageModelWrapper


def main(args):
    print("Loading data...")
    # Load processed data with image paths
    df = pd.read_csv(args.data_path)

    # Get image paths and labels
    image_paths = df['image_path'].tolist()
    labels = df['price'].values

    print(f"Training samples: {len(image_paths)}")

    # Initialize and train model
    print("\nTraining image model...")
    model = ImageModelWrapper()
    metrics = model.train(
        image_paths,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.val_split
    )

    # Print results
    print("\n" + "="*50)
    print("Training Results:")
    print("="*50)
    print(f"Final Validation MAE: {metrics['val_mae']:.2f}")
    print(f"Final Validation MAPE: {metrics['val_mape']:.2f}%")
    print(f"Best Validation Loss: {metrics['best_val_loss']:.4f}")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'image_model.pth')
    model.save(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data CSV with image paths')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split fraction')

    args = parser.parse_args()
    main(args)
