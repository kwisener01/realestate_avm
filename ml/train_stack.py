"""
Training script for stacker ensemble model
"""

import pandas as pd
import numpy as np
import argparse
import os
from stacker import StackerModel


def main(args):
    print("Loading data...")
    # Load processed data
    df = pd.read_csv(args.data_path)

    # Prepare data for different models
    labels = df['price'].values

    # Tabular features
    tabular_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                   'floors', 'year_built', 'year_renovated', 'latitude',
                   'longitude', 'property_type', 'neighborhood', 'condition',
                   'view_quality']
    available_tabular_cols = [col for col in tabular_cols if col in df.columns]
    tabular_data = df[available_tabular_cols]

    # Image paths
    image_paths = df['image_path'].tolist() if 'image_path' in df.columns else None

    # Text descriptions
    texts = df['description'].tolist() if 'description' in df.columns else None

    print(f"Training samples: {len(df)}")
    print(f"Tabular features: {len(available_tabular_cols)}")
    print(f"Using images: {image_paths is not None}")
    print(f"Using text: {texts is not None}")

    # Initialize stacker with pre-trained base models
    print("\nInitializing stacker ensemble...")
    stacker = StackerModel(
        tabular_model_path=os.path.join(args.model_dir, 'tabular_model.joblib'),
        image_model_path=os.path.join(args.model_dir, 'image_model.pth') if image_paths else None,
        text_model_path=os.path.join(args.model_dir, 'text_model.pth') if texts else None
    )

    # Train meta-model
    print("\nTraining stacker meta-learner...")
    metrics = stacker.train_meta_model(
        tabular_data=tabular_data,
        image_paths=image_paths,
        texts=texts,
        labels=labels,
        cv_folds=args.cv_folds
    )

    # Print results
    print("\n" + "="*50)
    print("Stacker Training Results:")
    print("="*50)
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"\nCross-Validation R² Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"\nModel Weights:")
    for model_name, weight in metrics['model_weights'].items():
        print(f"  {model_name}: {weight:.4f}")
    print(f"\nIntercept: {metrics['intercept']:.2f}")

    # Save models
    os.makedirs(args.output_dir, exist_ok=True)
    stacker.save_all_models(args.output_dir)
    print(f"\nAll models saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train stacker ensemble')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data CSV')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained base models')
    parser.add_argument('--output_dir', type=str, default='models/stacker',
                       help='Directory to save stacker models')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')

    args = parser.parse_args()
    main(args)
