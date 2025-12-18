"""
Simple training script for tabular-only model
Works without PyTorch/image dependencies
"""

import pandas as pd
import sys
import os

# Add ml directory to path
sys.path.append('ml')

from ml.tabular_model import TabularModel

def main():
    print("="*60)
    print("TRAINING REAL ESTATE AVM MODEL")
    print("="*60)

    # Load training data
    print("\nLoading training data...")
    df = pd.read_csv('data/processed/training_data.csv')

    # Prepare features and target
    y = df['price'].values
    feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                   'floors', 'year_built', 'year_renovated', 'latitude',
                   'longitude', 'property_type', 'neighborhood', 'condition',
                   'view_quality']
    X = df[feature_cols]

    print(f"[OK] Loaded {len(df)} properties")
    print(f"[OK] Features: {len(feature_cols)}")
    print(f"[OK] Price range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"[OK] Mean price: ${y.mean():,.0f}")

    # Train model
    print("\nTraining model...")
    print("-" * 60)
    model = TabularModel()
    metrics = model.train(X, y, validation_split=0.2)

    # Display results
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Validation R² Score: {metrics['val_r2']:.4f} ({metrics['val_r2']*100:.1f}%)")
    print(f"Validation MAE: ${metrics['val_mae']:,.0f}")
    print(f"Validation MAPE: {metrics['val_mape']:.2f}%")
    print()
    print("Interpretation:")
    if metrics['val_r2'] > 0.7:
        print("[OK] Excellent - Model explains >70% of price variance")
    elif metrics['val_r2'] > 0.5:
        print("[OK] Good - Model explains >50% of price variance")
    else:
        print("[!] Fair - Model performance could be improved")

    print(f"\nOn average, predictions are ${metrics['val_mae']:,.0f} off from actual price")

    # Save model
    os.makedirs('models/stacker', exist_ok=True)
    model_path = 'models/stacker/tabular_model.joblib'
    model.save(model_path)
    print(f"\n[OK] Model saved to: {model_path}")

    # Test prediction
    print("\n" + "="*60)
    print("TESTING PREDICTION")
    print("="*60)
    test_property = X.iloc[0:1]
    actual_price = y[0]
    predicted = model.predict(test_property)[0]
    error = abs(predicted - actual_price)
    error_pct = (error / actual_price) * 100

    print(f"Test Property Features:")
    for col in feature_cols[:5]:  # Show first 5 features
        print(f"  {col}: {test_property[col].values[0]}")
    print(f"  ...")
    print(f"\nActual Price: ${actual_price:,.0f}")
    print(f"Predicted Price: ${predicted:,.0f}")
    print(f"Error: ${error:,.0f} ({error_pct:.1f}%)")

    print("\n" + "="*60)
    print("[SUCCESS] MODEL READY FOR DEPLOYMENT!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test the API locally: python app/main.py")
    print("2. Deploy to Railway (models will be included)")
    print("3. Make predictions via web UI or API")


if __name__ == '__main__':
    main()
