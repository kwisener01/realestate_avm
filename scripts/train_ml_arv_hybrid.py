"""
Train the enhanced ML ARV model on all available county data.
This creates the model used in the hybrid ML+Zillow approach.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml_arv_service import MLARVPredictor
import pandas as pd


def main():
    print("\n" + "="*80)
    print("TRAINING HYBRID ML ARV MODEL")
    print("="*80)

    # Define paths to all county data
    data_dir = Path(__file__).parent.parent / 'data'
    county_files = [
        data_dir / 'decatur_properties.csv',
        # Add other county files as they become available
        # data_dir / 'cobb_county_properties.csv',
        # data_dir / 'fulton_county_properties.csv',
        # data_dir / 'clayton_county_properties.csv',
        # data_dir / 'gwinnett_county_properties.csv',
    ]

    # Filter for existing files
    existing_files = [str(f) for f in county_files if f.exists()]

    if not existing_files:
        print("\n  ERROR: No county data files found!")
        print(f"  Looking in: {data_dir}")
        return

    print(f"\n  Found {len(existing_files)} county data files")

    # Initialize predictor
    predictor = MLARVPredictor()

    # Prepare training data
    X, y = predictor.prepare_training_data(existing_files)

    # Train ensemble
    performances = predictor.train_ensemble(X, y)

    # Show feature importance for best tree-based model
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (Gradient Boosting)")
    print("="*80)

    gb_model = predictor.models['gradient_boosting']
    importances = gb_model.feature_importances_
    feature_importance = sorted(
        zip(predictor.feature_columns, importances),
        key=lambda x: x[1],
        reverse=True
    )

    for feat, imp in feature_importance:
        bar = '#' * int(imp * 50)  # Scale to 50 chars max
        print(f"  {feat:<25} {imp:.4f} {bar}")

    # Save model
    model_dir = Path(__file__).parent.parent / 'models'
    model_path = model_dir / 'ml_arv_hybrid.pkl'
    predictor.save_model(str(model_path))

    # Test prediction
    print("\n" + "="*80)
    print("TESTING PREDICTION")
    print("="*80)

    test_property = {
        'Building Sqft': 2000,
        'Bedrooms': 3,
        'Total Bathrooms': 2,
        'Lot Size Sqft': 8000,
        'Effective Year Built': 1995,
        'Total Assessed Value': 200000,
        'City': 'Decatur'
    }

    result = predictor.predict_with_confidence(test_property)

    print(f"\n  Test Property:")
    print(f"    {test_property['Building Sqft']} sqft, {test_property['Bedrooms']} bed, {test_property['Total Bathrooms']} bath")
    print(f"    Built: {test_property['Effective Year Built']}, Assessed: ${test_property['Total Assessed Value']:,}")
    print(f"\n  ML ARV Prediction:")
    print(f"    Primary: ${result['arv_prediction']:,}")
    print(f"    Range: ${result['arv_lower']:,} - ${result['arv_upper']:,}")
    print(f"    Confidence: {result['confidence']}")
    print(f"\n  Model Predictions:")
    for model_name, pred in result['model_predictions'].items():
        print(f"    {model_name:<20} ${pred:,}")

    print("\n" + "="*80)
    print("[SUCCESS] MODEL TRAINING COMPLETE")
    print("="*80)
    print(f"\n  Model saved to: {model_path}")
    print(f"  Ready for hybrid ML+Zillow predictions!")
    print("\n")


if __name__ == '__main__':
    main()
