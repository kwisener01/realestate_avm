"""
Test the hybrid ML+Zillow ARV prediction system.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ml_arv_service import MLARVPredictor, compare_with_zillow
from pathlib import Path


def test_predictions():
    print("\n" + "="*80)
    print("TESTING HYBRID ML+ZILLOW ARV SYSTEM")
    print("="*80)

    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'ml_arv_hybrid.pkl'

    if not model_path.exists():
        print(f"\nERROR: Model not found at {model_path}")
        print("Please run scripts/train_ml_arv_hybrid.py first!")
        return

    predictor = MLARVPredictor()
    predictor.load_model(str(model_path))

    # Test properties with different scenarios
    test_properties = [
        {
            'name': "Small Starter Home - Decatur",
            'data': {
                'Building Sqft': 1200,
                'Bedrooms': 2,
                'Total Bathrooms': 1,
                'Lot Size Sqft': 5000,
                'Effective Year Built': 1960,
                'Total Assessed Value': 150000,
                'City': 'Decatur'
            },
            'zestimate': 250000,
            'list_price': 180000
        },
        {
            'name': "Family Home - Decatur",
            'data': {
                'Building Sqft': 2500,
                'Bedrooms': 4,
                'Total Bathrooms': 2.5,
                'Lot Size Sqft': 10000,
                'Effective Year Built': 1990,
                'Total Assessed Value': 300000,
                'City': 'Decatur'
            },
            'zestimate': 550000,
            'list_price': 420000
        },
        {
            'name': "Luxury Property - Decatur",
            'data': {
                'Building Sqft': 4000,
                'Bedrooms': 5,
                'Total Bathrooms': 4,
                'Lot Size Sqft': 15000,
                'Effective Year Built': 2010,
                'Total Assessed Value': 600000,
                'City': 'Decatur'
            },
            'zestimate': 950000,
            'list_price': 750000
        },
        {
            'name': "Distressed Property - Unknown City",
            'data': {
                'Building Sqft': 1800,
                'Bedrooms': 3,
                'Total Bathrooms': 2,
                'Lot Size Sqft': 8000,
                'Effective Year Built': 1975,
                'Total Assessed Value': 120000,
                'City': 'Unknown'
            },
            'zestimate': 280000,
            'list_price': 150000
        },
    ]

    print("\n" + "-"*80)
    print("TESTING PREDICTIONS")
    print("-"*80)

    for prop in test_properties:
        print(f"\n{'='*80}")
        print(f"Property: {prop['name']}")
        print(f"{'='*80}")

        print(f"\nProperty Details:")
        print(f"  {prop['data']['Building Sqft']} sqft, {prop['data']['Bedrooms']} bed, {prop['data']['Total Bathrooms']} bath")
        print(f"  Built: {prop['data']['Effective Year Built']}, City: {prop['data']['City']}")
        print(f"  Assessed Value: ${prop['data']['Total Assessed Value']:,}")
        print(f"  List Price: ${prop['list_price']:,}")
        print(f"  Zillow Zestimate: ${prop['zestimate']:,}")

        # Get ML prediction
        ml_result = predictor.predict_with_confidence(prop['data'])

        print(f"\nML ARV Prediction:")
        print(f"  Primary: ${ml_result['arv_prediction']:,}")
        print(f"  Range: ${ml_result['arv_lower']:,} - ${ml_result['arv_upper']:,}")
        print(f"  Confidence: {ml_result['confidence']}")
        print(f"  CV: {ml_result['cv']:.1%}")

        print(f"\n  Model Breakdown:")
        for model_name, pred in ml_result['model_predictions'].items():
            print(f"    {model_name:<20} ${pred:,}")

        # Compare with Zillow
        comparison = compare_with_zillow(ml_result['arv_prediction'], prop['zestimate'])

        print(f"\nHybrid Analysis:")
        print(f"  Zillow ARV (80%): ${int(prop['zestimate'] * 0.80):,}")
        print(f"  ML ARV: ${ml_result['arv_prediction']:,}")
        print(f"  Primary ARV (Hybrid): ${comparison['primary_arv']:,}")
        print(f"  Agreement: {comparison['agreement']}")
        print(f"  Confidence: {comparison['confidence']}")

        if comparison['difference_pct']:
            print(f"  Difference: {comparison['difference_pct']:.1f}%")

        if comparison['flag_review']:
            print(f"  [!] FLAGGED FOR MANUAL REVIEW")

        # Deal analysis
        print(f"\nDeal Analysis:")
        mao = comparison['primary_arv'] * 0.50
        spread = comparison['primary_arv'] - prop['list_price']
        roi = (spread / prop['list_price']) * 100 if prop['list_price'] > 0 else 0

        print(f"  Maximum Allowable Offer: ${int(mao):,}")
        print(f"  Potential Spread: ${int(spread):,}")
        print(f"  Potential ROI: {roi:.1f}%")

        if prop['list_price'] <= mao:
            print(f"  Status: [GOOD DEAL] List price below MAO!")
        elif prop['list_price'] <= mao * 1.1:
            print(f"  Status: [MAYBE] List price slightly above MAO")
        else:
            print(f"  Status: [NO DEAL] List price too high")

    print("\n" + "="*80)
    print("[SUCCESS] HYBRID ARV TESTING COMPLETE")
    print("="*80)
    print("\nKey Insights:")
    print("- ML model provides confidence intervals for risk assessment")
    print("- Hybrid approach averages ML and Zillow when they agree")
    print("- Properties flagged when models disagree significantly")
    print("- Confidence scoring helps prioritize manual review")
    print("\n")


if __name__ == '__main__':
    test_predictions()
