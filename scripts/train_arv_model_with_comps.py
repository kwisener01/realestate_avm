import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import re
import pickle

class ImprovedARVModel:
    def __init__(self):
        self.model = None
        self.city_encoder = LabelEncoder()
        self.feature_columns = []

    def clean_price(self, price_str):
        """Convert price string to numeric"""
        if pd.isna(price_str):
            return None
        cleaned = re.sub(r'[,$]', '', str(price_str))
        try:
            return float(cleaned)
        except:
            return None

    def prepare_training_data(self, df_decatur):
        """Prepare Decatur data for training"""
        print("\nPreparing training data from Decatur comps...")

        # Filter for valid sales with complete data
        df = df_decatur.copy()

        # Keep only properties with Last Sale Amount (actual ARV)
        df = df[df['Last Sale Amount'].notna()].copy()

        # Keep only properties with key features
        df = df[df['Building Sqft'] > 0].copy()
        df = df[df['Bedrooms'] > 0].copy()
        df = df[df['Total Assessed Value'] > 0].copy()

        print(f"  Training set: {len(df)} properties with complete data")

        # Create features
        df['sqft'] = df['Building Sqft']
        df['beds'] = df['Bedrooms']
        df['baths'] = df['Total Bathrooms'].fillna(2.0)
        df['assessed_value'] = df['Total Assessed Value']
        df['lot_size'] = df['Lot Size Sqft']
        df['year_built'] = df['Effective Year Built'].fillna(1990)
        df['age'] = 2025 - df['year_built']

        # Target variable: Last Sale Amount (actual ARV)
        df['arv'] = df['Last Sale Amount']

        # Encode city
        df['city_encoded'] = self.city_encoder.fit_transform(df['City'].fillna('Unknown'))

        # Calculate price per sqft
        df['assessed_per_sqft'] = df['assessed_value'] / df['sqft']

        # Select features
        self.feature_columns = ['sqft', 'beds', 'baths', 'assessed_value',
                               'lot_size', 'age', 'city_encoded', 'assessed_per_sqft']

        X = df[self.feature_columns]
        y = df['arv']

        print(f"  Features: {self.feature_columns}")
        print(f"  Target: ARV (Last Sale Amount)")
        print(f"\n  ARV Statistics:")
        print(f"    Mean: ${y.mean():,.0f}")
        print(f"    Median: ${y.median():,.0f}")
        print(f"    Min: ${y.min():,.0f}")
        print(f"    Max: ${y.max():,.0f}")

        return X, y, df

    def train_model(self, X, y):
        """Train Random Forest model"""
        print("\nTraining Random Forest model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        print(f"\n  Model Performance:")
        print(f"    Train MAE: ${train_mae:,.0f}")
        print(f"    Test MAE: ${test_mae:,.0f}")
        print(f"    Train R²: {train_r2:.3f}")
        print(f"    Test R²: {test_r2:.3f}")

        # Feature importance
        print(f"\n  Feature Importance:")
        importances = self.model.feature_importances_
        for feat, imp in sorted(zip(self.feature_columns, importances),
                               key=lambda x: x[1], reverse=True):
            print(f"    {feat}: {imp:.3f}")

        return test_mae, test_r2

    def predict_arv(self, df_target):
        """Predict ARV for target properties"""
        print("\nPredicting ARV for target properties...")

        df = df_target.copy()

        # Clean list price
        df['list_price_clean'] = df['List Price'].apply(self.clean_price)

        # For properties without features, use estimated values
        # Based on list price and area
        df['sqft_est'] = df['list_price_clean'] * 0.006  # Rough estimate: ~$165/sqft
        df['beds_est'] = np.clip((df['list_price_clean'] / 50000).fillna(3), 2, 5)
        df['baths_est'] = np.clip((df['list_price_clean'] / 80000).fillna(2), 1, 3)
        df['assessed_value_est'] = df['list_price_clean'] * 0.8
        df['lot_size_est'] = 10000  # Average lot
        df['age_est'] = 30  # Average age

        # Encode cities (handle unseen cities)
        df['city_encoded'] = df['City'].apply(
            lambda x: self.city_encoder.transform([x])[0]
                     if x in self.city_encoder.classes_
                     else -1
        )

        # Calculate assessed per sqft
        df['assessed_per_sqft_est'] = df['assessed_value_est'] / df['sqft_est']

        # Create feature matrix
        X_pred = df[['sqft_est', 'beds_est', 'baths_est', 'assessed_value_est',
                     'lot_size_est', 'age_est', 'city_encoded', 'assessed_per_sqft_est']]
        X_pred.columns = self.feature_columns

        # Predict
        df['ARV_ML'] = self.model.predict(X_pred).astype(int)

        print(f"  Predicted ARV for {len(df)} properties")
        print(f"\n  Predicted ARV Statistics:")
        print(f"    Mean: ${df['ARV_ML'].mean():,.0f}")
        print(f"    Median: ${df['ARV_ML'].median():,.0f}")
        print(f"    Min: ${df['ARV_ML'].min():,.0f}")
        print(f"    Max: ${df['ARV_ML'].max():,.0f}")

        return df

def main():
    print("=" * 80)
    print("TRAINING ARV MODEL WITH ACTUAL SALES COMPS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df_decatur = pd.read_csv('../data/decatur_properties.csv')
    df_target = pd.read_csv('../data/listing_agents.csv')

    print(f"  Decatur training data: {len(df_decatur)} properties")
    print(f"  Target properties: {len(df_target)} properties")

    # Initialize model
    model = ImprovedARVModel()

    # Prepare and train
    X, y, df_train = model.prepare_training_data(df_decatur)
    test_mae, test_r2 = model.train_model(X, y)

    # Predict ARV for target properties
    df_results = model.predict_arv(df_target)

    # Calculate Deal Status
    print("\nCalculating Deal Status...")
    df_results['Deal_Status_ML'] = df_results.apply(
        lambda row: 'Deal' if row['list_price_clean'] <= (row['ARV_ML'] * 0.5)
                    else 'No Deal',
        axis=1
    )

    deal_count = (df_results['Deal_Status_ML'] == 'Deal').sum()
    print(f"  Deals found: {deal_count} ({deal_count/len(df_results)*100:.1f}%)")

    # Compare with old model
    df_old = pd.read_csv('../data/arv_results_full.csv')
    df_results['ARV_Old'] = df_old['ARV'].apply(lambda x: int(re.sub(r'[,$]', '', str(x))) if pd.notna(x) else 0)

    # Show comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON:")
    print("=" * 80)
    print(f"\n{'Property':<40} {'List Price':<12} {'Old ARV':<12} {'ML ARV':<12} {'Old':<10} {'New':<10}")
    print("-" * 95)

    for idx, row in df_results.head(10).iterrows():
        print(f"{row['Address'][:40]:<40} "
              f"${row['list_price_clean']:>10,.0f}  "
              f"${row['ARV_Old']:>10,.0f}  "
              f"${row['ARV_ML']:>10,.0f}  "
              f"{df_old.loc[idx, 'Deal Status']:<10} "
              f"{row['Deal_Status_ML']:<10}")

    # Save results
    output_file = '../data/arv_ml_results.csv'
    df_results.to_csv(output_file, index=False)

    print(f"\n{'-' * 80}")
    print(f"Results saved to: {output_file}")

    # Save model
    model_file = '../models/arv_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_file}")

    print("\n" + "=" * 80)
    print(f"Model Accuracy: MAE = ${test_mae:,.0f}, R² = {test_r2:.3f}")
    print(f"New Deals Found: {deal_count} properties")
    print("=" * 80)

if __name__ == '__main__':
    main()
