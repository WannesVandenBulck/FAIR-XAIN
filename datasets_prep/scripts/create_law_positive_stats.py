"""
Create law_positive.csv and update dataset_info pickle with positive instance statistics.

This script:
1. Filters law dataset for "positive" instances (those that passed the bar exam, target=0)
2. Saves them to law_positive.csv
3. Calculates feature statistics (averages and distributions) for positive instances
4. Updates the dataset_info pickle to include these statistics for comparison
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "law_dataset"
TRAIN_DATA_PATH = DATA_DIR / "train_cleaned.parquet"
PICKLE_PATH = DATA_DIR / "dataset_info"
OUTPUT_CSV = DATA_DIR / "law_positive.csv"

# Feature type definitions
CATEGORICAL_FEATURES = ['gender', 'race1', 'fulltime', 'fam_inc']
ORDINAL_FEATURES = ['fam_inc']
NOMINAL_FEATURES = ['gender', 'race1', 'fulltime']

FEATURE_MAPPINGS = {
    'gender': {0: 'Female', 1: 'Male'},
    'race1': {0: 'White', 1: 'Black', 2: 'Hispanic', 3: 'Asian', 4: 'Native American'},
    'fulltime': {1: 'Full-time', 1.0: 'Full-time', 2: 'Part-time', 2.0: 'Part-time'},
    'fam_inc': {1: 'Low', 2: 'Lower-middle', 3: 'Middle', 4: 'Upper-middle', 5: 'High', 
                1.0: 'Low', 2.0: 'Lower-middle', 3.0: 'Middle', 4.0: 'Upper-middle', 5.0: 'High'},
}

def map_value(feature_name, value):
    """Map encoded values to readable names"""
    if feature_name in FEATURE_MAPPINGS:
        mapping = FEATURE_MAPPINGS[feature_name]
        if value in mapping:
            return mapping[value]
        if float(value) in mapping:
            return mapping[float(value)]
    return str(value)

def get_feature_distribution(feature_name, data, is_categorical):
    """Calculate distribution (value counts/percentages) for categorical features"""
    if not is_categorical:
        return None
    
    value_counts = data[feature_name].value_counts().sort_index()
    total = len(data)
    distribution_str = ""
    for value, count in value_counts.items():
        pct = (count / total) * 100
        mapped_value = map_value(feature_name, value)
        distribution_str += f"Value {mapped_value}: {pct:.1f}%, "
    
    return distribution_str.rstrip(", ") if distribution_str else None

def main():
    print("="*70)
    print("Creating law_positive.csv and updating dataset_info statistics")
    print("="*70)
    
    # Load training data
    print("\nLoading training data...")
    df_law = pd.read_parquet(TRAIN_DATA_PATH)
    print(f"Total training instances: {len(df_law)}")
    
    # Filter for positive instances (bar = 0, meaning passed bar exam)
    df_positive = df_law[df_law['bar'] == 0].copy()
    print(f"Positive instances (passed bar exam, bar=0): {len(df_positive)}")
    
    # Save to CSV (without target column)
    print(f"\nSaving positive instances to {OUTPUT_CSV}...")
    df_positive_features = df_positive.drop('bar', axis=1)
    df_positive_features.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved {len(df_positive_features)} positive instances to law_positive.csv")
    
    # Load existing dataset_info pickle
    print(f"\nLoading existing pickle from {PICKLE_PATH}...")
    with open(PICKLE_PATH, 'rb') as f:
        dataset_info = pickle.load(f)
    
    feature_desc_df = dataset_info['feature_description'].copy()
    
    # Calculate statistics for positive instances for each feature
    print("\nCalculating positive instance statistics...")
    positive_stats = []
    
    for col in feature_desc_df['feature_name']:
        # Skip target variable
        if col == 'bar':
            continue
            
        is_cat = col in CATEGORICAL_FEATURES
        
        # Calculate average (only for non-categorical)
        avg_positive = df_positive[col].mean() if not is_cat else None
        
        # Calculate distribution (only for categorical)
        dist_positive = get_feature_distribution(col, df_positive, is_cat) if is_cat else None
        
        positive_stats.append({
            'feature_name': col,
            'feature_average_positive': avg_positive,
            'feature_distribution_positive': dist_positive
        })
        
        if is_cat and dist_positive:
            print(f"  {col}: {dist_positive}")
        elif avg_positive is not None:
            print(f"  {col}: avg = {avg_positive:.3f}")
    
    # Merge positive stats back into feature_desc_df
    positive_stats_df = pd.DataFrame(positive_stats)
    feature_desc_updated = feature_desc_df.merge(positive_stats_df, on='feature_name', how='left')
    
    # Update dataset_info
    dataset_info['feature_description'] = feature_desc_updated
    
    print(f"\nUpdating pickle with positive instance statistics...")
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(dataset_info, f)
    print(f"✓ Updated {PICKLE_PATH}")
    
    print("\n" + "="*70)
    print("✓ Complete! Ready to use positive instance statistics in prompts")
    print("="*70)

if __name__ == "__main__":
    main()
