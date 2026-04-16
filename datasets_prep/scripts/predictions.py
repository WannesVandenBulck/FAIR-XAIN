import pandas as pd
import pickle
import os

# Define dataset configurations
# Target encoding: 1 = adverse/bad class (denied credit, failed exam, left company, failed exams)
#                 0 = favorable/good class (approved credit, passed exam, stayed, passed exams)
datasets = {
    'credit': {
        'path': r'datasets_prep/data/credit_dataset',
        'target_col': 'credit_risk',           # Original column name in raw data
        'target_name': 'target_credit',        # Standardized column name (1=bad credit risk, 0=good)
        'output_file': 'credit_adverse'
    },
    'law': {
        'path': r'datasets_prep/data/law_dataset',
        'target_col': 'bar',                   # Original column name in raw data
        'target_name': 'target_law',           # Standardized column name (1=failed bar exam, 0=passed)
        'output_file': 'law_adverse'
    },
    'saudi': {
        'path': r'datasets_prep/data/saudi_dataset',
        'target_col': 'Attrition',             # Original column name in raw data
        'target_name': 'target_saudi',         # Standardized column name (1=left company, 0=stayed)
        'output_file': 'saudi_adverse'
    },
    'student': {
        'path': r'datasets_prep/data/student_dataset',
        'target_col': 'target',                # Original column name in raw data
        'target_name': 'target_student',       # Standardized column name (1=failed final exams, 0=passed)
        'output_file': 'student_adverse'
    }
}

def make_predictions(dataset_name, config):
    """Load model and make predictions for the test set.
    
    Predictions are filtered for class 1 (adverse/bad class).
    Target variable is renamed to standardized name (e.g., target_credit).
    The target value 1 represents the adverse outcome for each dataset.
    
    For law dataset: Uses model trained WITHOUT protected attributes (gender, race1).
    For other datasets: Uses standard model with all features.
    """
    
    dataset_path = config['path']
    target_col = config['target_col']  # Original column name
    target_name = config['target_name']  # Standardized column name
    output_file = config['output_file']
    
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Load the Random Forest model
    # For law dataset, use the model trained WITHOUT protected attributes
    if dataset_name == 'law':
        model_path = os.path.join(dataset_path, 'RF_no_protected.pkl')
        model_type = "WITHOUT protected attributes (gender, race1)"
    else:
        model_path = os.path.join(dataset_path, 'RF.pkl')
        model_type = "with all features"
    
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    print(f"  Model used: {model_type}")
    
    # Load test data
    test_path = os.path.join(dataset_path, 'test_cleaned.parquet')
    test_df = pd.read_parquet(test_path)
    
    # Separate features and target
    # Target value 1 = adverse outcome (e.g., bad credit, failed exam, left company)
    # Target value 0 = favorable outcome (e.g., good credit, passed exam, stayed)
    test_features = test_df.drop(columns=[target_col])
    test_target = test_df[target_col].rename(target_name)  # Rename to standardized name
    
    # For law dataset, create a copy without protected attributes for model prediction
    # but keep original test_features with all columns for saving later
    if dataset_name == 'law':
        protected_attributes = ['gender', 'race1']
        features_for_prediction = test_features.drop(columns=protected_attributes)
    else:
        features_for_prediction = test_features
    
    # Make predictions on test set
    test_pred = rf_model.predict(features_for_prediction)
    
    # Get prediction probabilities for class 1 (bad class)
    test_proba = rf_model.predict_proba(features_for_prediction)[:, 1]
    
    # Apply threshold tuning: use probability >= 0.4 instead of default 0.5
    # This is more permissive and catches more adverse cases
    threshold = 0.4
    adverse_mask = test_proba >= threshold
    
    # Get predicted class based on threshold (1 if >= threshold, 0 otherwise)
    test_pred_threshold = (test_proba >= threshold).astype(int)
    
    test_adverse = test_features[adverse_mask].copy()
    test_adverse['predicted_class'] = test_pred_threshold[adverse_mask]
    test_adverse['prediction_score'] = test_proba[adverse_mask]
    test_adverse[target_name] = test_target[adverse_mask]  # Use standardized target name
    
    # Capture original test set indices before resetting
    original_indices = test_features[adverse_mask].index.values
    
    # Add explicit instance_index column (sequential 0, 1, 2, ...)
    test_adverse.reset_index(drop=True, inplace=True)
    test_adverse.insert(0, 'instance_index', range(len(test_adverse)))
    test_adverse.insert(1, 'original_test_index', original_indices)
    
    # Save to CSV without index (instance_index is now an explicit column)
    output_path = os.path.join(dataset_path, f'{output_file}.csv')
    test_adverse.to_csv(output_path, index=False)
    
    print(f"  Saved {len(test_adverse)} adverse instances to {output_path}")
    
    return test_adverse

# Process all datasets
all_results = {}
for dataset_name, config in datasets.items():
    try:
        results = make_predictions(dataset_name, config)
        all_results[dataset_name] = results
    except Exception as e:
        print(f"  ERROR processing {dataset_name}: {str(e)}")

print("\n" + "="*60)
print("Prediction Summary:")
print("="*60)
for dataset_name, results in all_results.items():
    print(f"{dataset_name.upper()}: {len(results)} adverse instances")
