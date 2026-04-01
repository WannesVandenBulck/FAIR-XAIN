import pandas as pd
import pickle
import os

# Define dataset configurations
datasets = {
    'credit': {
        'path': r'datasets_prep/data/credit_dataset',
        'target_col': 'credit_risk',
        'output_file': 'credit_adverse'
    },
    'law': {
        'path': r'datasets_prep/data/law_dataset',
        'target_col': 'bar',
        'output_file': 'law_adverse'
    },
    'saudi': {
        'path': r'datasets_prep/data/saudi_dataset',
        'target_col': 'Attrition',
        'output_file': 'saudi_adverse'
    },
    'student': {
        'path': r'datasets_prep/data/student_dataset',
        'target_col': 'target',
        'output_file': 'student_adverse'
    }
}

def make_predictions(dataset_name, config):
    """Load model and make predictions for the test set."""
    
    dataset_path = config['path']
    target_col = config['target_col']
    output_file = config['output_file']
    
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Load the Random Forest model
    model_path = os.path.join(dataset_path, 'RF.pkl')
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    # Load test data
    test_path = os.path.join(dataset_path, 'test_cleaned.parquet')
    test_df = pd.read_parquet(test_path)
    
    # Separate features and target
    test_features = test_df.drop(columns=[target_col])
    test_target = test_df[target_col]
    
    # Make predictions on test set
    test_pred = rf_model.predict(test_features)
    
    # Get prediction probabilities for class 1 (bad class)
    test_proba = rf_model.predict_proba(test_features)[:, 1]
    
    # Filter for class 1 predictions (bad class) from test set
    test_adverse = test_features[test_pred == 1].copy()
    test_adverse['predicted_class'] = test_pred[test_pred == 1]
    test_adverse['prediction_score'] = test_proba[test_pred == 1]
    test_adverse['actual_target'] = test_target[test_pred == 1]
    
    # Save to CSV
    output_path = os.path.join(dataset_path, f'{output_file}.csv')
    test_adverse.to_csv(output_path)
    
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
