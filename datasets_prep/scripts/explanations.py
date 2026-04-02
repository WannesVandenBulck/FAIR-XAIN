import pandas as pd
import numpy as np
import pickle
import os
import shap
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Configuration
NUM_COUNTERFACTUALS = 3  # Change this to generate different numbers of counterfactuals

# Define dataset configurations
datasets = {
    'credit': {
        'path': r'datasets_prep/data/credit_dataset',
        'target_col': 'credit_risk',
        'adverse_file': 'credit_adverse.csv'
    },
    'law': {
        'path': r'datasets_prep/data/law_dataset',
        'target_col': 'bar',
        'adverse_file': 'law_adverse.csv'
    },
    'saudi': {
        'path': r'datasets_prep/data/saudi_dataset',
        'target_col': 'Attrition',
        'adverse_file': 'saudi_adverse.csv'
    },
    'student': {
        'path': r'datasets_prep/data/student_dataset',
        'target_col': 'target',
        'adverse_file': 'student_adverse.csv'
    }
}

def generate_explanations(dataset_name, config, num_cf=NUM_COUNTERFACTUALS):
    """Generate SHAP values and counterfactuals for adverse predictions."""
        
    dataset_path = config['path']
    target_col = config['target_col']
    adverse_file = config['adverse_file']
    
    # Load adverse predictions
    adverse_path = os.path.join(dataset_path, adverse_file)
    adverse_df = pd.read_csv(adverse_path, index_col=0)
        
    # Load the RF model
    model_path = os.path.join(dataset_path, 'RF.pkl')
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    # Load test data for context
    test_path = os.path.join(dataset_path, 'test_cleaned.parquet')
    test_df = pd.read_parquet(test_path)
    test_features = test_df.drop(columns=[target_col])
    
    # Get only the feature columns (exclude prediction-related columns)
    feature_cols = [col for col in adverse_df.columns if col not in 
                    ['predicted_class', 'prediction_score', 'actual_target']]
    adverse_features = adverse_df[feature_cols].copy()
    
    # ===== SHAP VALUES =====
    # Use TreeSHAP for RF models
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(adverse_features)
    
    # For binary classification, TreeExplainer returns shap values as a list [class_0, class_1]
    # Use class 1 (bad class) shap values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Handle different shapes - if still 3D or higher, take mean across extra dimensions
    while shap_values.ndim > 2:
        shap_values = shap_values.mean(axis=-1)
    
    # Ensure it's 2D
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)
    
    # Create dataframe with SHAP values (with instance index)
    shap_df = pd.DataFrame(
        shap_values,
        columns=[f'SHAP_{col}' for col in feature_cols],
        index=adverse_features.index
    )
    shap_df.index.name = 'instance_index'
        
    # Save SHAP values
    shap_output_path = os.path.join(dataset_path, f'{dataset_name}_shap.csv')
    shap_df.to_csv(shap_output_path)
    
    # ===== COUNTERFACTUALS (using NICE Algorithm) =====    
    # Get predictions for test set to identify good class instances
    test_pred = rf_model.predict(test_features)
    
    # Find instances with opposite class (good class, 0)
    good_class_indices = np.where(test_pred == 0)[0]
    good_class_instances = test_features.iloc[good_class_indices]
    
    # Fit KNN on good class instances for NICE algorithm
    all_counterfactuals = []
    
    if len(good_class_instances) > 0:
        nbrs = NearestNeighbors(n_neighbors=min(num_cf, len(good_class_instances)), 
                                algorithm='ball_tree').fit(good_class_instances.values)
        
        # Generate counterfactuals for each adverse instance
        for idx, (index, instance) in enumerate(adverse_features.iterrows()):
            if (idx + 1) % max(1, len(adverse_features) // 10) == 0:
                print(f"    Processing instance {idx + 1}/{len(adverse_features)}")
            
            try:
                # Find nearest neighbors in good class using NICE
                distances, indices = nbrs.kneighbors(instance.values.reshape(1, -1))
                
                # Extract the nearest instances as counterfactuals
                for cf_idx, neighbor_idx in enumerate(indices[0]):
                    cf_instance = good_class_instances.iloc[neighbor_idx].copy()
                    cf_instance['instance_index'] = index
                    cf_instance['CF_number'] = cf_idx + 1
                    cf_instance['distance_to_original'] = distances[0][cf_idx]
                    all_counterfactuals.append(cf_instance)
                    
            except Exception as e:
                print(f"    Warning: Could not generate counterfactuals for instance {index}: {str(e)}")
        
        print(f"  Generated {len(all_counterfactuals)} total counterfactuals using NICE")
        
        # Create counterfactual dataframe
        cf_df = pd.DataFrame(all_counterfactuals)
        
        # Reorganize columns: instance_index first, then CF_number, then features
        cols = ['instance_index', 'CF_number', 'distance_to_original']
        feature_cf_cols = [col for col in cf_df.columns if col not in cols]
        cf_df = cf_df[cols + feature_cf_cols]
        
        # Save counterfactuals
        cf_output_path = os.path.join(dataset_path, f'{dataset_name}_counterfactual.csv')
        cf_df.to_csv(cf_output_path, index=False)
        print(f"  Saved counterfactuals to {cf_output_path}")
    else:
        print(f"  Warning: No instances with good class (0) found for NICE algorithm")

def main():
    """Main function to generate all explanations and save to individual files."""
    
    for dataset_name, config in datasets.items():
        try:
            generate_explanations(dataset_name, config, NUM_COUNTERFACTUALS)
        except Exception as e:
            print(f"  ERROR processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
if __name__ == "__main__":
    main()
