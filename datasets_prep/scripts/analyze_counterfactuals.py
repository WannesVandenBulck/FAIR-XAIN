import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict, Counter

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Configuration
datasets = {
    'credit': {
        'path': r'datasets_prep/data/credit_dataset',
        'counterfactual_file': 'credit_counterfactual.csv',
        'adverse_file': 'credit_adverse.csv',
        'test_file': 'test_cleaned.parquet'
    },
    'law': {
        'path': r'datasets_prep/data/law_dataset',
        'counterfactual_file': 'law_counterfactual.csv',
        'adverse_file': 'law_adverse.csv',
        'test_file': 'test_cleaned.parquet'
    },
    'saudi': {
        'path': r'datasets_prep/data/saudi_dataset',
        'counterfactual_file': 'saudi_counterfactual.csv',
        'adverse_file': 'saudi_adverse.csv',
        'test_file': 'test_cleaned.parquet'
    },
    'student': {
        'path': r'datasets_prep/data/student_dataset',
        'counterfactual_file': 'student_counterfactual.csv',
        'adverse_file': 'student_adverse.csv',
        'test_file': 'test_cleaned.parquet'
    }
}

def is_numeric(series):
    """Check if a pandas series contains numeric data."""
    try:
        pd.to_numeric(series, errors='coerce')
        # If more than 80% can be converted to numeric, consider it numeric
        return (pd.to_numeric(series, errors='coerce').notna().sum() / len(series)) > 0.8
    except:
        return False

def analyze_counterfactuals(dataset_name, config):
    """Analyze counterfactual changes for each instance."""
    
    dataset_path = config['path']
    cf_file = config['counterfactual_file']
    adverse_file = config['adverse_file']
    
    # Load dataframes
    cf_path = os.path.join(dataset_path, cf_file)
    adverse_path = os.path.join(dataset_path, adverse_file)
    
    cf_df = pd.read_csv(cf_path)
    adverse_df = pd.read_csv(adverse_path)
    
    print(f"\n{dataset_name.upper()}")
    print(f"  Loaded {len(cf_df)} counterfactuals for {len(adverse_df)} adverse instances")
    
    # Metadata columns to exclude from analysis
    metadata_cols = {'instance_index', 'original_test_index', 'CF_number', 'distance_to_original'}
    
    # Get feature columns (exclude metadata)
    feature_cols = [col for col in cf_df.columns if col not in metadata_cols]
    
    # Determine which features are numeric
    numeric_features = set()
    for col in feature_cols:
        if is_numeric(cf_df[col]):
            numeric_features.add(col)
    
    print(f"  Analyzing {len(feature_cols)} features ({len(numeric_features)} numeric, {len(feature_cols) - len(numeric_features)} categorical)")
    
    # Store results for all instances
    results = {}
    
    # Process each instance
    for instance_idx in adverse_df['instance_index'].unique():
        # Get original adverse instance
        adverse_row = adverse_df[adverse_df['instance_index'] == instance_idx].iloc[0]
        
        # Get all counterfactuals for this instance
        cf_instances = cf_df[cf_df['instance_index'] == instance_idx]
        num_cfs = len(cf_instances)
        
        instance_stats = {}
        
        # Analyze each feature
        for feature in feature_cols:
            original_value = adverse_row[feature]
            cf_values = cf_instances[feature].values
            
            # Count how many changed
            changed_mask = cf_values != original_value
            num_changed = changed_mask.sum()
            percentage_changed = (num_changed / num_cfs) * 100 if num_cfs > 0 else 0
            
            feature_stats = {
                'percentage_changed': round(percentage_changed, 2),
                'num_changed': num_changed,
                'num_total': num_cfs,
                'original_value': original_value
            }
            
            if num_changed > 0:
                changed_values = cf_values[changed_mask]
                
                if feature in numeric_features:
                    # Numeric feature analysis
                    changes = changed_values.astype(float) - float(original_value)
                    feature_stats['average_change'] = round(float(np.mean(changes)), 4)
                    feature_stats['min_change'] = round(float(np.min(changes)), 4)
                    feature_stats['max_change'] = round(float(np.max(changes)), 4)
                    feature_stats['std_change'] = round(float(np.std(changes)), 4)
                    feature_stats['changes'] = [round(float(c), 4) for c in changes]
                else:
                    # Categorical feature analysis
                    value_counts = Counter(changed_values)
                    feature_stats['distribution'] = dict(value_counts)
                    feature_stats['unique_values'] = list(value_counts.keys())
            
            instance_stats[feature] = feature_stats
        
        results[f"instance_{instance_idx}"] = {
            'num_counterfactuals': num_cfs,
            'features': instance_stats
        }
    
    # Save results to JSON
    output_path = os.path.join(dataset_path, f'{dataset_name}_counterfactual_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"  Saved analysis to {output_path}")
    
    # Print sample statistics
    print(f"  Sample statistics (first instance):")
    first_instance = list(results.values())[0]
    print(f"    Total counterfactuals: {first_instance['num_counterfactuals']}")
    
    # Show a few features
    for feature_name, stats in list(first_instance['features'].items())[:3]:
        print(f"    {feature_name}:")
        print(f"      Changed in {stats['percentage_changed']}% of cases ({stats['num_changed']}/{stats['num_total']})")
        if 'average_change' in stats:
            print(f"      Average change: {stats['average_change']}")
        elif 'distribution' in stats:
            print(f"      Distribution: {stats['distribution']}")
    
    return results

def create_summary_csv(dataset_name, config, results):
    """Create a summary CSV with one row per feature per instance."""
    
    dataset_path = config['path']
    
    rows = []
    for instance_key, instance_data in results.items():
        instance_num = int(instance_key.split('_')[1])
        
        for feature_name, feature_stats in instance_data['features'].items():
            row = {
                'instance_index': instance_num,
                'feature_name': feature_name,
                'percentage_changed': feature_stats['percentage_changed'],
                'num_changed': int(feature_stats['num_changed']),
                'num_total': int(feature_stats['num_total']),
                'original_value': float(feature_stats['original_value']) if isinstance(feature_stats['original_value'], (int, float, np.integer, np.floating)) else str(feature_stats['original_value'])
            }
            
            # Add numeric-specific columns
            if 'average_change' in feature_stats:
                row['feature_type'] = 'numeric'
                row['average_change'] = feature_stats['average_change']
                row['min_change'] = feature_stats['min_change']
                row['max_change'] = feature_stats['max_change']
                row['std_change'] = feature_stats['std_change']
            else:
                row['feature_type'] = 'categorical'
                row['distribution'] = json.dumps(feature_stats.get('distribution', {}), cls=NumpyEncoder)
                row['unique_values'] = ','.join(str(v) for v in feature_stats.get('unique_values', []))
            
            rows.append(row)
    
    # Create dataframe
    summary_df = pd.DataFrame(rows)
    
    # Sort by instance and feature
    summary_df = summary_df.sort_values(['instance_index', 'feature_name']).reset_index(drop=True)
    
    # Save to CSV
    output_path = os.path.join(dataset_path, f'{dataset_name}_counterfactual_summary.csv')
    summary_df.to_csv(output_path, index=False)
    
    print(f"  Summary CSV saved to {output_path}")

def main():
    """Main function to analyze all counterfactual files."""
    
    print("=" * 70)
    print("COUNTERFACTUAL ANALYSIS: Feature Change Statistics")
    print("=" * 70)
    
    for dataset_name, config in datasets.items():
        try:
            results = analyze_counterfactuals(dataset_name, config)
            create_summary_csv(dataset_name, config, results)
        except FileNotFoundError as e:
            print(f"  Skipping {dataset_name}: File not found ({str(e)})")
        except Exception as e:
            print(f"  ERROR processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
