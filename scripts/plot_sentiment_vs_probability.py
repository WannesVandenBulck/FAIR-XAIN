"""
Create visualizations comparing sentiment scores with predicted failure probability.
Two plots: Gender comparison and Race comparison.
"""

import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import TextBlob for sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Error: textblob not installed. Install with: pip install textblob")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET = "law"  # Options: "law", "credit", "saudi", "student"
NUM_INSTANCES = 80
PROMPT_TYPE = "shap"  # Options: "shap" or "cf"

# Protected attributes for each dataset
PROTECTED_ATTRIBUTES = {
    "law": ["gender", "race1"],
    "credit": ["personal_status_sex", "age"],
    "saudi": ["Gender", "Age"],
    "student": ["sex", "age"],
}

# Mappings for protected attribute values
ATTRIBUTE_VALUE_MAPPINGS = {
    "law": {
        "gender": {0: "Female", 1: "Male"},
        "race1": {0: "White", 1: "Black", 2: "Hispanic", 3: "Asian", 4: "Native American", 5: "Puerto Rican", 6: "Other"},
    },
    "credit": {
        "personal_status_sex": {
            "A91": "Male", "A92": "Female", "A93": "Male", "A94": "Female",
            "A95": "Female", "A96": "Male", "A97": "Female", "A98": "Male",
        },
        "age": lambda x: f"Age {int(x)}" if isinstance(x, (int, float)) else str(x),
    },
    "saudi": {
        "Gender": {0: "Female", 1: "Male"},
        "Age": lambda x: f"Age {int(x)}" if isinstance(x, (int, float)) else str(x),
    },
    "student": {
        "sex": {0: "Female", 1: "Male"},
        "age": lambda x: f"Age {int(x)}" if isinstance(x, (int, float)) else str(x),
    },
}

# CSV paths
ADVERSE_CSV_PATHS = {
    "law": "datasets_prep/data/law_dataset/law_adverse.csv",
    "credit": "datasets_prep/data/credit_dataset/credit_adverse.csv",
    "saudi": "datasets_prep/data/saudi_dataset/saudi_adverse.csv",
    "student": "datasets_prep/data/student_dataset/student_adverse.csv",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def map_attribute_value(value, dataset, attribute):
    """Map attribute value to readable name."""
    if dataset not in ATTRIBUTE_VALUE_MAPPINGS:
        return str(value)
    
    attr_map = ATTRIBUTE_VALUE_MAPPINGS[dataset].get(attribute, {})
    if callable(attr_map):
        return attr_map(value)
    elif isinstance(attr_map, dict):
        return attr_map.get(value, str(value))
    return str(value)


def load_adverse_instances(dataset, num_instances=None):
    """Load adversely predicted instances."""
    csv_path = ADVERSE_CSV_PATHS[dataset]
    df = pd.read_csv(csv_path)
    
    if num_instances is not None:
        df = df.head(num_instances)
    
    return df


def load_narratives_from_results(dataset, instance_indices, prompt_type, provider="openai"):
    """Load narratives from results directory."""
    narratives = {}
    
    results_dir = Path(f"results/narratives/{dataset}/narratives/{prompt_type}/{provider}")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return narratives
    
    for instance_idx in instance_indices:
        narrative_file = results_dir / f"instance_{instance_idx}.json"
        
        if narrative_file.exists():
            try:
                with open(narrative_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'narrative' in data:
                        narratives[instance_idx] = data['narrative']
            except Exception as e:
                print(f"Error loading narrative for instance {instance_idx}: {e}")
    
    return narratives


def calculate_sentiment_scores(narratives):
    """Calculate sentiment scores using TextBlob for all narratives."""
    sentiment_scores = {}
    total = len(narratives)
    
    print(f"Calculating sentiment scores for {total} narratives...")
    
    for idx, (instance_idx, narrative) in enumerate(narratives.items()):
        if (idx + 1) % 10 == 0:
            print(f"  Processing narrative {idx + 1}/{total}...")
        
        try:
            blob = TextBlob(narrative)
            polarity = float(blob.sentiment.polarity)
            sentiment_scores[instance_idx] = polarity
        except Exception as e:
            print(f"  Error processing instance {instance_idx}: {e}")
            sentiment_scores[instance_idx] = 0.0
    
    return sentiment_scores


def prepare_plot_data(adverse_df, narratives, sentiment_scores, dataset, attribute):
    """
    Prepare data for plotting.
    
    Returns:
        Dictionary with {attribute_value: {"x": [...], "y": [...], "label": "..."}}
    """
    data_by_group = {}
    
    # Get unique values for this attribute
    unique_values = adverse_df[attribute].unique()
    
    for value in sorted(unique_values):
        # Get instances matching this attribute value
        matching_rows = adverse_df[adverse_df[attribute] == value]
        instance_indices = matching_rows['instance_index'].astype(int).tolist()
        
        # Get prediction scores and sentiment scores
        x_values = []  # Predicted probabilities
        y_values = []  # Sentiment scores
        
        for instance_idx in instance_indices:
            if instance_idx in narratives and instance_idx in sentiment_scores:
                pred_score = matching_rows[matching_rows['instance_index'] == instance_idx]['prediction_score'].values[0]
                sentiment = sentiment_scores[instance_idx]
                
                x_values.append(pred_score)
                y_values.append(sentiment)
        
        if x_values:  # Only add if we have data
            mapped_value = map_attribute_value(value, dataset, attribute)
            data_by_group[value] = {
                "x": np.array(x_values),
                "y": np.array(y_values),
                "label": str(mapped_value),
            }
    
    return data_by_group


def add_regression_line(ax, x, y, color, label_suffix=""):
    """Add a linear regression line to the plot."""
    if len(x) < 2:
        return
    
    # Calculate linear regression
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    # Generate points for the line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = p(x_line)
    
    # Calculate R-squared
    y_pred = p(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Plot the line
    ax.plot(x_line, y_line, color=color, linewidth=2.5, linestyle='--',
            label=f'Trend (R² = {r_squared:.3f}){label_suffix}')


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_gender_comparison(adverse_df, narratives, sentiment_scores, dataset):
    """Create scatter plot comparing male vs female narrative sentiments."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    attribute = "gender"
    data_by_group = prepare_plot_data(adverse_df, narratives, sentiment_scores, dataset, attribute)
    
    # Color mapping for gender
    color_map = {
        "Female": "#FF6B6B",  # Red
        "Male": "#4ECDC4",    # Blue/Teal
    }
    
    # Plot each gender group
    for value, data in data_by_group.items():
        label = data["label"]
        color = color_map.get(label, "#999999")
        
        ax.scatter(data["x"], data["y"], alpha=0.6, s=100, color=color, 
                  label=f"{label} (n={len(data['x'])})", edgecolors='black', linewidth=0.5)
        
        # Add regression line
        add_regression_line(ax, data["x"], data["y"], color, f" ({label})")
    
    # Combine all data for overall trend
    all_x = np.concatenate([data["x"] for data in data_by_group.values()])
    all_y = np.concatenate([data["y"] for data in data_by_group.values()])
    
    # Plot overall trend line
    if len(all_x) > 1:
        z = np.polyfit(all_x, all_y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_x.min(), all_x.max(), 100)
        y_line = p(x_line)
        ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5, label='Overall Trend')
    
    # Formatting
    ax.set_xlabel('Predicted Probability of Failure', fontsize=13, fontweight='bold')
    ax.set_ylabel('Narrative Sentiment Score (TextBlob)', fontsize=13, fontweight='bold')
    ax.set_title(f'Narrative Sentiment vs. Predicted Failure Probability by Gender\n{dataset.upper()} Dataset (n={len(adverse_df)} instances)', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.10, 0.10)
    ax.set_xlim(0.4, 1.0)
    
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    
    plt.tight_layout()
    output_path = Path(f"results/plots/sentiment_vs_probability_gender_{dataset}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_race_comparison(adverse_df, narratives, sentiment_scores, dataset):
    """Create scatter plot comparing different races' narrative sentiments."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    attribute = "race1" if dataset == "law" else None
    if not attribute or attribute not in adverse_df.columns:
        print(f"Race attribute not available for {dataset} dataset")
        return
    
    data_by_group = prepare_plot_data(adverse_df, narratives, sentiment_scores, dataset, attribute)
    
    # Color palette for races
    color_palette = {
        "White": "#1f77b4",          # Blue
        "Black": "#ff7f0e",          # Orange
        "Hispanic": "#2ca02c",       # Green
        "Asian": "#d62728",          # Red
        "Native American": "#9467bd", # Purple
        "Puerto Rican": "#8c564b",   # Brown
        "Other": "#7f7f7f",          # Gray
    }
    
    # Plot each race group
    for value, data in data_by_group.items():
        label = data["label"]
        color = color_palette.get(label, "#999999")
        
        ax.scatter(data["x"], data["y"], alpha=0.6, s=100, color=color,
                  label=f"{label} (n={len(data['x'])})", edgecolors='black', linewidth=0.5)
        
        # Add regression line
        add_regression_line(ax, data["x"], data["y"], color, f" ({label})")
    
    # Combine all data for overall trend
    all_x = np.concatenate([data["x"] for data in data_by_group.values()])
    all_y = np.concatenate([data["y"] for data in data_by_group.values()])
    
    # Plot overall trend line
    if len(all_x) > 1:
        z = np.polyfit(all_x, all_y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_x.min(), all_x.max(), 100)
        y_line = p(x_line)
        ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5, label='Overall Trend')
    
    # Formatting
    ax.set_xlabel('Predicted Probability of Failure', fontsize=13, fontweight='bold')
    ax.set_ylabel('Narrative Sentiment Score (TextBlob)', fontsize=13, fontweight='bold')
    ax.set_title(f'Narrative Sentiment vs. Predicted Failure Probability by Race\n{dataset.upper()} Dataset (n={len(adverse_df)} instances)',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.10, 0.10)
    ax.set_xlim(0.4, 1.0)
    
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    
    plt.tight_layout()
    output_path = Path(f"results/plots/sentiment_vs_probability_race_{dataset}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("SENTIMENT VS. PREDICTED PROBABILITY VISUALIZATION")
    print("=" * 70)
    
    # Load adverse instances
    print(f"\nLoading adverse instances from {DATASET} dataset...")
    adverse_df = load_adverse_instances(DATASET, NUM_INSTANCES)
    print(f"  Loaded {len(adverse_df)} instances")
    
    instance_indices = adverse_df['instance_index'].astype(int).tolist()
    
    # Load narratives
    print(f"\nLoading narratives (type: {PROMPT_TYPE})...")
    narratives = load_narratives_from_results(DATASET, instance_indices, PROMPT_TYPE)
    print(f"  Loaded {len(narratives)} narratives")
    
    if len(narratives) == 0:
        print("ERROR: No narratives found. Generate narratives first using make_narratives.py")
        return
    
    # Calculate sentiment scores
    sentiment_scores = calculate_sentiment_scores(narratives)
    
    # Create plots
    print("\n" + "=" * 70)
    print("Creating visualizations...")
    print("=" * 70)
    
    print("\n1. Gender Comparison Plot...")
    plot_gender_comparison(adverse_df, narratives, sentiment_scores, DATASET)
    
    print("\n2. Race Comparison Plot...")
    plot_race_comparison(adverse_df, narratives, sentiment_scores, DATASET)
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
