"""
Compare narratives for adversely predicted instances across protected attributes.

Edit the USER CONFIGURATION section below to change dataset, number of narratives, etc.
Then press the play button to run.
"""

import json
import sys
import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add parent directory to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import narrative generation function
from scripts.make_narratives import generate_narrative, get_available_instances, save_result

# Import AFINN for sentiment word lexicon
try:
    from afinn import Afinn
    AFINN_AVAILABLE = True
except ImportError:
    AFINN_AVAILABLE = False
    print("Warning: afinn not installed. Install with: pip install afinn")

# Import TextBlob for polarity and subjectivity analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: textblob not installed. Install with: pip install textblob")


# ============================================================================
# USER CONFIGURATION - CHANGE THESE VALUES
# ============================================================================

# Which dataset to analyze
DATASET = "law"  # Options: "law", "credit", "saudi", "student"

# Number of adversely predicted instances to analyze
NUM_INSTANCES = 80

# Type of explanation narratives
PROMPT_TYPE = "shap"  # Options: "shap" or "cf" 

# Generate missing narratives? (Set to True to create narratives)
GENERATE_NARRATIVES = True  # Set to True to generate missing narratives

# Save results to JSON file?
SAVE_RESULTS = True  # Set to True to save comparison results

# LLM provider for generating narratives
LLM_PROVIDER = "openai"  # Options: "openai", "anthropic", "ollama"
LLM_MODEL = None  # Leave as None for default model


# ============================================================================
# CONFIGURATION (Framework/Dataset definitions)
# ============================================================================

# Protected attributes for each dataset
PROTECTED_ATTRIBUTES = {
    "law": ["gender", "race1"],
    "credit": ["personal_status_sex", "age"],
    "saudi": ["Gender", "Age"],
    "student": ["sex", "age"],
}

# Mappings for protected attribute values to readable names
ATTRIBUTE_VALUE_MAPPINGS = {
    "law": {
        "gender": {
            0: "Female",
            1: "Male",
        },
        "race1": {
            0: "White",
            1: "Black",
            2: "Hispanic",
            3: "Asian",
            4: "Native American",
            5: "Puerto Rican",
            6: "Other",
        },
    },
    "credit": {
        "personal_status_sex": {
            "A91": "Male",
            "A92": "Female",
            "A93": "Male",
            "A94": "Female",
        },
        "age": {
            # Age is typically numeric, so we bin it
        },
    },
    "saudi": {
        "Gender": {
            "Male": "Male",
            "Female": "Female",
            "M": "Male",
            "F": "Female",
        },
        "Age": {
            # Age is typically numeric
        },
    },
    "student": {
        "sex": {
            "M": "Male",
            "F": "Female",
            0: "Female",
            1: "Male",
        },
        "age": {
            # Age is typically numeric
        },
    },
}

# Dataset adverse CSV paths
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
    """
    Map a protected attribute value to a readable name.
    
    Args:
        value: The raw value from the dataset
        dataset: Dataset name
        attribute: Attribute name
    
    Returns:
        Mapped value name, or original value if no mapping exists
    """
    if dataset in ATTRIBUTE_VALUE_MAPPINGS:
        if attribute in ATTRIBUTE_VALUE_MAPPINGS[dataset]:
            mapping = ATTRIBUTE_VALUE_MAPPINGS[dataset][attribute]
            if value in mapping:
                return mapping[value]
    
    # If no mapping found, return string representation of value
    return str(value)

# DATA LOADING

def load_adverse_instances(dataset, num_instances=None):
    """
    Load adversely predicted instances from dataset.
    
    Args:
        dataset: Dataset name ("law", "credit", "saudi", "student")
        num_instances: Number of instances to load (None = all)
    
    Returns:
        DataFrame with adverse instances
    """
    csv_path = ADVERSE_CSV_PATHS[dataset]
    df = pd.read_csv(csv_path)
    
    if num_instances is not None:
        df = df.head(num_instances)
    
    return df


def get_protected_attributes_values(adverse_df, dataset):
    """
    Extract protected attribute values for each instance.
    
    Args:
        adverse_df: DataFrame with adverse instances
        dataset: Dataset name
    
    Returns:
        Dictionary mapping instance_index to protected attribute values
    """
    protected_attrs = PROTECTED_ATTRIBUTES[dataset]
    attr_values = {}
    
    for _, row in adverse_df.iterrows():
        instance_idx = int(row["instance_index"])
        attrs = {attr: row[attr] for attr in protected_attrs}
        attr_values[instance_idx] = attrs
    
    return attr_values

# NARRATIVE LOADING
def load_or_generate_narratives(dataset, instance_indices, prompt_type, 
                                provider="openai", model=None, generate_if_missing=False):
    """
    Load narratives for instances. Optionally generate if missing.
    
    Args:
        dataset: Dataset name
        instance_indices: List of instance indices
        prompt_type: "shap" or "cf"
        provider: LLM provider ("openai", "anthropic", "ollama")
        model: Specific model name
        generate_if_missing: If True, generate missing narratives; if False, skip
    
    Returns:
        Dictionary mapping instance_idx to narrative text
    """
    narratives = {}
    missing_instances = []
    
    for instance_idx in instance_indices:
        filepath = Path(f"results/narratives/{dataset}/narratives/{prompt_type}/{provider}/instance_{instance_idx}.json")
        
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("status") == "success":
                    narratives[instance_idx] = data.get("narrative", "")
                else:
                    missing_instances.append(instance_idx)
        else:
            missing_instances.append(instance_idx)
    
    if missing_instances and generate_if_missing:
        print(f"Generating narratives for {len(missing_instances)} missing instances...")
        for idx, instance_idx in enumerate(missing_instances):
            try:
                result = generate_narrative(dataset, instance_idx, prompt_type, provider, model)
                if result.get("status") == "success":
                    narratives[instance_idx] = result.get("narrative", "")
                    # Save the result as JSON file
                    save_result(result, "results/narratives")
                    if (idx + 1) % 10 == 0:
                        print(f"  Generated and saved {idx + 1}/{len(missing_instances)} narratives")
            except Exception as e:
                print(f"  Error generating narrative for instance {instance_idx}: {e}")
    
    return narratives

# NARRATIVE ORGANIZATION
def group_narratives_by_protected_attributes(narratives, adverse_df, dataset):
    """
    Organize narratives by protected attribute values.
    
    Args:
        narratives: Dict mapping instance_idx -> narrative text
        adverse_df: DataFrame with adverse instances
        dataset: Dataset name
    
    Returns:
        Nested dict: {attr_name: {attr_value: {instance_idx: narrative}}}
    """
    protected_attrs = PROTECTED_ATTRIBUTES[dataset]
    grouped = {}
    
    for attr in protected_attrs:
        grouped[attr] = defaultdict(dict)
    
    for _, row in adverse_df.iterrows():
        instance_idx = int(row["instance_index"])
        
        if instance_idx not in narratives:
            continue
        
        narrative = narratives[instance_idx]
        
        for attr in protected_attrs:
            attr_value = row[attr]
            grouped[attr][attr_value][instance_idx] = narrative
    
    return grouped


def get_narrative_statistics(narratives):
    """
    Compute basic statistics on narratives.
    
    Args:
        narratives: Dictionary mapping instance_idx -> narrative text
    
    Returns:
        Dictionary with statistics
    """
    lengths = [len(text.split()) for text in narratives.values()]
    
    return {
        "count": len(narratives),
        "avg_words": np.mean(lengths) if lengths else 0,
        "avg_chars": np.mean([len(text) for text in narratives.values()]) if narratives else 0,
        "min_words": min(lengths) if lengths else 0,
        "max_words": max(lengths) if lengths else 0,
    }


# ============================================================================
# ANALYSIS FRAMEWORK (Placeholder functions for user to implement)
# ============================================================================

# ============================================================================
# ANALYSIS FRAMEWORK
# ============================================================================

class NarrativeAnalyzer:
    """Framework for analyzing narratives - explicit mentions and sentiment words."""
    
    def __init__(self, narratives, protected_groups):
        """
        Initialize analyzer.
        
        Args:
            narratives: Dict mapping instance_idx -> narrative text
            protected_groups: Nested dict of narratives grouped by protected attributes
        """
        self.narratives = narratives
        self.protected_groups = protected_groups
        self.results = {}
    
    def analyze_protected_attribute_mentions(self, narratives=None):
        """
        Count explicit mentions of race and gender in narratives.
        
        Uses word boundary matching (\\b) to avoid false positives from substring matches.
        E.g., "man" won't match "management" or "performance".
        
        Race mentions: Black, Hispanic, Asian, White, Native American, etc.
        Gender mentions: Male, Female, Man, Woman, etc.
        
        Args:
            narratives: Optional dict of narratives to analyze
        
        Returns:
            Dictionary mapping instance_idx -> {"race_mentions": int, "gender_mentions": int}
        """
        if narratives is None:
            narratives = self.narratives
        
        if not narratives:
            return {}
        
        # Define search terms for race and gender (will use word boundaries)
        race_terms = ["Black", "Hispanic", "Asian", "White", "Native American", 
                     "African", "Latino", "Caucasian", "Asians", "Africans"]
        gender_terms = ["Male", "Female", "Man", "Woman", "men", "women", "boy", "girl",
                       "masculine", "feminine"]
        
        # Compile regex patterns with word boundaries (case-insensitive)
        race_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in race_terms]
        gender_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in gender_terms]
        
        results = {}
        total = len(narratives)
        
        for idx, (instance_idx, narrative) in enumerate(narratives.items()):
            if (idx + 1) % 10 == 0:
                print(f"  Processing narrative {idx + 1}/{total}...")
            
            try:
                # Count race mentions using word boundaries
                race_count = sum(len(pattern.findall(narrative)) for pattern in race_patterns)
                
                # Count gender mentions using word boundaries
                gender_count = sum(len(pattern.findall(narrative)) for pattern in gender_patterns)
                
                results[instance_idx] = {
                    "race_mentions": race_count,
                    "gender_mentions": gender_count,
                }
            
            except Exception as e:
                print(f"  Error processing instance {instance_idx}: {e}")
                results[instance_idx] = {"race_mentions": 0, "gender_mentions": 0}
        
        return results
    
    def analyze_sentiment_words(self, narratives=None):
        """
        Count negative and positive words in narratives using AFINN lexicon.
        
        AFINN is a simple lexicon of ~3,500 English words rated for sentiment (-5 to +5).
        
        Args:
            narratives: Optional dict of narratives to analyze
        
        Returns:
            Dictionary mapping instance_idx -> {"negative_words": int, "positive_words": int}
        """
        if narratives is None:
            narratives = self.narratives
        
        if not narratives:
            return {}
        
        if not AFINN_AVAILABLE:
            print("Error: afinn library not available. Install with: pip install afinn")
            return {idx: {"negative_words": 0, "positive_words": 0} for idx in narratives.keys()}
        
        # Initialize AFINN lexicon
        print("Loading AFINN sentiment lexicon...")
        afinn = Afinn()
        print("AFINN lexicon loaded.")
        
        results = {}
        total = len(narratives)
        
        for idx, (instance_idx, narrative) in enumerate(narratives.items()):
            if (idx + 1) % 10 == 0:
                print(f"  Processing narrative {idx + 1}/{total}...")
            
            try:
                # Split narrative into words and check AFINN scores
                words = narrative.lower().split()
                
                negative_count = 0
                positive_count = 0
                
                # Count words in AFINN lexicon
                for word in words:
                    # Remove punctuation for better matching
                    word_clean = ''.join(c for c in word if c.isalnum())
                    score = afinn.score(word_clean)
                    
                    if score < 0:
                        negative_count += 1
                    elif score > 0:
                        positive_count += 1
                
                results[instance_idx] = {
                    "negative_words": negative_count,
                    "positive_words": positive_count,
                }
            
            except Exception as e:
                print(f"  Error processing instance {instance_idx}: {e}")
                results[instance_idx] = {"negative_words": 0, "positive_words": 0}
        
        return results
    
    def analyze_polarity_subjectivity(self, narratives=None):
        """
        Analyze polarity and subjectivity of narratives using TextBlob.
        
        Polarity: -1 (most negative) to 1 (most positive)
        Subjectivity: 0 (objective) to 1 (subjective)
        
        Args:
            narratives: Optional dict of narratives to analyze
        
        Returns:
            Dictionary mapping instance_idx -> {"polarity": float, "subjectivity": float}
        """
        if narratives is None:
            narratives = self.narratives
        
        if not narratives:
            return {}
        
        if not TEXTBLOB_AVAILABLE:
            print("Error: textblob library not available. Install with: pip install textblob")
            return {idx: {"polarity": 0.0, "subjectivity": 0.0} for idx in narratives.keys()}
        
        results = {}
        total = len(narratives)
        
        for idx, (instance_idx, narrative) in enumerate(narratives.items()):
            if (idx + 1) % 10 == 0:
                print(f"  Processing narrative {idx + 1}/{total}...")
            
            try:
                # Create TextBlob and extract sentiment
                blob = TextBlob(narrative)
                
                results[instance_idx] = {
                    "polarity": float(blob.sentiment.polarity),      # -1 to 1
                    "subjectivity": float(blob.sentiment.subjectivity),  # 0 to 1
                }
            
            except Exception as e:
                print(f"  Error processing instance {instance_idx}: {e}")
                results[instance_idx] = {"polarity": 0.0, "subjectivity": 0.0}
        
        return results
    
    def run_analysis(self):
        """
        Run all narrative analyses on the data.
        
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "="*70)
        print("Running narrative analyses...")
        print("="*70)
        
        # Protected Attribute Mentions
        print("\n1. Protected Attribute Mentions (Race and Gender)...")
        try:
            self.results["protected_mentions"] = self.analyze_protected_attribute_mentions()
        except Exception as e:
            print(f"Error: {e}")
            raise
        
        # Sentiment Words
        print("\n2. Negative and Positive Words...")
        try:
            self.results["sentiment_words"] = self.analyze_sentiment_words()
        except Exception as e:
            print(f"Error: {e}")
            self.results["sentiment_words"] = {}
        
        # Polarity and Subjectivity
        print("\n3. Polarity and Subjectivity (TextBlob)...")
        try:
            self.results["polarity_subjectivity"] = self.analyze_polarity_subjectivity()
        except Exception as e:
            print(f"Error: {e}")
            self.results["polarity_subjectivity"] = {}
        
        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70)
        
        return self.results


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(analyzer, adverse_df, dataset, prompt_type, num_instances):
    """
    Generate a summary report comparing narratives across protected attributes.
    
    Args:
        analyzer: NarrativeAnalyzer instance with completed analyses
        adverse_df: DataFrame with adverse instances and protected attributes
        dataset: Dataset name
        prompt_type: "shap" or "cf"
        num_instances: Number of instances analyzed
    """
    print("\n" + "="*70)
    print("NARRATIVE COMPARISON REPORT")
    print("="*70)
    print(f"Dataset: {dataset}")
    print(f"Prompt Type: {prompt_type.upper()}")
    print(f"Instances Analyzed: {num_instances}")
    print(f"Protected Attributes: {', '.join(PROTECTED_ATTRIBUTES[dataset])}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    # Print protected attribute mentions by protected class
    if "protected_mentions" in analyzer.results:
        print("\nPROTECTED ATTRIBUTE MENTIONS (Race and Gender):")
        protected_mentions = analyzer.results["protected_mentions"]
        
        protected_attrs = PROTECTED_ATTRIBUTES[dataset]
        for attr in protected_attrs:
            print(f"\n  {attr.upper()}:")
            unique_values = adverse_df[attr].unique()
            
            for value in sorted(unique_values):
                matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
                
                # Get mention counts for instances with this attribute value
                race_mentions = [protected_mentions.get(idx, {}).get("race_mentions", 0) for idx in matching_instances if idx in protected_mentions]
                gender_mentions = [protected_mentions.get(idx, {}).get("gender_mentions", 0) for idx in matching_instances if idx in protected_mentions]
                
                if race_mentions and gender_mentions:
                    mapped_value = map_attribute_value(value, dataset, attr)
                    total_narratives = len(race_mentions)
                    total_race_mentions = sum(race_mentions)
                    total_gender_mentions = sum(gender_mentions)
                    avg_race_mentions = np.mean(race_mentions)
                    avg_gender_mentions = np.mean(gender_mentions)
                    
                    print(f"    {mapped_value}:")
                    print(f"      Total Narratives: {total_narratives}")
                    print(f"      Total Race Mentions: {total_race_mentions} (Avg: {avg_race_mentions:.2f} per narrative)")
                    print(f"      Total Gender Mentions: {total_gender_mentions} (Avg: {avg_gender_mentions:.2f} per narrative)")
    
    # Print sentiment words by protected class
    if "sentiment_words" in analyzer.results and analyzer.results["sentiment_words"]:
        print("\nSENTIMENT WORDS (Negative and Positive):")
        sentiment_words = analyzer.results["sentiment_words"]
        
        protected_attrs = PROTECTED_ATTRIBUTES[dataset]
        for attr in protected_attrs:
            print(f"\n  {attr.upper()}:")
            unique_values = adverse_df[attr].unique()
            
            for value in sorted(unique_values):
                matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
                
                # Get sentiment word counts for instances with this attribute value
                negative_words = [sentiment_words.get(idx, {}).get("negative_words", 0) for idx in matching_instances if idx in sentiment_words]
                positive_words = [sentiment_words.get(idx, {}).get("positive_words", 0) for idx in matching_instances if idx in sentiment_words]
                
                if negative_words and positive_words:
                    mapped_value = map_attribute_value(value, dataset, attr)
                    print(f"    {mapped_value}:")
                    print(f"      Count: {len(negative_words)}")
                    print(f"      Avg Negative Words: {np.mean(negative_words):.2f} (±{np.std(negative_words):.2f})")
                    print(f"      Avg Positive Words: {np.mean(positive_words):.2f} (±{np.std(positive_words):.2f})")
    
    # Print polarity and subjectivity by protected class
    if "polarity_subjectivity" in analyzer.results and analyzer.results["polarity_subjectivity"]:
        print("\nPOLARITY AND SUBJECTIVITY (TextBlob):")
        print("  Polarity: -1 (negative) to 1 (positive)")
        print("  Subjectivity: 0 (objective) to 1 (subjective)")
        
        polarity_subjectivity = analyzer.results["polarity_subjectivity"]
        
        protected_attrs = PROTECTED_ATTRIBUTES[dataset]
        for attr in protected_attrs:
            print(f"\n  {attr.upper()}:")
            unique_values = adverse_df[attr].unique()
            
            for value in sorted(unique_values):
                matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
                
                # Get polarity and subjectivity for instances with this attribute value
                polarities = [polarity_subjectivity.get(idx, {}).get("polarity", 0.0) for idx in matching_instances if idx in polarity_subjectivity]
                subjectivities = [polarity_subjectivity.get(idx, {}).get("subjectivity", 0.0) for idx in matching_instances if idx in polarity_subjectivity]
                
                if polarities and subjectivities:
                    mapped_value = map_attribute_value(value, dataset, attr)
                    print(f"    {mapped_value}:")
                    print(f"      Count: {len(polarities)}")
                    print(f"      Avg Polarity: {np.mean(polarities):.3f} (±{np.std(polarities):.3f})")
                    print(f"      Avg Subjectivity: {np.mean(subjectivities):.3f} (±{np.std(subjectivities):.3f})")
    
    print("\n" + "="*70)


def save_results(analyzer, dataset, prompt_type, output_dir="results/comparisons"):
    """
    Save analysis results to JSON file.
    
    Args:
        analyzer: NarrativeAnalyzer instance
        dataset: Dataset name
        prompt_type: "shap" or "cf"
        output_dir: Directory to save results
    """
    output_path = Path(output_dir) / f"{dataset}_{prompt_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert results to JSON-serializable format
    results_json = {
        "dataset": dataset,
        "prompt_type": prompt_type,
        "timestamp": datetime.now().isoformat(),
        "num_instances": len(analyzer.narratives),
        "results": {}
    }
    
    # Convert numpy types to native Python types
    for key, value in analyzer.results.items():
        if isinstance(value, dict):
            results_json["results"][key] = {
                str(k): (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                for k, v in value.items()
            }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Get configuration from user settings at top of file
    dataset = DATASET
    num_instances = NUM_INSTANCES
    prompt_type = PROMPT_TYPE
    generate_flag = GENERATE_NARRATIVES
    save_flag = SAVE_RESULTS
    provider = LLM_PROVIDER
    model = LLM_MODEL
    
    # Validate inputs
    if dataset not in ["law", "credit", "saudi", "student"]:
        print(f"Error: Invalid dataset '{dataset}'. Choose from: law, credit, saudi, student")
        return
    
    if prompt_type not in ["shap", "cf"]:
        print(f"Error: Invalid prompt_type '{prompt_type}'. Choose from: shap, cf")
        return
    
    # Load adverse instances
    print(f"\nLoading adversely predicted instances from {dataset} dataset...")
    adverse_df = load_adverse_instances(dataset, num_instances)
    print(f"  Loaded {len(adverse_df)} instances")
    
    # Determine which instances to analyze
    instance_indices = adverse_df["instance_index"].astype(int).tolist()
    print(f"  Analyzing {len(instance_indices)} instances")
    
    # Load or generate narratives
    print(f"\nLoading narratives (type: {prompt_type}, provider: {provider})...")
    narratives = load_or_generate_narratives(
        dataset, instance_indices, prompt_type,
        provider, model, generate_if_missing=generate_flag
    )
    print(f"  Loaded {len(narratives)} narratives")
    
    if len(narratives) == 0:
        print("\nNo narratives found. Set GENERATE_NARRATIVES = True to create them.")
        return
    
    # Group narratives by protected attributes
    print(f"\nGrouping narratives by protected attributes...")
    protected_groups = group_narratives_by_protected_attributes(narratives, adverse_df, dataset)
    
    for attr, groups in protected_groups.items():
        print(f"  {attr}: {len(groups)} groups")
        for group_val, instances in groups.items():
            print(f"    {group_val}: {len(instances)} instances")
    
    # Initialize analyzer
    analyzer = NarrativeAnalyzer(narratives, protected_groups)
    
    # Run analysis
    analyzer.run_analysis()
    
    # Generate report
    generate_report(analyzer, adverse_df, dataset, prompt_type, len(narratives))
    
    # Save results if requested
    if save_flag:
        save_results(analyzer, dataset, prompt_type)


if __name__ == "__main__":
    main()
