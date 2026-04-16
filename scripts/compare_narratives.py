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

# Import HuggingFace transformers for sentiment analysis
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")


# ============================================================================
# USER CONFIGURATION - CHANGE THESE VALUES
# ============================================================================

# Which dataset to analyze
DATASET = "law"  # Options: "law", "credit", "saudi", "student"

# Number of adversely predicted instances to analyze
NUM_INSTANCES = None  # Set to None to analyze all available instances

# Type of explanation narratives
PROMPT_TYPE = "narrative"  # Options: "shap", "narrative", or "cf" 

# Generate missing narratives? (Set to True to create narratives)
GENERATE_NARRATIVES = True  # Set to True to generate missing narratives

# Save results to JSON file?
SAVE_RESULTS = True  # Set to True to save comparison results

# LLM provider for generating narratives
LLM_PROVIDER = "grok"  # Options: "openai", "anthropic", "grok", "ollama"
LLM_MODEL = "grok-4-1-fast-non-reasoning"  # Leave as None for default model


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
            0: "Female",
            1: "Male",
        },
        "Age": {
            0: "21-30",
            1: "31-40",
            2: "41+",
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
        provider: LLM provider ("openai", "anthropic", "grok", "ollama")
        model: Specific model name
        generate_if_missing: If True, generate missing narratives; if False, skip
    
    Returns:
        Dictionary mapping instance_idx to narrative text
    """
    narratives = {}
    missing_instances = []
    
    for instance_idx in instance_indices:
        filepath = Path(f"results/narratives/{dataset}/narratives/{prompt_type}/{provider}/{model}/instance_{instance_idx}.json")
        
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
        Count explicit mentions of protected attributes in narratives.
        
        For datasets with gender/race: counts mentions of gender (Male, Female, etc.) and race terms
        For datasets with gender/age: counts only gender mentions (since age is typically numeric)
        
        Uses word boundary matching (\\b) to avoid false positives from substring matches.
        E.g., "man" won't match "management" or "performance".
        
        Args:
            narratives: Optional dict of narratives to analyze
        
        Returns:
            Dictionary mapping instance_idx -> {"gender_mentions": int, "race_mentions": int}
        """
        if narratives is None:
            narratives = self.narratives
        
        if not narratives:
            return {}
        
        # Define search terms for gender and race (will use word boundaries)
        gender_terms = ["Male", "Female", "Man", "Woman", "men", "women", "boy", "girl",
                       "masculine", "feminine"]
        race_terms = ["Black", "Hispanic", "Asian", "White", "Native American", 
                     "African", "Latino", "Caucasian", "Asians", "Africans"]
        
        # Compile regex patterns with word boundaries (case-insensitive)
        gender_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in gender_terms]
        race_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in race_terms]
        
        results = {}
        total = len(narratives)
        
        for idx, (instance_idx, narrative) in enumerate(narratives.items()):
            if (idx + 1) % 10 == 0:
                print(f"  Processing narrative {idx + 1}/{total}...")
            
            try:
                # Count gender mentions using word boundaries
                gender_count = sum(len(pattern.findall(narrative)) for pattern in gender_patterns)
                
                # Count race mentions using word boundaries
                race_count = sum(len(pattern.findall(narrative)) for pattern in race_patterns)
                
                results[instance_idx] = {
                    "gender_mentions": gender_count,
                    "race_mentions": race_count,
                }
            
            except Exception as e:
                print(f"  Error processing instance {instance_idx}: {e}")
                results[instance_idx] = {"gender_mentions": 0, "race_mentions": 0}
        
        return results
    
    def analyze_emotion_distilroberta(self, narratives=None):
        """
        Analyze emotions using DistilRoBERTa (j-hartmann/emotion-english-distilroberta-base).
        
        Detects: anger, disgust, fear, joy, neutral, sadness, surprise
        
        Args:
            narratives: Optional dict of narratives to analyze
        
        Returns:
            Dictionary mapping instance_idx -> {"top_emotion": str, "scores": dict}
        """
        if narratives is None:
            narratives = self.narratives
        
        if not narratives or not TRANSFORMERS_AVAILABLE:
            return {}
        
        print("Loading DistilRoBERTa emotion model (j-hartmann/emotion-english-distilroberta-base)...")
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        
        results = {}
        total = len(narratives)
        
        for idx, (instance_idx, narrative) in enumerate(narratives.items()):
            if (idx + 1) % 10 == 0:
                print(f"  Processing narrative {idx + 1}/{total}...")
            
            try:
                # Truncate narrative to 512 tokens max for transformer
                narrative_short = narrative[:512]
                prediction = classifier(narrative_short)[0]
                
                results[instance_idx] = {
                    "model": "distilroberta",
                    "top_emotion": prediction["label"],
                    "confidence": float(prediction["score"])
                }
            except Exception as e:
                print(f"  Error processing instance {instance_idx}: {e}")
                results[instance_idx] = {"model": "distilroberta", "top_emotion": "neutral", "confidence": 0.0}
        
        return results
    
    def analyze_emotion_go_emotions(self, narratives=None):
        """
        Analyze emotions using Go Emotions (SamLowe/roberta-base-go_emotions).
        
        Detects: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity,
        desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, 
        gratitude, grief, joy, love, nervousness, neutral, optimism, pride, realization, 
        relief, remorse, sadness, surprise, etc.
        
        Args:
            narratives: Optional dict of narratives to analyze
        
        Returns:
            Dictionary mapping instance_idx -> {"top_emotion": str, "scores": dict}
        """
        if narratives is None:
            narratives = self.narratives
        
        if not narratives or not TRANSFORMERS_AVAILABLE:
            return {}
        
        print("Loading Go Emotions model (SamLowe/roberta-base-go_emotions)...")
        classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
        
        results = {}
        total = len(narratives)
        
        for idx, (instance_idx, narrative) in enumerate(narratives.items()):
            if (idx + 1) % 10 == 0:
                print(f"  Processing narrative {idx + 1}/{total}...")
            
            try:
                # Truncate narrative to 512 tokens max for transformer
                narrative_short = narrative[:512]
                predictions = classifier(narrative_short)[0]
                
                # Get top emotion
                top_pred = max(predictions, key=lambda x: x["score"])
                
                results[instance_idx] = {
                    "model": "go_emotions",
                    "top_emotion": top_pred["label"],
                    "confidence": float(top_pred["score"])
                }
            except Exception as e:
                print(f"  Error processing instance {instance_idx}: {e}")
                results[instance_idx] = {"model": "go_emotions", "top_emotion": "neutral", "confidence": 0.0}
        
        return results
    
    def analyze_empathy_strategy(self, narratives=None):
        """
        Analyze empathy and reassurance strategy (RyanDDD/empathy-strategy-classifier).
        
        Detects: empathetic, non-empathetic (and reassurance strategy)
        
        Args:
            narratives: Optional dict of narratives to analyze
        
        Returns:
            Dictionary mapping instance_idx -> {"empathy_label": str, "empathy_score": float}
        """
        if narratives is None:
            narratives = self.narratives
        
        if not narratives or not TRANSFORMERS_AVAILABLE:
            return {}
        
        print("Loading Empathy Strategy model (RyanDDD/empathy-strategy-classifier)...")
        classifier = pipeline("text-classification", model="RyanDDD/empathy-strategy-classifier")
        
        results = {}
        total = len(narratives)
        
        for idx, (instance_idx, narrative) in enumerate(narratives.items()):
            if (idx + 1) % 10 == 0:
                print(f"  Processing narrative {idx + 1}/{total}...")
            
            try:
                # Truncate narrative to 512 tokens max for transformer
                narrative_short = narrative[:512]
                prediction = classifier(narrative_short)[0]
                
                results[instance_idx] = {
                    "model": "empathy_strategy",
                    "empathy_label": prediction["label"],
                    "empathy_score": float(prediction["score"])
                }
            except Exception as e:
                print(f"  Error processing instance {instance_idx}: {e}")
                results[instance_idx] = {"model": "empathy_strategy", "empathy_label": "non-empathetic", "empathy_score": 0.0}
        
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
        
        # Sentiment Analysis - Three HuggingFace Models
        print("\n2. Emotion Analysis (DistilRoBERTa)...")
        try:
            self.results["emotion_distilroberta"] = self.analyze_emotion_distilroberta()
        except Exception as e:
            print(f"Error: {e}")
            self.results["emotion_distilroberta"] = {}
        
        print("\n3. Emotion Analysis (Go Emotions)...")
        try:
            self.results["emotion_go_emotions"] = self.analyze_emotion_go_emotions()
        except Exception as e:
            print(f"Error: {e}")
            self.results["emotion_go_emotions"] = {}
        
        print("\n4. Empathy & Reassurance Analysis...")
        try:
            self.results["empathy_strategy"] = self.analyze_empathy_strategy()
        except Exception as e:
            print(f"Error: {e}")
            self.results["empathy_strategy"] = {}
        
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
    # Print protected attribute mentions by protected class
    if "protected_mentions" in analyzer.results:
        protected_mentions = analyzer.results["protected_mentions"]
        protected_attrs = PROTECTED_ATTRIBUTES[dataset]
        
        # Determine which mention types are relevant for this dataset
        has_race = any(attr.lower() == "race1" or "race" in attr.lower() for attr in protected_attrs)
        has_gender = any(attr.lower() in ["gender", "sex", "personal_status_sex"] for attr in protected_attrs)
        
        if protected_mentions and (has_gender or has_race):
            print("\nPROTECTED ATTRIBUTE MENTIONS:")
            
            for attr in protected_attrs:
                print(f"\n  {attr.upper()}:")
                unique_values = adverse_df[attr].unique()
                
                for value in sorted(unique_values):
                    matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
                    
                    # Get mention counts for instances with this attribute value
                    if has_gender:
                        gender_mentions = [protected_mentions.get(idx, {}).get("gender_mentions", 0) for idx in matching_instances if idx in protected_mentions]
                    else:
                        gender_mentions = []
                    
                    if has_race:
                        race_mentions = [protected_mentions.get(idx, {}).get("race_mentions", 0) for idx in matching_instances if idx in protected_mentions]
                    else:
                        race_mentions = []
                    
                    # Only print if we have data
                    if gender_mentions or race_mentions:
                        mapped_value = map_attribute_value(value, dataset, attr)
                        total_narratives = len(gender_mentions) if gender_mentions else len(race_mentions)
                        
                        print(f"    {mapped_value}:")
                        print(f"      Total Narratives: {total_narratives}")
                        
                        if gender_mentions:
                            total_gender_mentions = sum(gender_mentions)
                            avg_gender_mentions = np.mean(gender_mentions)
                            print(f"      Total Gender Mentions: {total_gender_mentions} (Avg: {avg_gender_mentions:.2f} per narrative)")
                        
                        if race_mentions:
                            total_race_mentions = sum(race_mentions)
                            avg_race_mentions = np.mean(race_mentions)
                            print(f"      Total Race Mentions: {total_race_mentions} (Avg: {avg_race_mentions:.2f} per narrative)")
    
    # Print sentiment analysis results - DistilRoBERTa
    if "emotion_distilroberta" in analyzer.results and analyzer.results["emotion_distilroberta"]:
        print("\nEMOTION ANALYSIS (DistilRoBERTa - j-hartmann):")
        emotion_results = analyzer.results["emotion_distilroberta"]
        
        protected_attrs = PROTECTED_ATTRIBUTES[dataset]
        for attr in protected_attrs:
            print(f"\n  {attr.upper()}:")
            unique_values = adverse_df[attr].unique()
            
            for value in sorted(unique_values):
                matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
                
                emotions = [emotion_results.get(idx, {}).get("top_emotion", "neutral") for idx in matching_instances if idx in emotion_results]
                
                if emotions:
                    emotion_counts = {}
                    for emotion in emotions:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    mapped_value = map_attribute_value(value, dataset, attr)
                    print(f"    {mapped_value} (n={len(emotions)}):")
                    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(emotions)) * 100
                        print(f"      {emotion}: {count} ({percentage:.1f}%)")
    
    # Print sentiment analysis results - Go Emotions
    if "emotion_go_emotions" in analyzer.results and analyzer.results["emotion_go_emotions"]:
        print("\nEMOTION ANALYSIS (Go Emotions - SamLowe):")
        emotion_results = analyzer.results["emotion_go_emotions"]
        
        protected_attrs = PROTECTED_ATTRIBUTES[dataset]
        for attr in protected_attrs:
            print(f"\n  {attr.upper()}:")
            unique_values = adverse_df[attr].unique()
            
            for value in sorted(unique_values):
                matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
                
                emotions = [emotion_results.get(idx, {}).get("top_emotion", "neutral") for idx in matching_instances if idx in emotion_results]
                
                if emotions:
                    emotion_counts = {}
                    for emotion in emotions:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    mapped_value = map_attribute_value(value, dataset, attr)
                    print(f"    {mapped_value} (n={len(emotions)}):")
                    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(emotions)) * 100
                        print(f"      {emotion}: {count} ({percentage:.1f}%)")
    
    # Print empathy and reassurance analysis results
    if "empathy_strategy" in analyzer.results and analyzer.results["empathy_strategy"]:
        print("\nEMPATHY & REASSURANCE ANALYSIS (Empathy Strategy Classifier):")
        empathy_results = analyzer.results["empathy_strategy"]
        
        protected_attrs = PROTECTED_ATTRIBUTES[dataset]
        for attr in protected_attrs:
            print(f"\n  {attr.upper()}:")
            unique_values = adverse_df[attr].unique()
            
            for value in sorted(unique_values):
                matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
                
                empathy_labels = [empathy_results.get(idx, {}).get("empathy_label", "neutral") for idx in matching_instances if idx in empathy_results]
                empathy_scores = [empathy_results.get(idx, {}).get("empathy_score", 0.0) for idx in matching_instances if idx in empathy_results]
                
                if empathy_labels:
                    empathy_counts = {}
                    for label in empathy_labels:
                        empathy_counts[label] = empathy_counts.get(label, 0) + 1
                    
                    mapped_value = map_attribute_value(value, dataset, attr)
                    print(f"    {mapped_value} (n={len(empathy_labels)}):")
                    for label, count in sorted(empathy_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(empathy_labels)) * 100
                        if empathy_scores:
                            avg_score = np.mean([s for i, s in enumerate(empathy_scores) if empathy_labels[i] == label])
                            print(f"      {label}: {count} ({percentage:.1f}%) - Avg Score: {avg_score:.3f}")
                        else:
                            print(f"      {label}: {count} ({percentage:.1f}%)")
    
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
    
    if prompt_type not in ["shap", "narrative", "cf"]:
        print(f"Error: Invalid prompt_type '{prompt_type}'. Choose from: shap, narrative, cf")
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
