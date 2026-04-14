"""
Compare narratives for adversely predicted instances across protected attributes.

Edit the USER CONFIGURATION section below to change dataset, number of narratives, etc.
Then press the play button to run.
"""

import json
import sys
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

# Import sentiment analysis - use VADER (lightweight, rule-based)
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Warning: nltk not installed. Install with: pip install nltk")

# Import readability analysis
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("Warning: textstat not installed. Install with: pip install textstat")


# ============================================================================
# USER CONFIGURATION - CHANGE THESE VALUES
# ============================================================================

# Which dataset to analyze
DATASET = "law"  # Options: "law", "credit", "saudi", "student"

# Number of adversely predicted instances to analyze
NUM_INSTANCES = 150

# Type of explanation narratives
PROMPT_TYPE = "cf"  # Options: "shap" or "cf" 

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

class NarrativeAnalyzer:
    """Framework for analyzing narratives. Users can add specific analyzers."""
    
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
    
    # ========================================================================
    # SENTIMENT ANALYSIS FOR LAW SCHOOL ADMISSION NARRATIVES
    # ========================================================================
    
    def analyze_sentiment(self, narratives=None):
        """
        Analyze sentiment of law school admission narratives using VADER.
        
        VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lightweight,
        rule-based sentiment analyzer that doesn't require model downloads.
        Works well for text with mixed positive/negative expression.
        
        Args:
            narratives: Optional dict of narratives to analyze. 
                       If None, uses self.narratives
        
        Returns:
            Dictionary mapping instance_idx -> {"label": str, "score": float}
            where label is POSITIVE, NEGATIVE, or NEUTRAL and score is -1 to 1
        """
        if narratives is None:
            narratives = self.narratives
        
        if not narratives:
            return {}
        
        if not SENTIMENT_AVAILABLE:
            print("Error: nltk library not available. Install with: pip install nltk")
            return {idx: {"label": "UNKNOWN", "score": 0.0} for idx in narratives.keys()}
        
        # Initialize VADER sentiment analyzer (no model download needed)
        print("Initializing VADER sentiment analyzer...")
        sia = SentimentIntensityAnalyzer()
        print("VADER analyzer ready (lightweight, rule-based).")
        
        sentiments = {}
        total = len(narratives)
        
        for idx, (instance_idx, narrative) in enumerate(narratives.items()):
            if (idx + 1) % 10 == 0:
                print(f"  Processing narrative {idx + 1}/{total}...")
            
            try:
                # Get VADER sentiment scores
                scores = sia.polarity_scores(narrative)
                
                # compound score ranges from -1 (most negative) to 1 (most positive)
                # Classify: negative < -0.1, positive > 0.1, otherwise neutral
                compound = scores['compound']
                
                if compound >= 0.1:
                    label = "POSITIVE"
                elif compound <= -0.1:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
                
                sentiments[instance_idx] = {
                    "label": label,
                    "score": float(compound),  # -1 to 1 scale
                    "pos": float(scores['pos']),
                    "neu": float(scores['neu']),
                    "neg": float(scores['neg']),
                }
            
            except Exception as e:
                print(f"  Error processing instance {instance_idx}: {e}")
                sentiments[instance_idx] = {"label": "ERROR", "score": 0.0}
        
        return sentiments
    
    # ========================================================================
    # 1. EMPATHY/TONE ANALYSIS
    # ========================================================================
    
    def analyze_empathy(self, narratives=None):
        """
        Analyze empathy and tone of narratives.
        
        Detects empathetic language vs blame language.
        - Empathetic: "unfortunately", "despite", "challenging", "circumstances"
        - Blame: "failed", "couldn't", "weakness", "lack of"
        
        Args:
            narratives: Optional dict of narratives to analyze
        
        Returns:
            Dictionary mapping instance_idx -> {"empathy_score": float, "blame_score": float}
        """
        if narratives is None:
            narratives = self.narratives
        
        empathy_words = ["unfortunately", "despite", "challenging", "circumstances", 
                        "difficult", "struggled", "limited", "however", "although",
                        "still", "remarkable", "nonetheless"]
        blame_words = ["failed", "couldn't", "couldn't", "weakness", "lack", "insufficient",
                      "poor", "low", "difficult", "underperformed", "mistake"]
        
        results = {}
        total = len(narratives)
        
        for idx, (instance_idx, narrative) in enumerate(narratives.items()):
            if (idx + 1) % 10 == 0:
                print(f"  Processing narrative {idx + 1}/{total}...")
            
            try:
                text_lower = narrative.lower()
                
                # Count empathy and blame words
                empathy_count = sum(text_lower.count(word) for word in empathy_words)
                blame_count = sum(text_lower.count(word) for word in blame_words)
                
                # Normalize by text length
                word_count = len(narrative.split())
                empathy_score = empathy_count / max(word_count, 1) if word_count > 0 else 0
                blame_score = blame_count / max(word_count, 1) if word_count > 0 else 0
                
                # Overall empathy metric: empathy - blame
                overall_empathy = empathy_score - blame_score
                
                results[instance_idx] = {
                    "empathy_score": float(empathy_score),
                    "blame_score": float(blame_score),
                    "overall_tone": float(overall_empathy),  # Positive = empathetic, Negative = blaming
                }
            
            except Exception as e:
                print(f"  Error processing instance {instance_idx}: {e}")
                results[instance_idx] = {"empathy_score": 0.0, "blame_score": 0.0, "overall_tone": 0.0}
        
        return results
    
    # ========================================================================
    # 2. SPECIFICITY ANALYSIS (Feature Coverage)
    # ========================================================================
    
    def analyze_specificity(self, narratives=None):
        """
        Analyze specificity of narratives - how many concrete details/numbers mentioned.
        
        Measures:
        - Number of sentences
        - Average sentence length
        - Mentions of numbers/percentages/specific values
        - Feature coverage (how many different features are discussed)
        
        Args:
            narratives: Optional dict of narratives to analyze
        
        Returns:
            Dictionary mapping instance_idx -> specificity metrics
        """
        if narratives is None:
            narratives = self.narratives
        
        results = {}
        total = len(narratives)
        
        for idx, (instance_idx, narrative) in enumerate(narratives.items()):
            if (idx + 1) % 10 == 0:
                print(f"  Processing narrative {idx + 1}/{total}...")
            
            try:
                # Count sentences
                sentences = [s.strip() for s in narrative.split('.') if s.strip()]
                sentence_count = len(sentences)
                
                # Average sentence length
                avg_sentence_length = len(narrative.split()) / max(sentence_count, 1)
                
                # Count numbers/specific values (integers, decimals, percentages)
                import re
                numbers = re.findall(r'\d+\.?\d*\s*%?|\b\d+(?:\.\d+)?\b', narrative)
                number_count = len(numbers)
                
                # Count mentions of key explanation words (indicating feature discussion)
                explanation_words = ["feature", "value", "score", "important", "contributed", 
                                   "because", "reason", "due to", "compared to", "average",
                                   "typically", "usually", "percentage", "rate"]
                explanation_mentions = sum(1 for word in explanation_words 
                                         if word in narrative.lower())
                
                specificity_score = (number_count + explanation_mentions) / max(
                    len(narrative.split()), 1)
                
                results[instance_idx] = {
                    "sentence_count": float(sentence_count),
                    "avg_sentence_length": float(avg_sentence_length),
                    "number_mentions": float(number_count),
                    "explanation_detail": float(explanation_mentions),
                    "specificity_score": float(specificity_score),
                }
            
            except Exception as e:
                print(f"  Error processing instance {instance_idx}: {e}")
                results[instance_idx] = {
                    "sentence_count": 0.0,
                    "avg_sentence_length": 0.0,
                    "number_mentions": 0.0,
                    "explanation_detail": 0.0,
                    "specificity_score": 0.0,
                }
        
        return results
    
    # ========================================================================
    # 3. READABILITY ANALYSIS (Flesch-Kincaid)
    # ========================================================================
    
    def analyze_readability(self, narratives=None):
        """
        Analyze readability of narratives using Flesch-Kincaid metrics.
        
        Measures:
        - Flesch-Kincaid Grade Level (what US school grade is needed to understand)
        - Flesch Reading Ease (0-100, higher = easier)
        
        Args:
            narratives: Optional dict of narratives to analyze
        
        Returns:
            Dictionary mapping instance_idx -> readability metrics
        """
        if narratives is None:
            narratives = self.narratives
        
        if not TEXTSTAT_AVAILABLE:
            print("Warning: textstat not available. Install with: pip install textstat")
            return {idx: {"grade_level": 0.0, "reading_ease": 0.0} for idx in narratives.keys()}
        
        results = {}
        total = len(narratives)
        
        for idx, (instance_idx, narrative) in enumerate(narratives.items()):
            if (idx + 1) % 10 == 0:
                print(f"  Processing narrative {idx + 1}/{total}...")
            
            try:
                # Calculate Flesch-Kincaid metrics
                grade_level = textstat.flesch_kincaid_grade(narrative)
                reading_ease = textstat.flesch_reading_ease(narrative)
                
                results[instance_idx] = {
                    "flesch_kincaid_grade": float(grade_level),
                    "flesch_reading_ease": float(reading_ease),  # 0-100: 90+ = very easy, 60-70 = standard, <30 = difficult
                }
            
            except Exception as e:
                print(f"  Error processing instance {instance_idx}: {e}")
                results[instance_idx] = {"flesch_kincaid_grade": 0.0, "flesch_reading_ease": 0.0}
        
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
        
        # Sentiment Analysis
        print("\n1. Sentiment Analysis (VADER)...")
        try:
            self.results["sentiment"] = self.analyze_sentiment()
        except Exception as e:
            print(f"Error: {e}")
            raise
        
        # Empathy/Tone Analysis
        print("\n2. Empathy/Tone Analysis...")
        try:
            self.results["empathy"] = self.analyze_empathy()
        except Exception as e:
            print(f"Error: {e}")
            self.results["empathy"] = {}
        
        # Specificity Analysis
        print("\n3. Specificity Analysis...")
        try:
            self.results["specificity"] = self.analyze_specificity()
        except Exception as e:
            print(f"Error: {e}")
            self.results["specificity"] = {}
        
        # Readability Analysis
        print("\n4. Readability Analysis...")
        try:
            self.results["readability"] = self.analyze_readability()
        except Exception as e:
            print(f"Error: {e}")
            self.results["readability"] = {}
        
        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70)
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
    
    # Print narrative statistics
    print("\nOVERALL NARRATIVE STATISTICS:")
    stats = get_narrative_statistics(analyzer.narratives)
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Print statistics by protected attributes
    print("\nNARRATIVE STATISTICS BY PROTECTED ATTRIBUTE:")
    protected_attrs = PROTECTED_ATTRIBUTES[dataset]
    
    for attr in protected_attrs:
        print(f"\n  {attr.upper()}:")
        unique_values = adverse_df[attr].unique()
        
        for value in sorted(unique_values):
            matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
            group_narratives = {idx: analyzer.narratives[idx] for idx in matching_instances if idx in analyzer.narratives}
            
            if group_narratives:
                group_stats = get_narrative_statistics(group_narratives)
                mapped_value = map_attribute_value(value, dataset, attr)
                print(f"    {mapped_value}:")
                for key, val in group_stats.items():
                    print(f"      {key}: {val:.2f}" if isinstance(val, float) else f"      {key}: {val}")
    
    # Print sentiment analysis by protected attributes
    if "sentiment" in analyzer.results:
        print("\nSENTIMENT ANALYSIS BY PROTECTED ATTRIBUTE:")
        sentiment_results = analyzer.results["sentiment"]
        
        for attr in protected_attrs:
            print(f"\n  {attr.upper()}:")
            unique_values = adverse_df[attr].unique()
            
            for value in sorted(unique_values):
                # Get instances with this attribute value
                matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
                
                # Get sentiment scores for these instances
                scores = [sentiment_results.get(idx, {}).get("score", 0) for idx in matching_instances if idx in sentiment_results]
                labels = [sentiment_results.get(idx, {}).get("label", "UNKNOWN") for idx in matching_instances if idx in sentiment_results]
                
                if scores:
                    label_counts = pd.Series(labels).value_counts().to_dict()
                    mapped_value = map_attribute_value(value, dataset, attr)
                    print(f"    {mapped_value}:")
                    print(f"      Count: {len(scores)}")
                    print(f"      Avg Sentiment Score: {np.mean(scores):.3f} (±{np.std(scores):.3f})")
                    print(f"      Label Distribution: {label_counts}")
    
    # Print empathy analysis by protected attributes
    if "empathy" in analyzer.results and analyzer.results["empathy"]:
        print("\nEMPATHY/TONE ANALYSIS BY PROTECTED ATTRIBUTE:")
        empathy_results = analyzer.results["empathy"]
        
        for attr in protected_attrs:
            print(f"\n  {attr.upper()}:")
            unique_values = adverse_df[attr].unique()
            
            for value in sorted(unique_values):
                matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
                
                # Get empathy scores for these instances
                empathy_scores = [empathy_results.get(idx, {}).get("empathy_score", 0) for idx in matching_instances if idx in empathy_results]
                blame_scores = [empathy_results.get(idx, {}).get("blame_score", 0) for idx in matching_instances if idx in empathy_results]
                
                if empathy_scores:
                    mapped_value = map_attribute_value(value, dataset, attr)
                    print(f"    {mapped_value}:")
                    print(f"      Count: {len(empathy_scores)}")
                    print(f"      Avg Empathy Score: {np.mean(empathy_scores):.3f} (±{np.std(empathy_scores):.3f})")
                    print(f"      Avg Blame Score: {np.mean(blame_scores):.3f} (±{np.std(blame_scores):.3f})")
                    print(f"      Overall Tone: {np.mean(empathy_scores) - np.mean(blame_scores):.3f} ({'empathetic' if np.mean(empathy_scores) > np.mean(blame_scores) else 'blame-focused'})")
    
    # Print specificity analysis by protected attributes
    if "specificity" in analyzer.results and analyzer.results["specificity"]:
        print("\nSPECIFICITY ANALYSIS BY PROTECTED ATTRIBUTE:")
        specificity_results = analyzer.results["specificity"]
        
        for attr in protected_attrs:
            print(f"\n  {attr.upper()}:")
            unique_values = adverse_df[attr].unique()
            
            for value in sorted(unique_values):
                matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
                
                # Get specificity metrics for these instances
                specificity_scores = [specificity_results.get(idx, {}).get("specificity_score", 0) for idx in matching_instances if idx in specificity_results]
                sentence_counts = [specificity_results.get(idx, {}).get("sentence_count", 0) for idx in matching_instances if idx in specificity_results]
                number_mentions = [specificity_results.get(idx, {}).get("number_mentions", 0) for idx in matching_instances if idx in specificity_results]
                
                if specificity_scores:
                    mapped_value = map_attribute_value(value, dataset, attr)
                    print(f"    {mapped_value}:")
                    print(f"      Count: {len(specificity_scores)}")
                    print(f"      Avg Specificity Score: {np.mean(specificity_scores):.3f} (±{np.std(specificity_scores):.3f})")
                    print(f"      Avg Sentence Count: {np.mean(sentence_counts):.1f} (±{np.std(sentence_counts):.1f})")
                    print(f"      Avg Number Mentions: {np.mean(number_mentions):.1f} (±{np.std(number_mentions):.1f})")
    
    # Print readability analysis by protected attributes
    if "readability" in analyzer.results and analyzer.results["readability"]:
        print("\nREADABILITY ANALYSIS BY PROTECTED ATTRIBUTE:")
        readability_results = analyzer.results["readability"]
        
        for attr in protected_attrs:
            print(f"\n  {attr.upper()}:")
            unique_values = adverse_df[attr].unique()
            
            for value in sorted(unique_values):
                matching_instances = adverse_df[adverse_df[attr] == value]["instance_index"].astype(int).tolist()
                
                # Get readability metrics for these instances
                grade_levels = [readability_results.get(idx, {}).get("flesch_kincaid_grade", 0) for idx in matching_instances if idx in readability_results]
                reading_eases = [readability_results.get(idx, {}).get("flesch_reading_ease", 0) for idx in matching_instances if idx in readability_results]
                
                if grade_levels:
                    mapped_value = map_attribute_value(value, dataset, attr)
                    print(f"    {mapped_value}:")
                    print(f"      Count: {len(grade_levels)}")
                    print(f"      Avg Grade Level: {np.mean(grade_levels):.1f} (±{np.std(grade_levels):.1f})")
                    print(f"      Avg Reading Ease: {np.mean(reading_eases):.1f} (±{np.std(reading_eases):.1f})")
                    difficulty = "Very Easy" if np.mean(reading_eases) >= 90 else \
                                "Easy" if np.mean(reading_eases) >= 80 else \
                                "Standard" if np.mean(reading_eases) >= 60 else \
                                "Difficult" if np.mean(reading_eases) >= 30 else \
                                "Very Difficult"
                    print(f"      Difficulty Level: {difficulty}")
    
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
