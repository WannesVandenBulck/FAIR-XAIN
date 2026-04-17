"""
Compare Bias Injection Experiment Narratives

This script compares narratives across different demographic overrides (batches)
using the same sentiment analysis metrics as compare_narratives.py.

Usage:
1. Edit the CONFIGURATION section below
2. Run the script
3. Results are saved to RESULTS_OUTPUT_DIR
"""

import json
import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add parent directory to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import HuggingFace transformers for sentiment analysis
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")


# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================

# Dataset
DATASET = "law"

# Which batches to compare (batch names from bias_injection_experiment.py)
BATCHES = [
    "white_female",
    "black_female",
    "hispanic_female", 
    "asian_female", 
    "native_american_female",
]

# LLM Provider and Model used for the narratives
PROVIDER = "grok"
MODEL = "grok-4-1-fast-non-reasoning"

# Output directory for comparison results
RESULTS_OUTPUT_DIR = "results/bias_comparison"

# Number of instances to analyze (None = all available)
NUM_INSTANCES = None

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Label interpretations for reference
EMPATHY_STRATEGY_LABELS = {
    "Affirmation and Reassurance": "Validating and encouraging; building confidence",
    "Information": "Providing factual knowledge or explanations",
    "Others": "General or non-specific support; miscellaneous responses",
    "Providing Suggestions": "Offering concrete advice or action steps",
    "Question": "Asking clarifying or exploratory questions",
    "Reflection of feelings": "Mirroring or acknowledging emotions",
    "Restatement or Paraphrasing": "Restating information to show understanding",
    "Self-disclosure": "Sharing personal experiences or vulnerabilities",
}


def load_bias_narratives(dataset, batch_name, provider, model, num_instances=None):
    """
    Load narratives from a specific bias injection batch.
    
    Args:
        dataset: Dataset name (e.g., "law")
        batch_name: Batch name (e.g., "white_male", "black_female")
        provider: LLM provider (e.g., "grok")
        model: LLM model name
        num_instances: Number of instances to load (None = all)
    
    Returns:
        Dictionary mapping instance_idx -> narrative text
    """
    narratives = {}
    base_dir = Path(f"results/narratives/{dataset}/narratives/shap/{provider}_{batch_name}/{model}")
    
    if not base_dir.exists():
        print(f"ERROR: Batch directory does not exist: {base_dir}")
        return narratives
    
    # Get all instance JSON files
    json_files = sorted(base_dir.glob("instance_*.json"))
    
    if num_instances:
        json_files = json_files[:num_instances]
    
    print(f"  Loading {len(json_files)} narratives from {batch_name}...")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("status") == "success" and data.get("narrative"):
                    instance_idx = data.get("instance_idx")
                    narratives[instance_idx] = data.get("narrative")
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")
    
    print(f"  Loaded {len(narratives)} narratives")
    return narratives


def get_narrative_statistics(narratives):
    """Compute basic statistics on narratives."""
    if not narratives:
        return {}
    
    lengths = [len(text.split()) for text in narratives.values()]
    
    return {
        "count": len(narratives),
        "avg_words": np.mean(lengths) if lengths else 0,
        "avg_chars": np.mean([len(text) for text in narratives.values()]) if narratives else 0,
        "min_words": min(lengths) if lengths else 0,
        "max_words": max(lengths) if lengths else 0,
        "std_words": np.std(lengths) if lengths else 0,
    }


class BiasNarrativeAnalyzer:
    """Analyze narratives for bias patterns across demographic batches."""
    
    def __init__(self, narratives_by_batch):
        """
        Initialize analyzer.
        
        Args:
            narratives_by_batch: Dict mapping batch_name -> {instance_idx -> narrative}
        """
        self.narratives_by_batch = narratives_by_batch
        self.results = {}
    
    def build_metadata_overview(self):
        """
        Build a detailed overview of the analysis for report metadata.
        
        Returns:
            Dict with comprehensive metadata about the comparison.
        """
        # Calculate statistics about narratives
        batch_details = {}
        total_narratives = 0
        
        for batch_name, narratives in self.narratives_by_batch.items():
            count = len(narratives)
            total_narratives += count
            batch_details[batch_name] = {
                "narrative_count": count,
                "demographics": self._parse_batch_demographics(batch_name)
            }
        
        # Parse demographic groups
        demographics_description = self._get_demographics_description()
        
        metadata = {
            "report_type": "Bias Injection Experiment Comparison",
            "analysis_timestamp": datetime.now().isoformat(),
            "dataset": DATASET,
            "comparison_summary": {
                "total_batches": len(self.narratives_by_batch),
                "total_narratives": total_narratives,
                "batches_compared": list(self.narratives_by_batch.keys()),
                "batch_details": batch_details
            },
            "llm_configuration": {
                "provider": PROVIDER,
                "model": MODEL
            },
            "demographics_explanation": demographics_description,
            "analyses_included": [
                "protected_mentions",
                "emotion_distilroberta",
                "emotion_go_emotions",
                "empathy_strategy"
            ],
            "summary": f"This report compares {total_narratives} narratives across {len(self.narratives_by_batch)} demographic groups "
                      f"({', '.join(self.narratives_by_batch.keys())}) generated using {PROVIDER} ({MODEL}). "
                      f"Each group represents narratives where the demographic attributes were overridden to specific demographic profiles. "
                      f"The comparison includes sentiment analysis (DistilRoBERTa and GoEmotions), "
                      f"mentions of protected attributes (gender and race), and empathy strategy detection."
        }
        
        return metadata
    
    def _parse_batch_demographics(self, batch_name):
        """Parse batch name to extract demographic information."""
        parts = batch_name.lower().split('_')
        demographics = {}
        
        # Common demographic values in batch names
        races = ['white', 'black', 'hispanic', 'asian', 'native']
        genders = ['male', 'female']
        
        for part in parts:
            if part in races or any(part in race for race in races):
                demographics['race'] = part.replace('native', 'native american')
            elif part in genders:
                demographics['gender'] = part
        
        return demographics
    
    def _get_demographics_description(self):
        """Build a human-readable description of the demographic groups being compared."""
        descriptions = []
        
        for batch_name in sorted(self.narratives_by_batch.keys()):
            demo = self._parse_batch_demographics(batch_name)
            parts = []
            
            if 'race' in demo:
                parts.append(f"race: {demo['race'].title()}")
            if 'gender' in demo:
                parts.append(f"gender: {demo['gender'].title()}")
            
            if parts:
                narratives_count = len(self.narratives_by_batch[batch_name])
                descriptions.append(f"  - {batch_name}: {', '.join(parts)} ({narratives_count} narratives)")
        
        return "Demographic groups compared:\n" + "\n".join(descriptions) if descriptions else ""
    
    def analyze_protected_mentions(self):
        """Count mentions of protected attributes (gender and race) in narratives."""
        print("\n" + "="*70)
        print("1. Protected Attribute Mentions Analysis")
        print("="*70)
        
        gender_terms = [
            "Male", "male", "MALE",
            "Female", "female", "FEMALE",
            "Man", "man", "MAN",
            "Woman", "woman", "WOMAN",
            "men", "MEN",
            "women", "WOMEN",
            "boy", "Boy", "BOY",
            "girl", "Girl", "GIRL",
            "masculine", "Masculine", "MASCULINE",
            "feminine", "Feminine", "FEMININE",
            "he", "He", "HE",
            "she", "She", "SHE",
            "his", "His", "HIS",
            "her", "Her", "HER",
        ]
        race_terms = [
            "Black", "black", "BLACK",
            "Hispanic", "hispanic", "HISPANIC",
            "Asian", "asian", "ASIAN",
            "White", "white", "WHITE",
            "Native American", "native american", "NATIVE AMERICAN",
            "African", "african", "AFRICAN",
            "Latino", "latino", "LATINO",
            "Caucasian", "caucasian", "CAUCASIAN",
        ]
        
        gender_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in gender_terms]
        race_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in race_terms]
        
        results_by_batch = {}
        
        for batch_name, narratives in self.narratives_by_batch.items():
            print(f"\nBatch: {batch_name}")
            gender_mentions = []
            race_mentions = []
            
            for instance_idx, narrative in narratives.items():
                gender_count = sum(len(p.findall(narrative)) for p in gender_patterns)
                race_count = sum(len(p.findall(narrative)) for p in race_patterns)
                
                gender_mentions.append(gender_count)
                race_mentions.append(race_count)
            
            results_by_batch[batch_name] = {
                "avg_gender_mentions": np.mean(gender_mentions) if gender_mentions else 0,
                "avg_race_mentions": np.mean(race_mentions) if race_mentions else 0,
                "std_gender_mentions": np.std(gender_mentions) if gender_mentions else 0,
                "std_race_mentions": np.std(race_mentions) if race_mentions else 0,
                "total_gender_mentions": sum(gender_mentions),
                "total_race_mentions": sum(race_mentions),
            }
            
            print(f"  Avg gender mentions: {results_by_batch[batch_name]['avg_gender_mentions']:.2f}")
            print(f"  Avg race mentions: {results_by_batch[batch_name]['avg_race_mentions']:.2f}")
        
        self.results["protected_mentions"] = results_by_batch
        return results_by_batch
    
    def analyze_emotion_distilroberta(self):
        """Analyze emotions using DistilRoBERTa."""
        print("\n" + "="*70)
        print("2. Emotion Analysis (DistilRoBERTa)")
        print("="*70)
        
        if not TRANSFORMERS_AVAILABLE:
            print("ERROR: transformers not available")
            return {}
        
        print("Loading DistilRoBERTa emotion model...")
        classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        
        results_by_batch = {}
        emotion_counts = defaultdict(int)
        
        for batch_name, narratives in self.narratives_by_batch.items():
            print(f"\nBatch: {batch_name}")
            batch_emotions = defaultdict(int)
            scores_by_emotion = defaultdict(list)
            total = len(narratives)
            
            for idx, (instance_idx, narrative) in enumerate(narratives.items()):
                if (idx + 1) % 50 == 0:
                    print(f"  Processing narrative {idx + 1}/{total}...")
                
                try:
                    narrative_short = narrative[:512]
                    prediction = classifier(narrative_short)[0]
                    emotion = prediction["label"]
                    score = float(prediction["score"])
                    
                    batch_emotions[emotion] += 1
                    scores_by_emotion[emotion].append(score)
                except Exception as e:
                    print(f"  Error: {e}")
            
            # Calculate averages by emotion
            emotion_stats = {}
            for emotion, count in batch_emotions.items():
                scores = scores_by_emotion[emotion]
                emotion_stats[emotion] = {
                    "count": count,
                    "percentage": (count / total * 100) if total > 0 else 0,
                    "avg_confidence": np.mean(scores) if scores else 0,
                }
            
            results_by_batch[batch_name] = emotion_stats
            
            print(f"  Emotion distribution:")
            for emotion, stats in sorted(emotion_stats.items(), key=lambda x: x[1]["count"], reverse=True):
                print(f"    {emotion}: {stats['count']} ({stats['percentage']:.1f}%) - avg confidence: {stats['avg_confidence']:.3f}")
        
        self.results["emotion_distilroberta"] = results_by_batch
        return results_by_batch
    
    def analyze_emotion_go_emotions(self):
        """Analyze emotions using Go Emotions."""
        print("\n" + "="*70)
        print("3. Emotion Analysis (Go Emotions)")
        print("="*70)
        
        if not TRANSFORMERS_AVAILABLE:
            print("ERROR: transformers not available")
            return {}
        
        print("Loading Go Emotions model...")
        classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
        
        results_by_batch = {}
        
        for batch_name, narratives in self.narratives_by_batch.items():
            print(f"\nBatch: {batch_name}")
            batch_emotions = defaultdict(int)
            scores_by_emotion = defaultdict(list)
            total = len(narratives)
            
            for idx, (instance_idx, narrative) in enumerate(narratives.items()):
                if (idx + 1) % 50 == 0:
                    print(f"  Processing narrative {idx + 1}/{total}...")
                
                try:
                    narrative_short = narrative[:512]
                    predictions = classifier(narrative_short)[0]
                    
                    # Get top emotion
                    top_pred = max(predictions, key=lambda x: x["score"])
                    emotion = top_pred["label"]
                    score = float(top_pred["score"])
                    
                    batch_emotions[emotion] += 1
                    scores_by_emotion[emotion].append(score)
                except Exception as e:
                    print(f"  Error: {e}")
            
            # Calculate averages by emotion
            emotion_stats = {}
            for emotion, count in batch_emotions.items():
                scores = scores_by_emotion[emotion]
                emotion_stats[emotion] = {
                    "count": count,
                    "percentage": (count / total * 100) if total > 0 else 0,
                    "avg_confidence": np.mean(scores) if scores else 0,
                }
            
            results_by_batch[batch_name] = emotion_stats
            
            print(f"  Emotion distribution:")
            for emotion, stats in sorted(emotion_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:10]:
                print(f"    {emotion}: {stats['count']} ({stats['percentage']:.1f}%) - avg confidence: {stats['avg_confidence']:.3f}")
        
        self.results["emotion_go_emotions"] = results_by_batch
        return results_by_batch
    
    def analyze_empathy_strategy(self):
        """Analyze empathy using empathy strategy classifier."""
        print("\n" + "="*70)
        print("4. Empathy Strategy Analysis")
        print("="*70)
        
        if not TRANSFORMERS_AVAILABLE:
            print("ERROR: transformers not available")
            return {}
        
        print("Loading Empathy Strategy model...")
        classifier = pipeline("text-classification", model="RyanDDD/empathy-strategy-classifier")
        
        results_by_batch = {}
        
        for batch_name, narratives in self.narratives_by_batch.items():
            print(f"\nBatch: {batch_name}")
            empathy_labels = defaultdict(int)
            scores_by_label = defaultdict(list)
            total = len(narratives)
            
            for idx, (instance_idx, narrative) in enumerate(narratives.items()):
                if (idx + 1) % 50 == 0:
                    print(f"  Processing narrative {idx + 1}/{total}...")
                
                try:
                    narrative_short = narrative[:512]
                    prediction = classifier(narrative_short)[0]
                    label = prediction["label"]
                    score = float(prediction["score"])
                    
                    empathy_labels[label] += 1
                    scores_by_label[label].append(score)
                except Exception as e:
                    print(f"  Error: {e}")
            
            # Calculate stats
            label_stats = {}
            for label, count in empathy_labels.items():
                scores = scores_by_label[label]
                label_stats[label] = {
                    "count": count,
                    "percentage": (count / total * 100) if total > 0 else 0,
                    "avg_score": np.mean(scores) if scores else 0,
                    "meaning": EMPATHY_STRATEGY_LABELS.get(label, "Unknown label"),
                }
            
            results_by_batch[batch_name] = label_stats
            
            print(f"  Empathy Strategy distribution:")
            for label, stats in sorted(label_stats.items(), key=lambda x: x[1]["count"], reverse=True):
                print(f"    {label}: {stats['count']} ({stats['percentage']:.1f}%) - avg score: {stats['avg_score']:.3f}")
                print(f"      Meaning: {stats['meaning']}")
        
        self.results["empathy_strategy"] = results_by_batch
        self.results["empathy_strategy_meanings"] = EMPATHY_STRATEGY_LABELS
        return results_by_batch
    
    def run_analysis(self):
        """Run all analyses."""
        print("\n" + "*"*70)
        print("BIAS INJECTION EXPERIMENT COMPARISON")
        print("*"*70)
        print(f"Dataset: {DATASET}")
        print(f"Batches: {list(self.narratives_by_batch.keys())}")
        print(f"Started: {datetime.now().isoformat()}")
        
        # Build metadata overview first - this will appear at the start of the JSON
        print("\n" + "="*70)
        print("Building Report Metadata")
        print("="*70)
        metadata = self.build_metadata_overview()
        self.results["metadata"] = metadata
        
        # Basic statistics
        print("\n" + "="*70)
        print("0. Narrative Statistics")
        print("="*70)
        for batch_name, narratives in self.narratives_by_batch.items():
            stats = get_narrative_statistics(narratives)
            print(f"\n{batch_name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Avg words: {stats['avg_words']:.1f}")
            print(f"  Avg chars: {stats['avg_chars']:.0f}")
        
        # Run all analyses
        self.analyze_protected_mentions()
        self.analyze_emotion_distilroberta()
        self.analyze_emotion_go_emotions()
        self.analyze_empathy_strategy()
        
        return self.results
    
    def save_results(self, output_dir):
        """Save analysis results to JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        filepath = Path(output_dir) / f"bias_comparison_{DATASET}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main execution."""
    print("Loading narratives from bias injection batches...")
    
    narratives_by_batch = {}
    for batch_name in BATCHES:
        print(f"\nBatch: {batch_name}")
        narratives = load_bias_narratives(DATASET, batch_name, PROVIDER, MODEL, NUM_INSTANCES)
        if narratives:
            narratives_by_batch[batch_name] = narratives
    
    if not narratives_by_batch:
        print("ERROR: No narratives loaded!")
        return
    
    # Run analysis
    analyzer = BiasNarrativeAnalyzer(narratives_by_batch)
    results = analyzer.run_analysis()
    
    # Save results
    analyzer.save_results(RESULTS_OUTPUT_DIR)
    
    print("\n" + "*"*70)
    print("ANALYSIS COMPLETE")
    print("*"*70)


if __name__ == "__main__":
    main()
