"""
Faithfulness and Hallucination Metrics for Bias Injection Narratives

This script computes:
1. Faithfulness: LLM extracts features, then compares rank/sign/value vs actual SHAP (following GitHub repo exactly)
2. Perplexity: Language model perplexity of narratives as hallucination indicator

Following methodology from: https://github.com/ADMAntwerp/SHAPnarrative-metrics
"""

import json
import sys
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import LLM client for extraction
try:
    from llm_tools.llm_client import LLMClient
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: llm_tools.llm_client not available")

# Import perplexity models
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/torch not available. Install with: pip install transformers torch bitsandbytes")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset and batches (None = auto-discover all batches in results directory)
DATASET = "law"
BATCHES = None  # Set None to auto-discover, or list: ["white_male", "black_female"]

# LLM details for narrative loading
PROVIDER = "grok"
MODEL = "grok-4-1-fast-non-reasoning"

# Extraction and perplexity settings
EXTRACTION_LLM = "gpt-4o"  # For feature extraction from narratives
PPL_MODELS = ["meta-llama/Meta-Llama-3-8B"]  # For perplexity computation

# Output directory
RESULTS_OUTPUT_DIR = "results/faithfulness_metrics"

# Number of instances (None = all)
NUM_INSTANCES = None

# ============================================================================
# END CONFIGURATION
# ============================================================================


def discover_all_batches(dataset, provider, model):
    """Auto-discover all available batch directories."""
    base_dir = Path(f"results/narratives/{dataset}/narratives/shap")
    batches = []
    
    if not base_dir.exists():
        print(f"ERROR: Results directory does not exist: {base_dir}")
        return []
    
    # Find all directories matching provider_batchname/model
    for provider_batch_dir in base_dir.glob(f"{provider}_*"):
        model_dir = provider_batch_dir / model
        if model_dir.exists():
            batch_name = provider_batch_dir.name.replace(f"{provider}_", "")
            batches.append(batch_name)
    
    return sorted(batches)


def load_shap_values(dataset, instance_indices=None):
    """Load SHAP values for instances."""
    shap_path = Path(f"datasets_prep/data/{dataset}_dataset/{dataset}_shap.csv")
    shap_df = pd.read_csv(shap_path)
    
    if instance_indices:
        shap_df = shap_df[shap_df['instance_index'].isin(instance_indices)]
    
    return shap_df


def load_narratives_from_batch(dataset, batch_name, provider, model, num_instances=None):
    """Load narratives from a bias injection batch."""
    narratives = {}
    base_dir = Path(f"results/narratives/{dataset}/narratives/shap/{provider}_{batch_name}/{model}")
    
    if not base_dir.exists():
        print(f"ERROR: Batch directory does not exist: {base_dir}")
        return narratives, []
    
    json_files = sorted(base_dir.glob("instance_*.json"))
    
    if num_instances:
        json_files = json_files[:num_instances]
    
    instance_indices = []
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("status") == "success" and data.get("narrative"):
                    instance_idx = data.get("instance_idx")
                    narratives[instance_idx] = data.get("narrative")
                    instance_indices.append(instance_idx)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return narratives, sorted(instance_indices)


class ExtractionModel:
    """
    Extract features and their SHAP value estimates from narratives.
    Follows exact implementation from GitHub repo: ADMAntwerp/SHAPnarrative-metrics
    """
    
    def __init__(self, llm_client):
        """Initialize with LLM client."""
        self.llm_client = llm_client
    
    def generate_extraction_prompt(self, narrative_text, dataset_description=""):
        """
        Generate extraction prompt following the GitHub repo approach.
        Asks LLM to extract features mentioned in narrative as a structured dict.
        """
        prompt = f"""An LLM was asked to create a narrative explaining an AI model's prediction. 
The narrative is shown below.

Dataset description: {dataset_description}

Narrative:
{narrative_text}

Your task is to extract the features mentioned in the narrative as reasons for the prediction.
Provide your answer as a Python dictionary where:
- Keys are feature names (use EXACT feature names as they appear in the narrative)
- Values are dictionaries with:
  - "rank": importance order (0 = most important, 1 = second, etc.)
  - "sign": +1 if feature contributed to outcome 1, -1 if contributed to outcome 0
  - "value": numeric value if explicitly mentioned, otherwise None
  - "assumption": 1-sentence summary of the feature's contribution to the prediction

Example format:
{{"feature_1": {{"rank": 0, "sign": 1, "value": 0.45, "assumption": "Feature 1 value was high, supporting the prediction."}},
  "feature_2": {{"rank": 1, "sign": -1, "value": None, "assumption": "Feature 2 was below average, reducing the prediction probability."}}}}

IMPORTANT: Provide ONLY the dictionary as output, no other text or explanation.
Only include features that were actually mentioned in the narrative."""
        return prompt
    
    def extract_dict_from_str(self, extracted_str):
        """Extract dictionary from LLM response."""
        try:
            start_index = extracted_str.find("{")
            end_index = extracted_str.rfind("}")
            if start_index == -1 or end_index == -1:
                return {}
            dict_str = extracted_str[start_index : end_index + 1]
            extracted_dict = ast.literal_eval(dict_str)
            return extracted_dict
        except Exception as e:
            print(f"Error extracting dict: {e}")
            return {}
    
    @staticmethod
    def get_diff(extracted_dict, explanation_df):
        """
        Compare extracted features to actual SHAP values.
        Follows exact implementation from GitHub repo's ExtractionModel.get_diff()
        
        Returns: (rank_diff, sign_diff, value_diff, real_rank, extracted_rank)
        Where:
        - 0 = correct match
        - non-zero number = error/difference
        - np.nan = non-numeric extracted value
        - np.inf = hallucinated feature (not in real data)
        """
        # Create dataframe from extracted dict
        if not extracted_dict:
            return [], [], [], [], []
        
        df_extracted = pd.DataFrame(extracted_dict).T
        df_extracted.reset_index(inplace=True)
        df_extracted.rename(columns={"index": "feature_name"}, inplace=True)
        
        # Ensure explanation is sorted by SHAP magnitude
        explanation_df = explanation_df.copy()
        explanation_df["abs_SHAP"] = explanation_df["SHAP_value"].abs()
        explanation_df = explanation_df.sort_values(by="abs_SHAP", ascending=False).reset_index(drop=True)
        explanation_df["rank"] = explanation_df.index
        
        # Create sign column from SHAP values
        explanation_df["sign"] = explanation_df["SHAP_value"].map(lambda x: int(np.sign(x)))
        
        # Find overlap
        df_extracted = df_extracted[df_extracted['feature_name'].isin(explanation_df['feature_name'])]
        df_real = explanation_df[explanation_df['feature_name'].isin(df_extracted['feature_name'])].sort_values(by="feature_name")
        
        rank_diff = []
        sign_diff = []
        value_diff = []
        real_rank = []
        extracted_rank = []
        
        # Compute differences
        for _, row in df_extracted.iterrows():
            feature = row['feature_name']
            real_row = df_real[df_real['feature_name'] == feature]
            
            if real_row.empty:
                # Hallucinated feature - add inf marker
                rank_diff.append(np.inf)
                sign_diff.append(np.inf)
                value_diff.append(np.inf)
                continue
            
            real_row = real_row.iloc[0]
            real_rank.append(int(real_row['rank']))
            
            # Rank difference
            extracted_r = row.get('rank')
            if isinstance(extracted_r, (int, float)) and not np.isnan(float(extracted_r)):
                rank_diff.append(abs(int(extracted_r) - int(real_row['rank'])))
                extracted_rank.append(int(extracted_r))
            else:
                rank_diff.append(np.nan)
                extracted_rank.append(np.nan)
            
            # Sign difference: 0 if same sign, 1 if different
            extracted_s = row.get('sign')
            if extracted_s in [-1, 1]:
                sign_diff.append(0 if extracted_s == real_row['sign'] else 1)
            else:
                sign_diff.append(np.nan)
            
            # Value difference
            extracted_v = row.get('value')
            if isinstance(extracted_v, (int, float)) and not np.isnan(float(extracted_v)):
                # If value is mentioned, compute difference from actual
                value_diff.append(abs(float(extracted_v) - float(real_row['SHAP_value'])))
            else:
                value_diff.append(np.nan)
        
        return rank_diff, sign_diff, value_diff, real_rank, extracted_rank
    
    def extract_narrative(self, narrative_text):
        """Extract features from a single narrative using LLM."""
        prompt = self.generate_extraction_prompt(narrative_text)
        response = self.llm_client.generate_response(prompt)
        extracted_dict = self.extract_dict_from_str(response)
        return extracted_dict


def average_zero(df):
    """
    Compute accuracy from difference dataframe.
    Following GitHub repo's average_zero() function.
    Returns: proportion of 0 values (exact matches) to all non-NaN values
    """
    if df.empty or df.size == 0:
        return 0
    
    # Flatten and get non-nan values
    flat = df.values.flatten()
    non_nan = flat[~np.isnan(flat)]
    
    if len(non_nan) == 0:
        return 0
    
    # Count zeros (exact matches)
    zeros = (non_nan == 0).sum()
    return float(zeros / len(non_nan))


class PerplexityAnalyzer:
    """Compute perplexity of narratives as hallucination indicator."""
    
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", hf_token=None):
        """Initialize with model for perplexity computation."""
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load language model with 4-bit quantization."""
        if not TRANSFORMERS_AVAILABLE:
            print("WARNING: transformers not available, cannot compute perplexity")
            return
        
        print(f"Loading {self.model_name} model for perplexity computation...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=bnb_config,
                token=self.hf_token
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def compute_perplexity(self, text):
        """Compute perplexity of text. Lower = more coherent."""
        if not self.model or not self.tokenizer:
            return None
        
        try:
            inputs = self.tokenizer(text, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
            
            perplexity = torch.exp(loss).item()
            return float(perplexity)
        
        except Exception as e:
            print(f"Error computing perplexity: {e}")
            return None


class BiasMetricsAnalyzer:
    """Analyze faithfulness and perplexity across demographic batches."""
    
    def __init__(self, narratives_by_batch, shap_df):
        """Initialize analyzer."""
        self.narratives_by_batch = narratives_by_batch
        self.shap_df = shap_df
        self.results = {}
    
    def compute_faithfulness_metrics(self):
        """
        Compute faithfulness metrics across batches.
        Following GitHub repo exactly: rank_accuracy, sign_accuracy, value_accuracy
        """
        print("\n" + "="*80)
        print("FAITHFULNESS ANALYSIS (Rank/Sign/Value Accuracy)")
        print("="*80)
        
        if not LLM_AVAILABLE:
            print("ERROR: llm_tools not available. Cannot compute faithfulness without LLM extraction.")
            return {}
        
        try:
            llm_client = LLMClient()
            extractor = ExtractionModel(llm_client)
        except Exception as e:
            print(f"ERROR: Could not initialize LLM client: {e}")
            return {}
        
        results_by_batch = {}
        
        for batch_name, narratives in self.narratives_by_batch.items():
            print(f"\nBatch: {batch_name}")
            
            rank_accuracies = []
            sign_accuracies = []
            value_accuracies = []
            
            total = len(narratives)
            
            for idx, (instance_idx, narrative) in enumerate(narratives.items()):
                if (idx + 1) % 50 == 0 or (idx + 1) == total:
                    print(f"  Processing narrative {idx + 1}/{total}...")
                
                try:
                    # Get real SHAP values
                    real_shap = self.shap_df[self.shap_df['instance_index'] == instance_idx]
                    if real_shap.empty:
                        continue
                    
                    # Build explanation dataframe
                    row = real_shap.iloc[0]
                    features = [col[5:] for col in self.shap_df.columns if col.startswith('SHAP_')]
                    explanation_data = {
                        'feature_name': features,
                        'SHAP_value': [float(row[f'SHAP_{feat}']) for feat in features],
                    }
                    explanation_df = pd.DataFrame(explanation_data)
                    
                    # Extract features from narrative
                    extracted_dict = extractor.extract_narrative(narrative)
                    
                    if not extracted_dict:
                        continue
                    
                    # Compute differences
                    rank_diff, sign_diff, value_diff, _, _ = ExtractionModel.get_diff(extracted_dict, explanation_df)
                    
                    if rank_diff:
                        # Compute accuracies
                        rank_acc = average_zero(pd.DataFrame([rank_diff]))
                        sign_acc = average_zero(pd.DataFrame([sign_diff]))
                        value_acc = average_zero(pd.DataFrame([value_diff]))
                        
                        rank_accuracies.append(rank_acc)
                        sign_accuracies.append(sign_acc)
                        value_accuracies.append(value_acc)
                
                except Exception as e:
                    print(f"    Error on instance {instance_idx}: {e}")
            
            # Aggregate
            batch_results = {
                "rank_accuracy": float(np.mean(rank_accuracies)) if rank_accuracies else 0,
                "rank_accuracy_std": float(np.std(rank_accuracies)) if rank_accuracies else 0,
                "sign_accuracy": float(np.mean(sign_accuracies)) if sign_accuracies else 0,
                "sign_accuracy_std": float(np.std(sign_accuracies)) if sign_accuracies else 0,
                "value_accuracy": float(np.mean(value_accuracies)) if value_accuracies else 0,
                "value_accuracy_std": float(np.std(value_accuracies)) if value_accuracies else 0,
                "count": len(rank_accuracies),
            }
            
            results_by_batch[batch_name] = batch_results
            
            print(f"  Rank Accuracy: {batch_results['rank_accuracy']:.3f} (±{batch_results['rank_accuracy_std']:.3f})")
            print(f"  Sign Accuracy: {batch_results['sign_accuracy']:.3f} (±{batch_results['sign_accuracy_std']:.3f})")
            print(f"  Value Accuracy: {batch_results['value_accuracy']:.3f} (±{batch_results['value_accuracy_std']:.3f})")
        
        self.results["faithfulness"] = results_by_batch
        return results_by_batch
    
    def compute_perplexity_metrics(self, ppl_models=PPL_MODELS):
        """Compute perplexity metrics across batches."""
        print("\n" + "="*80)
        print("PERPLEXITY ANALYSIS (Hallucination Indicator)")
        print("="*80)
        
        if not TRANSFORMERS_AVAILABLE:
            print("ERROR: transformers not available")
            return {}
        
        results_by_batch = {}
        
        for ppl_model in ppl_models:
            print(f"\nUsing model: {ppl_model}")
            analyzer = PerplexityAnalyzer(ppl_model)
            
            for batch_name, narratives in self.narratives_by_batch.items():
                print(f"  Batch: {batch_name}")
                
                perplexities = []
                total = len(narratives)
                
                for idx, (instance_idx, narrative) in enumerate(narratives.items()):
                    if (idx + 1) % 50 == 0 or (idx + 1) == total:
                        print(f"    Processing narrative {idx + 1}/{total}...")
                    
                    try:
                        ppl = analyzer.compute_perplexity(narrative)
                        if ppl is not None:
                            perplexities.append(ppl)
                    except Exception as e:
                        print(f"    Error: {e}")
                
                # Aggregate
                batch_results = {
                    "avg_perplexity": float(np.mean(perplexities)) if perplexities else 0,
                    "std_perplexity": float(np.std(perplexities)) if perplexities else 0,
                    "count": len(perplexities),
                }
                
                if batch_name not in results_by_batch:
                    results_by_batch[batch_name] = {}
                
                results_by_batch[batch_name][ppl_model] = batch_results
                
                print(f"    Perplexity: {batch_results['avg_perplexity']:.2f} (±{batch_results['std_perplexity']:.2f})")
        
        self.results["perplexity"] = results_by_batch
        return results_by_batch
    
    def run_analysis(self):
        """Run all analyses."""
        print("\n" + "*"*80)
        print("FAITHFULNESS AND HALLUCINATION METRICS")
        print("*"*80)
        print(f"Dataset: {DATASET}")
        print(f"Batches: {list(self.narratives_by_batch.keys())}")
        print(f"Started: {datetime.now().isoformat()}")
        
        self.compute_faithfulness_metrics()
        self.compute_perplexity_metrics(PPL_MODELS)
        
        return self.results
    
    def save_results(self, output_dir):
        """Save analysis results to JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        filepath = Path(output_dir) / f"faithfulness_perplexity_{DATASET}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main execution."""
    print("Loading data...")
    
    # Auto-discover batches if BATCHES is None
    batches_to_use = BATCHES
    if batches_to_use is None:
        print(f"Auto-discovering batches in results directory...")
        batches_to_use = discover_all_batches(DATASET, PROVIDER, MODEL)
        if batches_to_use:
            print(f"Found batches: {batches_to_use}")
        else:
            print("ERROR: No batches found")
            return
    
    # Load SHAP values
    all_shap = load_shap_values(DATASET)
    
    # Load narratives from batches
    narratives_by_batch = {}
    for batch_name in batches_to_use:
        print(f"\nBatch: {batch_name}")
        narratives, instance_indices = load_narratives_from_batch(
            DATASET, batch_name, PROVIDER, MODEL, NUM_INSTANCES
        )
        if narratives:
            narratives_by_batch[batch_name] = narratives
            print(f"  Loaded {len(narratives)} narratives")
    
    if not narratives_by_batch:
        print("ERROR: No narratives loaded!")
        return
    
    # Filter SHAP to only instances with narratives
    all_narr_indices = set()
    for narr_dict in narratives_by_batch.values():
        all_narr_indices.update(narr_dict.keys())
    
    shap_df = all_shap[all_shap['instance_index'].isin(all_narr_indices)]
    
    # Run analysis
    analyzer = BiasMetricsAnalyzer(narratives_by_batch, shap_df)
    results = analyzer.run_analysis()
    
    # Save results
    analyzer.save_results(RESULTS_OUTPUT_DIR)
    
    print("\n" + "*"*80)
    print("ANALYSIS COMPLETE")
    print("*"*80)
    
    # Print summary
    print("\nSUMMARY ACROSS BATCHES:")
    print("="*80)
    
    if "faithfulness" in results:
        print("\nFAITHFULNESS (Rank/Sign/Value Accuracy - 0 to 1):")
        for batch, metrics in results["faithfulness"].items():
            print(f"  {batch}:")
            print(f"    Rank Accuracy: {metrics.get('rank_accuracy', 0):.3f}")
            print(f"    Sign Accuracy: {metrics.get('sign_accuracy', 0):.3f}")
            print(f"    Value Accuracy: {metrics.get('value_accuracy', 0):.3f}")
    
    if "perplexity" in results:
        print("\nPERPLEXITY (Lower = less hallucination):")
        for batch, models_dict in results["perplexity"].items():
            print(f"  {batch}:")
            if isinstance(models_dict, dict):
                for model, metrics in models_dict.items():
                    ppl = metrics.get('avg_perplexity', 0)
                    print(f"    {model}: {ppl:.2f}")


if __name__ == "__main__":
    main()


# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset and batches (None = auto-discover all batches in results directory)
DATASET = "law"
BATCHES = None  # Set None to auto-discover, or list: ["white_male", "black_female"]

# LLM details for narrative loading
PROVIDER = "grok"
MODEL = "grok-4-1-fast-non-reasoning"

# Extraction and perplexity settings
EXTRACTION_LLM = "gpt-4o"  # For feature extraction from narratives
PPL_MODELS = ["meta-llama/Meta-Llama-3-8B"]  # For perplexity computation

# Output directory
RESULTS_OUTPUT_DIR = "results/faithfulness_metrics"

# Number of instances (None = all)
NUM_INSTANCES = None

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Feature names and their expected directions from SHAP (for validation)
FEATURE_INFO = {
    "decile1": {"type": "numeric", "range": (1, 10)},
    "decile3": {"type": "numeric", "range": (1, 10)},
    "fam_inc": {"type": "categorical", "values": ["Low", "Lower-middle", "Middle", "Upper-middle", "High"]},
    "lsat": {"type": "numeric", "range": (95, 180)},
    "ugpa": {"type": "numeric", "range": (0, 4)},
    "fulltime": {"type": "categorical", "values": ["Full-time", "Part-time"]},
}


def discover_all_batches(dataset, provider, model):
    """Auto-discover all available batch directories."""
    base_dir = Path(f"results/narratives/{dataset}/narratives/shap")
    batches = []
    
    if not base_dir.exists():
        print(f"ERROR: Results directory does not exist: {base_dir}")
        return []
    
    # Find all directories matching provider_batchname/model
    for provider_batch_dir in base_dir.glob(f"{provider}_*"):
        model_dir = provider_batch_dir / model
        if model_dir.exists():
            batch_name = provider_batch_dir.name.replace(f"{provider}_", "")
            batches.append(batch_name)
    
    return sorted(batches)


def load_shap_values(dataset, instance_indices=None):
    """Load SHAP values for instances."""
    shap_path = Path(f"datasets_prep/data/{dataset}_dataset/{dataset}_shap.csv")
    shap_df = pd.read_csv(shap_path)
    
    if instance_indices:
        shap_df = shap_df[shap_df['instance_index'].isin(instance_indices)]
    
    return shap_df


def load_narratives_from_batch(dataset, batch_name, provider, model, num_instances=None):
    """Load narratives from a bias injection batch."""
    narratives = {}
    base_dir = Path(f"results/narratives/{dataset}/narratives/shap/{provider}_{batch_name}/{model}")
    
    if not base_dir.exists():
        print(f"ERROR: Batch directory does not exist: {base_dir}")
        return narratives, []
    
    json_files = sorted(base_dir.glob("instance_*.json"))
    
    if num_instances:
        json_files = json_files[:num_instances]
    
    instance_indices = []
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("status") == "success" and data.get("narrative"):
                    instance_idx = data.get("instance_idx")
                    narratives[instance_idx] = data.get("narrative")
                    instance_indices.append(instance_idx)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return narratives, sorted(instance_indices)


class FaithfulnessAnalyzer:
    """Compute faithfulness of narratives against SHAP values."""
    
    def __init__(self, shap_df):
        """Initialize with SHAP values."""
        self.shap_df = shap_df
        self.shap_by_idx = {}
        
        for _, row in shap_df.iterrows():
            idx = int(row['instance_index'])
            shap_vals = {}
            for col in shap_df.columns:
                if col.startswith('SHAP_'):
                    feature = col[5:]  # Remove 'SHAP_' prefix
                    shap_vals[feature] = float(row[col])
            self.shap_by_idx[idx] = shap_vals
    
    def extract_feature_mentions(self, narrative_text):
        """
        Extract mentions of SHAP values and their directions from narrative.
        
        Looks for patterns like:
        - "feature X pushed toward failure/passing"
        - "feature X contributed positively/negatively"
        - "feature X increased/decreased the prediction"
        """
        feature_mentions = defaultdict(lambda: {"directions": [], "values": []})
        
        # Define patterns to match
        patterns = {
            "pushed_toward": r"(\w+)\s+pushed\s+toward\s+(failure|passing|fail|pass)",
            "contributed": r"(\w+)\s+contributed\s+(positively|negatively|toward)",
            "increased_decreased": r"(\w+)\s+(increased|decreased|raised|lowered)",
            "positive_negative": r"(positive|negative|favorable|unfavorable)\s+(\w+)",
        }
        
        for feature in FEATURE_INFO.keys():
            # Look for feature mentions (case-insensitive)
            feature_pattern = re.compile(rf"\b{feature}\b", re.IGNORECASE)
            if feature_pattern.search(narrative_text):
                feature_mentions[feature]["mentioned"] = True
            else:
                feature_mentions[feature]["mentioned"] = False
        
        return feature_mentions
    
    def compute_faithfulness_score(self, instance_idx, narrative_text):
        """
        Compute faithfulness score: how well narrative matches SHAP values.
        
        Score from 0-1 where:
        - 1.0 = perfect alignment with SHAP values
        - Higher is more faithful
        """
        if instance_idx not in self.shap_by_idx:
            return None
        
        shap_vals = self.shap_by_idx[instance_idx]
        
        # Extract feature mentions from narrative
        feature_mentions = self.extract_feature_mentions(narrative_text)
        
        # Compute mentioned features
        mentioned_features = [f for f, info in feature_mentions.items() if info.get("mentioned", False)]
        total_features = len(shap_vals)
        
        # Mention ratio: how many features are mentioned
        mention_ratio = len(mentioned_features) / total_features if total_features > 0 else 0
        
        # Direction alignment: check if top features by magnitude are mentioned
        sorted_features = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = [f for f, _ in sorted_features[:5]]  # Top 5 by magnitude
        
        top_features_mentioned = sum(1 for f in top_features if feature_mentions[f].get("mentioned", False))
        top_feature_ratio = top_features_mentioned / min(5, len(top_features)) if top_features else 0
        
        # Combine metrics: emphasis on mentioning top features
        faithfulness_score = (0.3 * mention_ratio + 0.7 * top_feature_ratio)
        
        return {
            "faithfulness_score": faithfulness_score,
            "mention_ratio": mention_ratio,
            "top_feature_mention_ratio": top_feature_ratio,
            "mentioned_features": mentioned_features,
            "top_features": top_features,
        }


class PerplexityAnalyzer:
    """Compute perplexity of narratives as a hallucination indicator."""
    
    def __init__(self):
        """Initialize GPT-2 model for perplexity computation."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load GPT-2 model."""
        if not TRANSFORMERS_AVAILABLE:
            print("WARNING: transformers not available, cannot compute perplexity")
            return
        
        print("Loading GPT-2 model for perplexity computation...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.model.eval()
    
    def compute_perplexity(self, narrative_text):
        """
        Compute perplexity of narrative text.
        
        Lower perplexity = more coherent/natural
        Higher perplexity = less coherent (potential hallucination indicator)
        """
        if not self.model:
            return None
        
        try:
            # Tokenize
            input_ids = self.tokenizer.encode(narrative_text, return_tensors="pt").to(self.device)
            
            # Compute loss
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
            
            # Perplexity = exp(loss)
            perplexity = torch.exp(loss).item()
            
            # Normalize by narrative length (longer narratives may have higher perplexity)
            num_tokens = input_ids.shape[1]
            normalized_perplexity = perplexity / (num_tokens / 100) if num_tokens > 0 else perplexity
            
            return {
                "perplexity": float(perplexity),
                "normalized_perplexity": float(normalized_perplexity),
                "num_tokens": int(num_tokens),
            }
        
        except Exception as e:
            print(f"Error computing perplexity: {e}")
            return None


class BiasMetricsAnalyzer:
    """Analyze faithfulness and perplexity across demographic batches."""
    
    def __init__(self, narratives_by_batch, shap_df):
        """Initialize analyzer."""
        self.narratives_by_batch = narratives_by_batch
        self.shap_df = shap_df
        self.results = {}
    
    def compute_faithfulness_metrics(self):
        """Compute faithfulness metrics across batches."""
        print("\n" + "="*80)
        print("FAITHFULNESS ANALYSIS")
        print("="*80)
        
        analyzer = FaithfulnessAnalyzer(self.shap_df)
        results_by_batch = {}
        
        for batch_name, narratives in self.narratives_by_batch.items():
            print(f"\nBatch: {batch_name}")
            
            faithfulness_scores = []
            mention_ratios = []
            top_feature_ratios = []
            
            total = len(narratives)
            
            for idx, (instance_idx, narrative) in enumerate(narratives.items()):
                if (idx + 1) % 50 == 0:
                    print(f"  Processing narrative {idx + 1}/{total}...")
                
                try:
                    result = analyzer.compute_faithfulness_score(instance_idx, narrative)
                    if result:
                        faithfulness_scores.append(result["faithfulness_score"])
                        mention_ratios.append(result["mention_ratio"])
                        top_feature_ratios.append(result["top_feature_mention_ratio"])
                except Exception as e:
                    print(f"  Error: {e}")
            
            # Aggregate
            batch_results = {
                "avg_faithfulness": np.mean(faithfulness_scores) if faithfulness_scores else 0,
                "std_faithfulness": np.std(faithfulness_scores) if faithfulness_scores else 0,
                "avg_mention_ratio": np.mean(mention_ratios) if mention_ratios else 0,
                "avg_top_feature_mention": np.mean(top_feature_ratios) if top_feature_ratios else 0,
                "count": len(faithfulness_scores),
            }
            
            results_by_batch[batch_name] = batch_results
            
            print(f"  Average Faithfulness Score: {batch_results['avg_faithfulness']:.3f} (±{batch_results['std_faithfulness']:.3f})")
            print(f"  Feature Mention Ratio: {batch_results['avg_mention_ratio']:.2%}")
            print(f"  Top Feature Mention Ratio: {batch_results['avg_top_feature_mention']:.2%}")
        
        self.results["faithfulness"] = results_by_batch
        return results_by_batch
    
    def compute_perplexity_metrics(self):
        """Compute perplexity metrics across batches."""
        print("\n" + "="*80)
        print("PERPLEXITY ANALYSIS (Hallucination Indicator)")
        print("="*80)
        
        if not TRANSFORMERS_AVAILABLE:
            print("ERROR: transformers not available")
            return {}
        
        analyzer = PerplexityAnalyzer()
        results_by_batch = {}
        
        for batch_name, narratives in self.narratives_by_batch.items():
            print(f"\nBatch: {batch_name}")
            
            perplexities = []
            normalized_perplexities = []
            
            total = len(narratives)
            
            for idx, (instance_idx, narrative) in enumerate(narratives.items()):
                if (idx + 1) % 50 == 0:
                    print(f"  Processing narrative {idx + 1}/{total}...")
                
                try:
                    result = analyzer.compute_perplexity(narrative)
                    if result:
                        perplexities.append(result["perplexity"])
                        normalized_perplexities.append(result["normalized_perplexity"])
                except Exception as e:
                    print(f"  Error: {e}")
            
            # Aggregate
            batch_results = {
                "avg_perplexity": np.mean(perplexities) if perplexities else 0,
                "std_perplexity": np.std(perplexities) if perplexities else 0,
                "avg_normalized_perplexity": np.mean(normalized_perplexities) if normalized_perplexities else 0,
                "count": len(perplexities),
            }
            
            results_by_batch[batch_name] = batch_results
            
            print(f"  Average Perplexity: {batch_results['avg_perplexity']:.2f} (±{batch_results['std_perplexity']:.2f})")
            print(f"  Normalized Perplexity: {batch_results['avg_normalized_perplexity']:.2f}")
            print(f"  Interpretation: {'High hallucination risk' if batch_results['avg_perplexity'] > 30 else 'Low hallucination risk'} (threshold: 30)")
        
        self.results["perplexity"] = results_by_batch
        return results_by_batch
    
    def run_analysis(self):
        """Run all analyses."""
        print("\n" + "*"*80)
        print("FAITHFULNESS AND HALLUCINATION METRICS")
        print("*"*80)
        print(f"Dataset: {DATASET}")
        print(f"Batches: {list(self.narratives_by_batch.keys())}")
        print(f"Started: {datetime.now().isoformat()}")
        
        self.compute_faithfulness_metrics()
        self.compute_perplexity_metrics()
        
        return self.results
    
    def save_results(self, output_dir):
        """Save analysis results to JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        filepath = Path(output_dir) / f"faithfulness_perplexity_{DATASET}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main execution."""
    print("Loading data...")
    
    # Auto-discover batches if BATCHES is None
    batches_to_use = BATCHES
    if batches_to_use is None:
        print(f"Auto-discovering batches in results directory...")
        batches_to_use = discover_all_batches(DATASET, PROVIDER, MODEL)
        if batches_to_use:
            print(f"Found batches: {batches_to_use}")
        else:
            print("ERROR: No batches found")
            return
    
    # Load SHAP values
    all_shap = load_shap_values(DATASET)
    
    # Load narratives from batches
    narratives_by_batch = {}
    for batch_name in batches_to_use:
        print(f"\nBatch: {batch_name}")
        narratives, instance_indices = load_narratives_from_batch(
            DATASET, batch_name, PROVIDER, MODEL, NUM_INSTANCES
        )
        if narratives:
            narratives_by_batch[batch_name] = narratives
            print(f"  Loaded {len(narratives)} narratives")
    
    if not narratives_by_batch:
        print("ERROR: No narratives loaded!")
        return
    
    # Filter SHAP to only instances with narratives
    all_narr_indices = set()
    for narr_dict in narratives_by_batch.values():
        all_narr_indices.update(narr_dict.keys())
    
    shap_df = all_shap[all_shap['instance_index'].isin(all_narr_indices)]
    
    # Run analysis
    analyzer = BiasMetricsAnalyzer(narratives_by_batch, shap_df)
    results = analyzer.run_analysis()
    
    # Save results
    analyzer.save_results(RESULTS_OUTPUT_DIR)
    
    print("\n" + "*"*80)
    print("ANALYSIS COMPLETE")
    print("*"*80)
    
    # Print summary
    print("\nSUMMARY ACROSS BATCHES:")
    print("="*80)
    
    if "faithfulness" in results:
        print("\nFAITHFULNESS (Rank/Sign/Value Accuracy - 0 to 1):")
        for batch, metrics in results["faithfulness"].items():
            print(f"  {batch}:")
            print(f"    Rank Accuracy: {metrics.get('rank_accuracy', 0):.3f}")
            print(f"    Sign Accuracy: {metrics.get('sign_accuracy', 0):.3f}")
    
    if "perplexity" in results:
        print("\nPERPLEXITY (Lower = less hallucination):")
        for batch, models_dict in results["perplexity"].items():
            print(f"  {batch}:")
            if isinstance(models_dict, dict):
                for model, metrics in models_dict.items():
                    ppl = metrics.get('avg_perplexity', 0)
                    print(f"    {model}: {ppl:.2f}")
            else:
                print(f"    {models_dict.get('avg_perplexity', 0):.2f}")


if __name__ == "__main__":
    main()
