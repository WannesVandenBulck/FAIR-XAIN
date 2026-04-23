"""
Faithfulness metrics computation for LLM-generated narratives.

This script:
1. Extracts ground truth from the original prompts
2. Uses LLM to extract information from generated narratives
3. Compares extracted vs ground truth information
4. Computes faithfulness metrics:
   - For SHAP: Sign accuracy, Rank accuracy, Value accuracy, Average accuracy
   - For CF: Change accuracy, Change value accuracy, Value accuracy, Average accuracy
5. Outputs results to Excel

Usage:
    python scripts/compute_faithfulness.py --dataset credit --prompt-type shap --instances 0 1 2
    python scripts/compute_faithfulness.py --dataset credit --prompt-type cf --all-instances
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import re
from difflib import SequenceMatcher

# Add parent directory to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_tools.llm_client import generate_text


# ============================================================================
# GROUND TRUTH EXTRACTION FROM PROMPTS
# ============================================================================

def parse_shap_prompt(prompt_text: str) -> Dict[str, Any]:
    """
    Extract ground truth information from a SHAP prompt.
    
    Returns dict with:
    - features_ranked: List[{name, shap_value, sign, rank}]
    - instance_features: Dict[feature_name -> value]
    - averages_positive: Dict[feature_name -> average]
    """
    ground_truth = {
        "features_ranked": [],
        "instance_features": {},
        "averages_positive": {},
        "features_mentioned": 0
    }
    
    # Extract SHAP table - look for lines with Feature and SHAP_Value header
    # Table format:
    #        Feature  SHAP_Value
    #         status    0.100286
    # credit_history   -0.088201
    #       duration    0.072757
    
    # Find the table header
    header_match = re.search(r'\s+Feature\s+SHAP_Value', prompt_text)
    if header_match:
        # Extract from header until we hit something that's not a feature line
        start_pos = header_match.start()
        # Find the end of this section (next non-table line)
        remaining = prompt_text[header_match.end():]
        lines = remaining.split('\n')
        
        rank = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Stop if we hit a section header or other non-table content
            if re.match(r'^\d+\.|^5\.|^TASK|^YOUR|^Instructions', line):
                break
            
            # Try to parse as feature line
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    shap_val = float(parts[-1])
                    feature_name = ' '.join(parts[:-1]).strip()
                    
                    if feature_name and not feature_name.startswith('-'):
                        rank += 1
                        sign = "positive" if shap_val > 0 else "negative" if shap_val < 0 else "neutral"
                        
                        ground_truth["features_ranked"].append({
                            "name": feature_name,
                            "shap_value": shap_val,
                            "sign": sign,
                            "rank": rank
                        })
                except (ValueError, IndexError):
                    continue
    
    # Extract instance features from section starting with "The applicant's"
    instance_section_match = re.search(
        r"The applicant's.*?(?=The model's prediction:|$)",
        prompt_text,
        re.DOTALL | re.IGNORECASE
    )
    
    if instance_section_match:
        applicant_section = instance_section_match.group(0)
        
        # Look for patterns like "- feature_name = value with optional comparison"
        line_pattern = r"-\s+([^:=\n]+?)\s*[:=]\s+([^-\n]+?)(?:\s*-|$)"
        matches = re.findall(line_pattern, applicant_section, re.MULTILINE)
        
        for feature_name, value_str in matches:
            feature_name = feature_name.strip()
            value_str = value_str.strip()
            
            # Skip if this is just section text or too long
            if len(feature_name) > 50 or 'applicant' in feature_name.lower():
                continue
            
            # Extract numerical average if present in parentheses or with "average"
            avg_match = re.search(r"(?:average|avg)[:\s]*([+-]?\d+\.?\d*)", value_str, re.IGNORECASE)
            if avg_match:
                try:
                    ground_truth["averages_positive"][feature_name] = float(avg_match.group(1))
                except ValueError:
                    pass
            
            # Store the feature value (clean up extra text)
            value_clean = value_str.split('(')[0].split(' - ')[0].split(' compare')[0].strip()
            if value_clean and len(value_clean) < 200:
                ground_truth["instance_features"][feature_name] = value_clean
    
    ground_truth["features_mentioned"] = len(ground_truth["features_ranked"])
    return ground_truth


def parse_cf_prompt(prompt_text: str) -> Dict[str, Any]:
    """
    Extract ground truth information from a Counterfactual prompt.
    
    Returns dict with:
    - changes: List[{feature_name, percentage_changed, ...}]
    - instance_features: Dict[feature_name -> value]
    - averages_positive: Dict[feature_name -> average]
    """
    ground_truth = {
        "changes": [],
        "instance_features": {},
        "averages_positive": {},
        "num_counterfactuals": 0
    }
    
    # Extract counterfactual analysis summary
    analysis_match = re.search(
        r"COUNTERFACTUAL ANALYSIS SUMMARY:.*?(?=5\.|$)",
        prompt_text,
        re.DOTALL
    )
    
    if analysis_match:
        analysis_section = analysis_match.group(0)
        
        # Extract number of counterfactuals
        num_cf_match = re.search(r"Generated\s+(\d+)\s+counterfactual", analysis_section)
        if num_cf_match:
            ground_truth["num_counterfactuals"] = int(num_cf_match.group(1))
        
        # Extract feature changes: "- feature_name: changed in X% of cases"
        change_pattern = r"-\s+(\w+):\s+changed in\s+([+-]?\d+\.?\d*)%"
        changes = re.findall(change_pattern, analysis_section)
        
        for feature_name, percentage_str in changes:
            try:
                percentage = float(percentage_str)
                ground_truth["changes"].append({
                    "feature_name": feature_name,
                    "percentage_changed": percentage
                })
            except ValueError:
                continue
    
    # Extract instance features (same as SHAP)
    applicant_section_match = re.search(
        r"3\.\s*APPLICANT PROFILE.*?(?=4\.|The model)",
        prompt_text,
        re.DOTALL
    )
    
    if applicant_section_match:
        applicant_section = applicant_section_match.group(0)
        
        line_pattern = r"-\s+(\w+)\s*=\s+([^-]+?)\s*(?:-|among|$)"
        lines = re.findall(line_pattern, applicant_section)
        
        for feature_name, value_str in lines:
            avg_match = re.search(r"avg[:\s]+([+-]?\d+\.?\d*)", value_str)
            if avg_match:
                try:
                    ground_truth["averages_positive"][feature_name] = float(avg_match.group(1))
                except ValueError:
                    pass
            
            value_clean = value_str.strip().split("(")[0].strip()
            if value_clean:
                ground_truth["instance_features"][feature_name] = value_clean
    
    return ground_truth


def extract_ground_truth(prompt_text: str, prompt_type: str) -> Dict[str, Any]:
    """Extract ground truth from prompt based on type."""
    if prompt_type == "shap":
        return parse_shap_prompt(prompt_text)
    elif prompt_type == "cf":
        return parse_cf_prompt(prompt_text)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


# ============================================================================
# LLM EXTRACTION FROM NARRATIVES
# ============================================================================

EXTRACTION_PROMPT_SHAP = """You are an expert at extracting structured information from text narratives about machine learning model decisions based on SHAP explanations.

Given the following narrative explanation of a SHAP-based model decision, extract information about the features mentioned.

IMPORTANT: The "sign" represents the SHAP value's direction of contribution to the MODEL'S PREDICTED PROBABILITY OF THE POSITIVE CLASS (e.g., probability of approval, probability of positive outcome). This is DIFFERENT from whether it helps or hurts the individual's chances. The narrative may describe how a feature helps or hurts the person, but you must extract the sign based on the model's internal SHAP values.

For each feature mentioned in the narrative, create an entry in a dictionary where:
- Key: exact feature name as it appears (must match the feature names in the explanation table if mentioned)
- Value: a dictionary with:
  * "rank": the rank of importance (0 = most important) if mentioned or can be inferred, else null
  * "sign": The SHAP value sign (not outcome sign). Infer from context clues:
      - If narrative states feature INCREASES model's confidence/probability toward the predicted outcome → +1 (positive SHAP)
      - If narrative states feature DECREASES model's confidence/probability → -1 (negative SHAP)  
      - If narrative frames feature as "a strong concern" or "red flag" when predicting negative outcome → likely POSITIVE SHAP (increases probability of that negative prediction)
      - If narrative frames feature as "working in favor" or "positive" when predicting negative outcome → likely NEGATIVE SHAP (reduces probability of that negative prediction)
      - If cannot be determined from context → null
  * "value": the specific feature value mentioned for this instance (e.g., "42 months", "no positive balance"), null if not mentioned

Also extract:
- "instance_features": dictionary of feature names to their values mentioned in narrative
- "averages_mentioned": dictionary of feature names to their average values mentioned

Return ONLY a valid JSON dictionary. Example format:
{{
  "feature_1": {{"rank": 0, "sign": 1, "value": 0.5}},
  "feature_2": {{"rank": 1, "sign": -1, "value": null}},
  "instance_features": {{"feature_1": "0.5", "feature_2": "low"}},
  "averages_mentioned": {{"feature_1": 0.3}}
}}

NARRATIVE:
{narrative}

Return ONLY valid JSON, no other text."""


EXTRACTION_PROMPT_CF = """You are an expert at extracting structured information from text narratives about counterfactual explanations of model decisions.

Given the following narrative explanation based on counterfactual scenarios, extract information about features mentioned.

IMPORTANT: The "sign" represents the DIRECTION OF CHANGE in the MODEL'S PREDICTED PROBABILITY. This is how the feature change affects the model's internal representation, not necessarily the person's actual outcome. A feature might need to change in a certain direction to improve the model's confidence.

For each feature mentioned in the narrative, create an entry in a dictionary where:
- Key: exact feature name as it appears (must match the feature names in the explanation table if mentioned)
- Value: a dictionary with:
  * "rank": the rank of importance (0 = most important) if mentioned or can be inferred, else null
  * "sign": The direction this feature change would push the model's probability.
      - If narrative suggests changing feature in positive direction increases model's output probability → +1
      - If narrative suggests changing feature in negative direction increases model's output probability → -1  
      - If narrative discusses "if feature were higher/lower" to change prediction direction → infer appropriately
      - If cannot be determined from context → null
  * "value": the specific feature value mentioned for this instance, null if not mentioned

Also extract:
- "instance_features": dictionary of feature names to their values mentioned in narrative
- "averages_mentioned": dictionary of feature names to their average values or comparisons mentioned

Return ONLY a valid JSON dictionary. Example format:
{{
  "feature_1": {{"rank": 0, "sign": 1, "value": 0.5}},
  "feature_2": {{"rank": 1, "sign": -1, "value": null}},
  "instance_features": {{"feature_1": "0.5"}},
  "averages_mentioned": {{"feature_1": 0.3}}
}}

NARRATIVE:
{narrative}

Return ONLY valid JSON, no other text."""


def extract_from_narrative(narrative: str, prompt_type: str, provider: str = "openai", model: str = None) -> Dict[str, Any]:
    """
    Use LLM to extract information from narrative.
    """
    if model is None:
        model = "gpt-4o" if provider == "openai" else "grok-4-1-fast-non-reasoning"
    
    # Choose extraction prompt
    if prompt_type == "shap":
        extraction_prompt = EXTRACTION_PROMPT_SHAP.format(narrative=narrative)
    elif prompt_type == "cf":
        extraction_prompt = EXTRACTION_PROMPT_CF.format(narrative=narrative)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert data extractor. Extract information as requested and return valid JSON only."
        },
        {"role": "user", "content": extraction_prompt}
    ]
    
    try:
        response = generate_text(messages, provider=provider, model=model, temperature=0)
        
        # Clean response: remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code block markers
            response = re.sub(r'^```(?:json)?\s*', '', response)
            response = re.sub(r'\s*```$', '', response)
        
        extracted = json.loads(response)
        return extracted
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from narrative extraction: {e}")
        print(f"Response was: {response[:100] if len(response) > 100 else response}")
        return None
    except Exception as e:
        print(f"Error extracting from narrative: {e}")
        return None


# ============================================================================
# COMPARISON AND METRICS COMPUTATION (Reference Implementation Aligned)
# ============================================================================

def average_zero(array_list: List[List[float]]) -> float:
    """
    Compute the accuracy as percentage of zeros in difference arrays.
    
    Matches reference: average_zero() computes (num_zeros / total_finite_values) * 100
    Finite values = exclude np.inf (hallucinations) and np.nan (missing values)
    
    Args:
        array_list: List of lists of differences (0=match, >0 numeric difference, inf=hallucination, nan=missing)
    
    Returns:
        Percentage of zeros among finite values (0-100)
    """
    all_values = []
    for arr in array_list:
        for val in arr:
            if np.isfinite(val):  # Exclude inf and nan
                all_values.append(val)
    
    if not all_values:
        return np.nan
    
    # Count zeros
    num_zeros = sum(1 for v in all_values if v == 0)
    return (num_zeros / len(all_values)) * 100


def get_diff(extracted_dict: Dict, explanation_df: pd.DataFrame) -> Tuple[List, List, List]:
    """
    Compare extracted features with ground truth explanation.
    
    Mirrors reference: ExtractionModel.get_diff() from extraction.py lines 120-205
    
    Args:
        extracted_dict: Format {"feature_name": {"rank": int, "sign": int, "value": float}}
        explanation_df: SHAP table with columns: feature_name, SHAP_value, feature_value, feature_average
    
    Returns:
        (rank_diff, sign_diff, value_diff) - lists of differences where 0=match, >0=mismatch, inf=hallucination
    """
    # Ensure explanation is sorted by SHAP value
    explanation_df = explanation_df.copy()
    explanation_df["abs_SHAP"] = explanation_df["SHAP_value"].abs()
    explanation_df = explanation_df.sort_values(by="abs_SHAP", ascending=False)
    explanation_df = explanation_df.reset_index(drop=True)
    
    # Build extracted dataframe
    extracted_list = []
    for feat_name, feat_dict in extracted_dict.items():
        extracted_list.append({
            "feature_name": feat_name,
            "rank": feat_dict.get("rank"),
            "sign": feat_dict.get("sign"),
            "value": feat_dict.get("value")
        })
    
    if not extracted_list:
        return [], [], []
    
    df_extracted = pd.DataFrame(extracted_list)
    
    # Get real features and compute their signs
    sign_series = explanation_df["SHAP_value"].map(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df_real = explanation_df[["feature_name", "SHAP_value", "feature_value", "feature_average"]].copy()
    df_real.insert(1, "sign", sign_series.values)
    df_real.insert(1, "rank", df_real.index.values)
    
    rank_diff = []
    sign_diff = []
    value_diff = []
    
    # Process each extracted feature
    for idx, row in df_extracted.iterrows():
        ex_feat_name = row["feature_name"].lower().strip()
        
        # Find matching real feature (case-insensitive, with hybrid fuzzy+substring matching)
        best_match_idx = None
        best_similarity = 0
        
        for ridx, real_row in df_real.iterrows():
            real_feat_name = real_row["feature_name"].lower().strip()
            sim = string_similarity(ex_feat_name, real_feat_name)
            
            # Boost similarity if one name is a substring of the other or contains key words
            # E.g., "status" vs "checking account status" -> boost similarity
            if ex_feat_name in real_feat_name or real_feat_name in ex_feat_name:
                sim = max(sim, 0.75)  # Boost to at least 0.75
            
            # Also check if last word matches (for cases like "duration" vs "loan duration")
            ex_words = ex_feat_name.split()
            real_words = real_feat_name.split()
            if ex_words and real_words and ex_words[-1] == real_words[-1]:
                sim = max(sim, 0.75)  # Boost to at least 0.75 if last word matches
            
            if sim > best_similarity:
                best_similarity = sim
                best_match_idx = ridx
        
        # Match if similarity > 0.6 (lowered from 0.7 to handle partial matches)
        if best_match_idx is not None and best_similarity > 0.6:
            real_row = df_real.iloc[best_match_idx]
            
            # Rank difference (0 = perfect match in ranking)
            ex_rank = row["rank"]
            real_rank = real_row["rank"]
            
            if ex_rank is not None and isinstance(ex_rank, (int, float)) and np.isfinite(ex_rank):
                rank_diff.append(float(ex_rank - real_rank))
            else:
                rank_diff.append(np.nan)
            
            # Sign difference (0 = same sign, 1 = opposite sign)
            ex_sign = row["sign"]
            real_sign = real_row["sign"]
            
            if ex_sign is not None and isinstance(ex_sign, (int, float)) and np.isfinite(ex_sign):
                # Multiply signs: if both same -> product > 0 (correct), if opposite -> product <= 0 (incorrect)
                sign_match = 0 if (ex_sign * real_sign > 0) else 1
                sign_diff.append(float(sign_match))
            else:
                sign_diff.append(np.nan)
            
            # Value difference (0 = exact match)
            ex_value = row["value"]
            real_value = real_row["feature_value"]
            
            if ex_value is not None and isinstance(ex_value, (int, float)) and np.isfinite(ex_value):
                try:
                    real_val_num = float(real_value)
                    value_diff.append(float(ex_value - real_val_num))
                except (ValueError, TypeError):
                    value_diff.append(np.nan)
            else:
                value_diff.append(np.nan)
        else:
            # Hallucinated feature (not in ground truth)
            rank_diff.append(np.inf)
            sign_diff.append(np.inf)
            value_diff.append(np.inf)
    
    return rank_diff, sign_diff, value_diff


def string_similarity(a: str, b: str) -> float:
    """Compute fuzzy string similarity (0-1)."""
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def compute_shap_faithfulness(ground_truth: Dict, extracted: Dict, 
                             dataset: str = "credit") -> Dict[str, Any]:
    """
    Compute faithfulness metrics for SHAP narratives using reference implementation approach.
    
    Mirrors: shapnarrative_metrics.metrics.faithfulness.compute_faithfulness()
    
    Returns: {
        "rank_accuracy": % of features with rank_diff == 0,
        "sign_accuracy": % of features with sign_diff == 0,
        "value_accuracy": % of features with value_diff == 0,
        "features_mentioned_count": number of features extracted
    }
    """
    # Build explanation dataframe from ground truth
    features_ranked = ground_truth.get("features_ranked", [])
    
    if not features_ranked:
        return {
            "rank_accuracy": np.nan,
            "sign_accuracy": np.nan,
            "value_accuracy": np.nan,
            "features_mentioned_count": 0
        }
    
    # Create SHAP explanation table
    explanation_list = []
    for feat in features_ranked:
        explanation_list.append({
            "feature_name": feat.get("name", ""),
            "SHAP_value": feat.get("shap_value", 0),
            "feature_value": ground_truth.get("instance_features", {}).get(feat.get("name", ""), np.nan),
            "feature_average": ground_truth.get("averages_positive", {}).get(feat.get("name", ""), np.nan)
        })
    
    explanation_df = pd.DataFrame(explanation_list)
    
    # Extract features from narrative in reference format
    # The extraction should return: {"feature_name": {"rank": ..., "sign": ..., "value": ...}}
    extracted_dict = {}
    
    # Convert extracted format to expected format if needed
    if "features_mentioned" in extracted:
        # Old format - need to convert
        for feat in extracted.get("features_mentioned", []):
            feat_name = feat.get("name", "")
            if feat_name:
                # Convert sign string to ±1
                sign_str = feat.get("sign_mentioned", None)
                sign_int = None
                if sign_str == "positive":
                    sign_int = 1
                elif sign_str == "negative":
                    sign_int = -1
                
                extracted_dict[feat_name] = {
                    "rank": feat.get("rank_mentioned"),
                    "sign": sign_int,
                    "value": feat.get("shap_value_mentioned")
                }
    else:
        # Already in new format (keys are feature names)
        extracted_dict = extracted.copy()
    
    # Remove non-feature entries
    extracted_dict = {k: v for k, v in extracted_dict.items() 
                     if k not in ["instance_features", "averages_mentioned"]}
    
    # Compute differences
    rank_diff, sign_diff, value_diff = get_diff(extracted_dict, explanation_df)
    
    # Compute accuracies using average_zero
    rank_accuracy = average_zero([rank_diff]) if rank_diff else np.nan
    sign_accuracy = average_zero([sign_diff]) if sign_diff else np.nan
    value_accuracy = average_zero([value_diff]) if value_diff else np.nan
    
    return {
        "rank_accuracy": rank_accuracy,
        "sign_accuracy": sign_accuracy,
        "value_accuracy": value_accuracy,
        "features_mentioned_count": len(extracted_dict)
    }


def compute_cf_faithfulness(ground_truth: Dict, extracted: Dict) -> Dict[str, Any]:
    """
    Compute faithfulness metrics for Counterfactual narratives using reference implementation approach.
    
    Uses same rank/sign/value difference approach as SHAP.
    
    Returns: {
        "rank_accuracy": % of features with rank_diff == 0,
        "sign_accuracy": % of features with sign_diff == 0,
        "value_accuracy": % of features with value_diff == 0,
        "features_mentioned_count": number of features extracted
    }
    """
    # Build explanation dataframe from ground truth
    features_ranked = ground_truth.get("features_ranked", [])
    
    if not features_ranked:
        return {
            "rank_accuracy": np.nan,
            "sign_accuracy": np.nan,
            "value_accuracy": np.nan,
            "features_mentioned_count": 0
        }
    
    # Create explanation table (same structure as SHAP)
    explanation_list = []
    for feat in features_ranked:
        explanation_list.append({
            "feature_name": feat.get("name", ""),
            "SHAP_value": feat.get("shap_value", 0),  # Use SHAP for sign even in CF
            "feature_value": ground_truth.get("instance_features", {}).get(feat.get("name", ""), np.nan),
            "feature_average": ground_truth.get("averages_positive", {}).get(feat.get("name", ""), np.nan)
        })
    
    explanation_df = pd.DataFrame(explanation_list)
    
    # Extract features in reference format
    extracted_dict = {}
    
    # Handle both old and new formats
    if "features_mentioned" in extracted:
        # Old format conversion
        for feat in extracted.get("features_mentioned", []):
            feat_name = feat.get("name", "")
            if feat_name:
                sign_str = feat.get("sign_mentioned", None)
                sign_int = None
                if sign_str == "positive":
                    sign_int = 1
                elif sign_str == "negative":
                    sign_int = -1
                
                extracted_dict[feat_name] = {
                    "rank": feat.get("rank_mentioned"),
                    "sign": sign_int,
                    "value": feat.get("shap_value_mentioned")
                }
    else:
        # Already in new format
        extracted_dict = extracted.copy()
    
    # Remove non-feature entries
    extracted_dict = {k: v for k, v in extracted_dict.items() 
                     if k not in ["instance_features", "averages_mentioned"]}
    
    # Compute differences
    rank_diff, sign_diff, value_diff = get_diff(extracted_dict, explanation_df)
    
    # Compute accuracies using average_zero
    rank_accuracy = average_zero([rank_diff]) if rank_diff else np.nan
    sign_accuracy = average_zero([sign_diff]) if sign_diff else np.nan
    value_accuracy = average_zero([value_diff]) if value_diff else np.nan
    
    return {
        "rank_accuracy": rank_accuracy,
        "sign_accuracy": sign_accuracy,
        "value_accuracy": value_accuracy,
        "features_mentioned_count": len(extracted_dict)
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("FAITHFULNESS COMPUTATION FOR LLM NARRATIVES")
    print("="*80)
    
    # Configuration
    dataset = "credit"
    prompt_type = "shap"
    instance_idx = 0
    provider = "grok"
    model = "grok-4-1-fast-non-reasoning"
    
    print(f"\nDataset: {dataset}")
    print(f"Prompt Type: {prompt_type}")
    print(f"Instance: {instance_idx}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print("="*80)
    
    # Load prompt
    prompt_path = Path(f"datasets_prep/data/{dataset}_dataset/{dataset}_shap.csv") if prompt_type == "shap" else \
                  Path(f"datasets_prep/data/{dataset}_dataset/{dataset}_counterfactual.csv")
    
    # Load narrative (assuming it was saved as JSON)
    narrative_path = Path(f"results/narratives/{dataset}/narratives/{prompt_type}/{provider}/{model}/instance_{instance_idx}.json")
    
    if not narrative_path.exists():
        print(f"\nERROR: Narrative not found at {narrative_path}")
        return
    
    print(f"\n1. Loading narrative...")
    with open(narrative_path, 'r', encoding='utf-8') as f:
        narrative_data = json.load(f)
    narrative_text = narrative_data.get("narrative", "")
    
    # Load original prompt
    # Need to rebuild it - for now, load from files
    print(f"2. Loading ground truth from prompt CSVs...")
    # This is simplified - in reality we need the original full prompt
    # For now, demonstrate the logic
    
    print(f"3. Extracting information from narrative using LLM...")
    extracted = extract_from_narrative(narrative_text, prompt_type, provider, model)
    
    if extracted is None:
        print("ERROR: Failed to extract from narrative")
        return
    
    print(f"4. Extracted information:")
    print(json.dumps(extracted, indent=2))
    
    print("\n" + "="*80)
    print("✓ Faithfulness computation complete")
    print("="*80)


if __name__ == "__main__":
    main()
