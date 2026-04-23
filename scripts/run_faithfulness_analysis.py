"""
Faithfulness analysis orchestration and Excel reporting.

This script:
1. Loads generated narratives and their original prompts
2. Computes ground truth from prompts
3. Extracts information from narratives using LLM
4. Computes faithfulness metrics
5. Generates Excel report

Usage:
    python scripts/run_faithfulness_analysis.py --dataset credit --prompt-type shap --provider grok --model grok-4-1-fast-non-reasoning
"""

import sys
import json
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import re

# Add parent directory to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_faithfulness import (
    extract_ground_truth, 
    extract_from_narrative,
    compute_shap_faithfulness,
    compute_cf_faithfulness
)
from llm_tools.prompts import prompt_credit, prompt_law


# ============================================================================
# CONFIGURATION - EDIT THIS SECTION TO CUSTOMIZE ANALYSIS
# ============================================================================

# Which narratives to analyze
DATASET = "credit"  # Options: "credit", "law", "saudi", "student"
PROMPT_TYPE = "shap"  # Options: "shap", "cf"

# Which instances to analyze
USE_ALL_INSTANCES = True  # Set to True to analyze all available instances
INSTANCE_INDICES = None  # Put instances here if USE_ALL_INSTANCES=False

# Bias batch (if using bias_injection_experiment)
# Leave empty ("") for narratives without bias batch, or specify batch name like "female_married_age35"
BIAS_BATCH = "male_single_age25"  # Examples: "", "female_married_age35", "female_single_age25", etc.
                 # If empty, looks for narratives at: narratives/{provider}/{model}/
                 # If specified, looks for: narratives/{provider}_{BIAS_BATCH}/{model}/

# Narrative generation settings (which narratives were generated)
NARRATIVE_PROVIDER = "grok"  # Provider that generated the narratives
NARRATIVE_MODEL = "grok-4-1-fast-non-reasoning"  # Model that generated the narratives

# Extraction settings (which LLM will extract information from narratives)
EXTRACTION_PROVIDER = "openai"  # Provider for extraction: "openai", "grok", "anthropic", etc.
EXTRACTION_MODEL = "gpt-4o"  # Model for extraction

# Output
OUTPUT_PATH = "results/faithfulness_report.xlsx"  # Where to save Excel report

# ============================================================================
# END CONFIGURATION - DON'T EDIT BELOW THIS LINE (unless you know what you're doing)
# ============================================================================


def rebuild_prompt_from_narrative(dataset: str, instance_idx: int, prompt_type: str, 
                                   personal_status_sex_override=None, age_override=None,
                                   gender_override=None, race_override=None) -> str:
    """
    Rebuild the original prompt that was used to generate the narrative.
    This ensures we have the exact ground truth that was given to the narrative-generating LLM.
    """
    if dataset == "credit":
        if prompt_type == "shap":
            return prompt_credit.build_shap_prompt(
                instance_idx,
                personal_status_sex_override=personal_status_sex_override,
                age_override=age_override
            )
        elif prompt_type == "cf":
            return prompt_credit.build_cf_prompt(
                instance_idx,
                personal_status_sex_override=personal_status_sex_override,
                age_override=age_override
            )
    elif dataset == "law":
        if prompt_type == "shap":
            return prompt_law.build_shap_prompt(
                instance_idx,
                gender_override=gender_override,
                race_override=race_override
            )
        elif prompt_type == "cf":
            return prompt_law.build_cf_prompt(
                instance_idx,
                gender_override=gender_override,
                race_override=race_override
            )
    
    raise ValueError(f"Unknown dataset or prompt type: {dataset}, {prompt_type}")


def load_narrative(dataset: str, instance_idx: int, prompt_type: str, 
                  provider: str, model: str, bias_batch: str = "") -> Dict:
    """Load a generated narrative JSON file."""
    # Construct path based on whether bias_batch is specified
    if bias_batch:
        provider_dir = f"{provider}_{bias_batch}"
    else:
        provider_dir = provider
    
    narrative_path = Path(f"results/narratives/{dataset}/narratives/{prompt_type}/{provider_dir}/{model}/instance_{instance_idx}.json")
    
    if not narrative_path.exists():
        return None
    
    with open(narrative_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_overrides_from_narrative_metadata(narrative_data: Dict) -> Dict:
    """Extract override information from narrative JSON metadata."""
    return {
        "personal_status_sex_override": narrative_data.get("personal_status_sex_override"),
        "age_override": narrative_data.get("age_override"),
        "gender_override": narrative_data.get("gender_override"),
        "race_override": narrative_data.get("race_override"),
    }


def run_faithfulness_analysis(dataset: str, prompt_type: str, provider: str, model: str,
                             instances: List[int], extraction_provider: str = "openai",
                             extraction_model: str = None, bias_batch: str = "") -> List[Dict]:
    """
    Run faithfulness analysis for a set of narratives.
    
    Returns list of dicts with faithfulness metrics for each instance.
    """
    if extraction_model is None:
        extraction_model = "gpt-4o" if extraction_provider == "openai" else "grok-4-1-fast-non-reasoning"
    
    results = []
    
    for instance_idx in instances:
        batch_str = f"_{bias_batch}" if bias_batch else ""
        print(f"\n[{dataset}/{prompt_type}/{provider}{batch_str}/{model}] Instance {instance_idx}...", end=" ", flush=True)
        
        try:
            # Load narrative
            narrative_data = load_narrative(dataset, instance_idx, prompt_type, provider, model, bias_batch)
            if narrative_data is None:
                print("SKIP (not found)")
                continue
            
            # Get overrides from narrative metadata
            overrides = extract_overrides_from_narrative_metadata(narrative_data)
            
            # Rebuild original prompt to get ground truth
            try:
                original_prompt = rebuild_prompt_from_narrative(
                    dataset, instance_idx, prompt_type,
                    personal_status_sex_override=overrides["personal_status_sex_override"],
                    age_override=overrides["age_override"],
                    gender_override=overrides["gender_override"],
                    race_override=overrides["race_override"]
                )
            except Exception as e:
                print(f"ERROR (rebuild prompt: {str(e)[:80]})")
                continue
            
            # Extract ground truth from prompt
            ground_truth = extract_ground_truth(original_prompt, prompt_type)
            
            # Extract from narrative using LLM
            narrative_text = narrative_data.get("narrative", "")
            if not narrative_text:
                print("SKIP (empty narrative)")
                continue
            
            extracted = extract_from_narrative(narrative_text, prompt_type, extraction_provider, extraction_model)
            if extracted is None:
                print("ERROR (extraction failed)")
                continue
            
            # Compute faithfulness metrics
            if prompt_type == "shap":
                metrics = compute_shap_faithfulness(ground_truth, extracted)
            elif prompt_type == "cf":
                metrics = compute_cf_faithfulness(ground_truth, extracted)
            else:
                print("ERROR (unknown prompt type)")
                continue
            
            # Store result
            result = {
                "dataset": dataset,
                "prompt_type": prompt_type,
                "instance_idx": instance_idx,
                "bias_batch": bias_batch if bias_batch else "none",
                "narrative_provider": provider,
                "narrative_model": model,
                "extraction_provider": extraction_provider,
                "extraction_model": extraction_model,
                "timestamp": datetime.now().isoformat(),
                **metrics
            }
            
            results.append(result)
            print(f"OK")
            
        except KeyError as e:
            print(f"ERROR (missing key: {e})")
            import traceback
            traceback.print_exc()
            continue
        except Exception as e:
            print(f"ERROR ({str(e)[:80]})")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def create_excel_report(results: List[Dict], output_path: str = "faithfulness_report.xlsx"):
    """
    Create Excel report from faithfulness results.
    """
    if not results:
        print("No results to report")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    column_order = [
        "dataset", "prompt_type", "instance_idx",
        "narrative_provider", "narrative_model",
        "extraction_provider", "extraction_model",
        "features_mentioned_count",
        "rank_accuracy", "sign_accuracy", "value_accuracy", 
        "timestamp"
    ]
    
    # Keep only columns that exist
    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: All detailed results
        df.to_excel(writer, sheet_name="Detailed Results", index=False)
        
        # Sheet 2: Summary by model and dataset
        summary_cols = [c for c in df.columns if "accuracy" in c or "faithfulness" in c]
        summary = df.groupby(["dataset", "prompt_type", "narrative_provider", "narrative_model"])[summary_cols].agg(['mean', 'std', 'min', 'max', 'count'])
        summary.to_excel(writer, sheet_name="Summary Statistics")
        
        # Format columns
        worksheet = writer.sheets["Detailed Results"]
        for idx, col in enumerate(df.columns, 1):
            worksheet.column_dimensions[chr(64 + idx)].width = 20
    
    print(f"\n✓ Report saved to: {output_path}")
    print(f"  Total instances analyzed: {len(df)}")
    print(f"  Datasets: {df['dataset'].unique().tolist()}")
    print(f"  Prompt types: {df['prompt_type'].unique().tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Compute faithfulness metrics for narratives")
    parser.add_argument("--dataset", default=DATASET, choices=["credit", "law", "saudi", "student"])
    parser.add_argument("--prompt-type", default=PROMPT_TYPE, choices=["shap", "cf"])
    parser.add_argument("--provider", default=NARRATIVE_PROVIDER, help="Provider that generated narratives")
    parser.add_argument("--model", default=NARRATIVE_MODEL, help="Model that generated narratives")
    parser.add_argument("--instances", type=int, nargs="+", default=INSTANCE_INDICES if not USE_ALL_INSTANCES else None)
    parser.add_argument("--all-instances", action="store_true", default=USE_ALL_INSTANCES, help="Analyze all available instances")
    parser.add_argument("--extraction-provider", default=EXTRACTION_PROVIDER, help="Provider for extraction")
    parser.add_argument("--extraction-model", default=EXTRACTION_MODEL, help="Model for extraction")
    parser.add_argument("--bias-batch", default=BIAS_BATCH, help="Bias batch name (for narratives from bias_injection_experiment)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    
    args = parser.parse_args()
    
    # Determine instances to analyze
    if args.all_instances:
        # Load all available instance indices from CSV
        csv_path = f"datasets_prep/data/{args.dataset}_dataset/{args.dataset}_{args.prompt_type}.csv"
        try:
            df = pd.read_csv(csv_path)
            instances = sorted(df['instance_index'].unique().tolist())
        except Exception as e:
            print(f"ERROR: Could not load instances from {csv_path}: {e}")
            return
    else:
        instances = args.instances if args.instances else INSTANCE_INDICES
    
    print("\n" + "="*80)
    print("FAITHFULNESS ANALYSIS FOR LLM NARRATIVES")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Prompt Type: {args.prompt_type}")
    print(f"\nNarrative Generation:")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    if args.bias_batch:
        print(f"  Bias Batch: {args.bias_batch}")
    print(f"\nExtraction & Analysis:")
    print(f"  Provider: {args.extraction_provider}")
    print(f"  Model: {args.extraction_model}")
    print(f"\nInstances:")
    if args.all_instances:
        print(f"  Mode: ALL AVAILABLE ({len(instances)} total)")
    else:
        print(f"  Mode: CUSTOM ({len(instances)} selected)")
    print(f"  Analyzing: {instances[:10]}{'...' if len(instances) > 10 else ''}")
    print("="*80)
    
    # Run analysis
    results = run_faithfulness_analysis(
        args.dataset,
        args.prompt_type,
        args.provider,
        args.model,
        instances,
        args.extraction_provider,
        args.extraction_model,
        args.bias_batch
    )
    
    # Generate report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    create_excel_report(results, args.output)
    
    print("\n" + "="*80)
    print("[OK] Analysis complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
