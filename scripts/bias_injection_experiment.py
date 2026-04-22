"""
Bias Injection Experiment Script

This script orchestrates the bias injection experiment by generating narratives for law dataset instances
with different protected attribute overrides.

The experiment tests whether LLM narratives exhibit bias based on gender and race 

INSTRUCTIONS:
1. Edit the CONFIGURATION section below to customize what you want to generate
2. Press the Run button or execute this script
3. Results will be saved to the configured output directory
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_narratives import get_available_instances, generate_narrative, save_result


# CONFIGURATION - EDIT THIS SECTION TO CUSTOMIZE THE EXPERIMENT

# Dataset to use
DATASET = "credit"  # Options: "law", "credit", "saudi", "student"

# Explanation method (prompt type)
EXPLANATION_METHOD = "cf"  # Options: "shap", "cf" (counterfactual)

# Which instances to process (choose one option only)
USE_ALL_INSTANCES = False  # Set to True to use all available instances
INSTANCE_INDICES = list(range(1))  # put in range

# LLM Provider and Model
PROVIDER = "grok"  # Options: "openai", "anthropic", "grok", "ollama"
MODEL = "grok-4-1-fast-non-reasoning"  # Examples: "gpt-4o", "claude-3-opus-20240229", "grok-3-mini", "grok-4-1-fast-reasoning"

# Output directory
OUTPUT_DIR = "results/narratives"

# Generate all gender-race combinations? (overrides BATCHES_TO_GENERATE if True)
GENERATE_ALL_COMBINATIONS = False  # Set to True to generate all 10 combinations automatically

# Batch configurations: Define which batches to generate (only used if GENERATE_ALL_COMBINATIONS=False)
# Set to empty list [] to skip a batch, or uncomment the config you want to run
# Note: For LAW dataset use gender_override and race_override
#       For CREDIT dataset use personal_status_sex_override and age_override
if DATASET == "law":
    BATCHES_TO_GENERATE = [
        {
            "name": "white_female",
            "gender_override": "female",
            "race_override": "white",
            "description": "All instances as white females"
        },
        {
            "name": "black_female",
            "gender_override": "female",
            "race_override": "black",
            "description": "All instances as black females"
        },
        {
            "name": "hispanic_female",
            "gender_override": "female",
            "race_override": "hispanic",
            "description": "All instances as hispanic females"
        },
        {
            "name": "asian_female", 
            "gender_override": "female",    
            "race_override": "asian",
            "description": "All instances as asian females"
        },
        {
            "name": "native_american_female",
            "gender_override": "female",    
            "race_override": "native american",
            "description": "All instances as native american females"
        },
    ]
elif DATASET == "credit":
    BATCHES_TO_GENERATE = [
        {
            "name": "male_single_age25",
            "personal_status_sex_override": "male single",
            "age_override": 25,
            "description": "All instances as 25-year-old single males"
        },
        {
            "name": "male_single_age50",
            "personal_status_sex_override": "male single",
            "age_override": 50,
            "description": "All instances as 50-year-old single males"
        },
        {
            "name": "female_single_age25",
            "personal_status_sex_override": "female single",
            "age_override": 25,
            "description": "All instances as 25-year-old single females"
        },
        {
            "name": "female_single_age50",
            "personal_status_sex_override": "female single",
            "age_override": 50,
            "description": "All instances as 50-year-old single females"
        },
        {
            "name": "female_married_age35",
            "personal_status_sex_override": "female divorced/separated/married",
            "age_override": 35,
            "description": "All instances as 35-year-old married females"
        },
    ]
else:
    # Default to law format
    BATCHES_TO_GENERATE = []

# Dry run mode (set to True to see what would be done without calling LLM)
DRY_RUN = False

# ============================================================================
# END OF CONFIGURATION - DON'T EDIT BELOW THIS LINE (unless you know what you're doing)
# ============================================================================

def generate_all_demographic_combinations():
    """
    Generate all combinations of demographic attributes based on dataset.
    
    Returns:
        List of batch configurations with all demographic combinations
    """
    combinations = []
    
    if DATASET == "law":
        # For law dataset: gender and race combinations
        genders = ["male", "female"]
        races = ["white", "black", "hispanic", "asian", "native american"]
        
        for gender in genders:
            for race in races:
                combinations.append({
                    "name": f"{race}_{gender}",
                    "gender_override": gender,
                    "race_override": race,
                    "description": f"All instances as {race} {gender}s"
                })
    
    elif DATASET == "credit":
        # For credit dataset: personal_status_sex and age combinations
        statuses = ["male single", "female single", "male married/widowed", "female divorced/separated/married"]
        ages = [25, 35, 50, 65]
        
        for status in statuses:
            for age in ages:
                status_key = status.lower().replace(" ", "_").replace("/", "_")
                combinations.append({
                    "name": f"{status_key}_age{age}",
                    "personal_status_sex_override": status,
                    "age_override": age,
                    "description": f"All instances as {status}, age {age}"
                })
    
    return combinations


def run_batch(dataset, instances, provider, model, batch_config, output_base_dir):
    """
    Run a single batch of narrative generation with specified overrides.
    
    Args:
        dataset: Dataset name (e.g., "law", "credit")
        instances: List of instance indices
        provider: LLM provider (e.g., "openai", "grok")
        model: LLM model name
        batch_config: Dictionary with batch configuration and overrides
        output_base_dir: Base output directory
    
    Returns:
        dict with results summary
    """
    batch_name = batch_config["name"]
    description = batch_config.get("description", "")
    
    print(f"\n{'='*80}")
    print(f"BATCH: {batch_name}")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"LLM Provider: {provider}")
    print(f"LLM Model: {model}")
    print(f"Total instances: {len(instances)}")
    print(f"Output directory: {output_base_dir}/{dataset}/narratives/{EXPLANATION_METHOD}/{provider}_{batch_name}/")
    print(f"Description: {description}")
    
    # Print overrides based on dataset
    if dataset == "law":
        gender_override = batch_config.get("gender_override")
        race_override = batch_config.get("race_override")
        if gender_override or race_override:
            print(f"\nAttribute Overrides:")
            if gender_override:
                print(f"  - Gender: {gender_override}")
            if race_override:
                print(f"  - Race: {race_override}")
        else:
            print(f"\nAttribute Overrides: None (original attributes)")
    elif dataset == "credit":
        personal_status_sex_override = batch_config.get("personal_status_sex_override")
        age_override = batch_config.get("age_override")
        if personal_status_sex_override or age_override:
            print(f"\nAttribute Overrides:")
            if personal_status_sex_override:
                print(f"  - Personal Status/Sex: {personal_status_sex_override}")
            if age_override:
                print(f"  - Age: {age_override}")
        else:
            print(f"\nAttribute Overrides: None (original attributes)")
    
    print(f"\n" + "="*80)
    
    # Process instances
    successful = 0
    failed = 0
    
    for i, instance_idx in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}] Instance {instance_idx}...", end=" ", flush=True)
        
        try:
            # Generate narrative with dataset-specific overrides
            if dataset == "law":
                result = generate_narrative(
                    dataset,
                    instance_idx,
                    EXPLANATION_METHOD,
                    provider=provider,
                    model=model,
                    gender_override=batch_config.get("gender_override"),
                    race_override=batch_config.get("race_override")
                )
            elif dataset == "credit":
                result = generate_narrative(
                    dataset,
                    instance_idx,
                    EXPLANATION_METHOD,
                    provider=provider,
                    model=model,
                    personal_status_sex_override=batch_config.get("personal_status_sex_override"),
                    age_override=batch_config.get("age_override")
                )
            else:
                result = generate_narrative(
                    dataset,
                    instance_idx,
                    EXPLANATION_METHOD,
                    provider=provider,
                    model=model,
                    gender_override=batch_config.get("gender_override"),
                    race_override=batch_config.get("race_override")
                )
            
            # Add batch name to result for tracking
            result["bias_batch"] = batch_name
            
            # Create output directory with batch name
            result_dir = Path(output_base_dir) / dataset / "narratives" / EXPLANATION_METHOD / f"{provider}_{batch_name}" / model
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # Save result
            import json
            filepath = result_dir / f"instance_{instance_idx}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            if result["status"] == "success":
                print("OK")
                successful += 1
            else:
                print(f"ERROR: {result['error']}")
                failed += 1
        
        except Exception as e:
            print(f"FAILED: {str(e)}")
            failed += 1
    
    # Print batch summary
    print(f"\n{'-'*80}")
    print(f"Batch '{batch_name}' Summary:")
    print(f"  Successful: {successful}/{len(instances)}")
    print(f"  Failed: {failed}/{len(instances)}")
    print(f"{'-'*80}\n")
    
    return {
        "batch_name": batch_name,
        "total_instances": len(instances),
        "successful": successful,
        "failed": failed
    }


def main():
    """Main function to run the bias injection experiment based on configuration."""
    
    print(f"\n{'*'*80}")
    print(f"BIAS INJECTION EXPERIMENT")
    print(f"{'*'*80}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Provider: {PROVIDER}")
    print(f"  Model: {MODEL}")
    
    # Determine which batches to generate
    if GENERATE_ALL_COMBINATIONS:
        batches_to_run = generate_all_demographic_combinations()
        print(f"  Mode: GENERATING ALL DEMOGRAPHIC COMBINATIONS (10 batches)")
    else:
        batches_to_run = BATCHES_TO_GENERATE
        print(f"  Mode: CUSTOM BATCHES ({len(batches_to_run)} batches)")
    
    print(f"  Total batches to generate: {len(batches_to_run)}")
    
    # Determine instances to process
    if USE_ALL_INSTANCES:
        instances = sorted(get_available_instances(DATASET, EXPLANATION_METHOD))
        print(f"  Instances: ALL ({len(instances)} total)")
    else:
        instances = sorted(INSTANCE_INDICES)
        print(f"  Instances: {len(instances)} selected ({INSTANCE_INDICES[:5]}{'...' if len(INSTANCE_INDICES) > 5 else ''})")
    
    if DRY_RUN:
        print(f"  Mode: DRY RUN (no LLM calls)")
    
    print(f"\nBatches to generate:")
    for i, batch in enumerate(batches_to_run, 1):
        print(f"  {i}. {batch['name']}: {batch['description']}")
    
    if DRY_RUN:
        print(f"\n[DRY RUN] Would process {len(instances)} instances")
        print(f"[DRY RUN] Sample instances: {instances[:3]}...")
        return
    
    # Run all batches
    batch_results = []
    
    for batch_config in batches_to_run:
        result = run_batch(
            DATASET,
            instances,
            PROVIDER,
            MODEL,
            batch_config,
            OUTPUT_DIR
        )
        batch_results.append(result)
    
    # Print final summary
    print(f"\n{'*'*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'*'*80}")
    print(f"Total batches completed: {len(batch_results)}")
    
    for result in batch_results:
        print(f"\nBatch '{result['batch_name']}':")
        print(f"  Success: {result['successful']}/{result['total_instances']}")
        if result['failed'] > 0:
            print(f"  Failed: {result['failed']}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"Completed: {datetime.now().isoformat()}")
    print(f"{'*'*80}\n")


if __name__ == "__main__":
    main()
