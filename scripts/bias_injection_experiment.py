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
DATASET = "law"  # Options: "law", "credit", "saudi", "student"

# Which instances to process (choose one option only)
USE_ALL_INSTANCES = False  # Set to True to use all available instances
INSTANCE_INDICES = list(range(101))  # put in range

# LLM Provider and Model
PROVIDER = "grok"  # Options: "openai", "anthropic", "grok", "ollama"
MODEL = "grok-4-1-fast-non-reasoning"  # Examples: "gpt-4o", "claude-3-opus-20240229", "grok-3-mini", "grok-4-1-fast-reasoning"

# Output directory
OUTPUT_DIR = "results/narratives"

# Generate all gender-race combinations? (overrides BATCHES_TO_GENERATE if True)
GENERATE_ALL_COMBINATIONS = False  # Set to True to generate all 10 combinations automatically

# Batch configurations: Define which batches to generate (only used if GENERATE_ALL_COMBINATIONS=False)
# Set to empty list [] to skip a batch, or uncomment the config you want to run
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

# Dry run mode (set to True to see what would be done without calling LLM)
DRY_RUN = False

# ============================================================================
# END OF CONFIGURATION - DON'T EDIT BELOW THIS LINE (unless you know what you're doing)
# ============================================================================

def generate_all_demographic_combinations():
    """
    Generate all combinations of gender and race.
    
    Returns:
        List of batch configurations with all gender-race combinations
    """
    genders = ["male", "female"]
    races = ["white", "black", "hispanic", "asian", "native american"]
    
    combinations = []
    for gender in genders:
        for race in races:
            combinations.append({
                "name": f"{race}_{gender}",
                "gender_override": gender,
                "race_override": race,
                "description": f"All instances as {race} {gender}s"
            })
    
    return combinations


def run_batch(dataset, instances, provider, model, gender_override, race_override, batch_name, output_base_dir):
    """
    Run a single batch of narrative generation with specified overrides.
    
    Args:
        dataset: Dataset name (e.g., "law")
        instances: List of instance indices
        provider: LLM provider (e.g., "openai", "grok")
        model: LLM model name
        gender_override: Gender to use for all instances (None for no override)
        race_override: Race to use for all instances (None for no override)
        batch_name: Name for this batch (e.g., "white_male", "black_female")
        output_base_dir: Base output directory
    
    Returns:
        dict with results summary
    """
    print(f"\n{'='*80}")
    print(f"BATCH: {batch_name}")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"LLM Provider: {provider}")
    print(f"LLM Model: {model}")
    print(f"Total instances: {len(instances)}")
    print(f"Output directory: {output_base_dir}/{dataset}/narratives/shap/{provider}_{batch_name}/")
    
    if gender_override or race_override:
        print(f"\nAttribute Overrides:")
        if gender_override:
            print(f"  - Gender: {gender_override}")
        if race_override:
            print(f"  - Race: {race_override}")
    else:
        print(f"\nAttribute Overrides: None (original attributes)")
    
    print(f"\n" + "="*80)
    
    # Process instances
    successful = 0
    failed = 0
    
    for i, instance_idx in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}] Instance {instance_idx}...", end=" ", flush=True)
        
        try:
            # Generate narrative
            result = generate_narrative(
                dataset,
                instance_idx,
                "shap",
                provider=provider,
                model=model,
                gender_override=gender_override,
                race_override=race_override
            )
            
            # Add batch name to result for tracking
            result["bias_batch"] = batch_name
            
            # Create output directory with batch name
            result_dir = Path(output_base_dir) / dataset / "narratives" / "shap" / f"{provider}_{batch_name}" / model
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
        instances = sorted(get_available_instances(DATASET, "shap"))
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
            batch_config.get("gender_override"),
            batch_config.get("race_override"),
            batch_config["name"],
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
