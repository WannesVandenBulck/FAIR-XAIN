"""
Generate and display narratives in a readable format.

Quick generation mode: Edit the QUICK_GENERATE_CONFIG section below, set QUICK_GENERATE_ENABLED = True, and run:
    python scripts/view_narratives.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================================
# QUICK GENERATE CONFIGURATION
# ============================================================================
# Edit these settings and run the script to quickly generate a single narrative

QUICK_GENERATE_ENABLED = True  # Set to True to generate a narrative

QUICK_GENERATE_CONFIG = {
    "dataset": "credit",                    # Dataset: "law", "credit", "student", "saudi"
    "instance": 0,                       # Instance index to generate narrative for
    "prompt_type": "cf",               # "shap" or "cf"
    "provider": "openai",                  # LLM provider for generation: "openai", "grok", etc.
    "model": "gpt-4o",  # Model name
    "display": True,                     # Display narrative after generation
    "save_to_file": "narrative_output4.md",                # Save to file: None, or path like "narrative.md" (markdown format if .md)
    # Optional bias injection settings (only used if model supports it)
    "personal_status_sex_override": "female single",             # For credit: personal_status_sex override like "male single"
    "age_override": 50,               # For credit: age override as number
}

# ============================================================================
# END CONFIGURATION
# ============================================================================


def load_narrative(dataset, instance, prompt_type, provider="openai", model=None):
    """Load a single narrative JSON file."""
    if model is None:
        # Try to find the first available model for this provider
        provider_path = Path(f"results/narratives/{dataset}/narratives/{prompt_type}/{provider}")
        if provider_path.exists():
            models = [d.name for d in provider_path.iterdir() if d.is_dir()]
            if models:
                model = models[0]
        if model is None:
            return None
    
    filepath = Path(f"results/narratives/{dataset}/narratives/{prompt_type}/{provider}/{model}/instance_{instance}.json")
    
    if not filepath.exists():
        return None
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_narrative_quick(dataset, instance, prompt_type, provider, model, 
                            gender_override=None, race_override=None):
    """Quick generation of a single narrative using LLM."""
    try:
        from llm_tools.llm_client import generate_text
        # Import the appropriate prompt module based on dataset
        if dataset == "law":
            from llm_tools.prompts import prompt_law as prompt_module
        elif dataset == "credit":
            from llm_tools.prompts import prompt_credit as prompt_module
        elif dataset == "saudi":
            from llm_tools.prompts import prompt_saudi as prompt_module
        elif dataset == "student":
            from llm_tools.prompts import prompt_student as prompt_module
        else:
            print(f"ERROR: Unknown dataset: {dataset}")
            return None
    except ImportError as e:
        print(f"ERROR: Could not import required modules: {e}")
        return None
    
    print(f"Generating narrative for {dataset} instance {instance} ({prompt_type})...")
    
    try:
        # Build prompt based on dataset and prompt type
        if prompt_type == "shap":
            if dataset == "law":
                prompt = prompt_module.build_shap_prompt(instance, gender_override=gender_override, race_override=race_override)
            elif dataset == "credit":
                prompt = prompt_module.build_shap_prompt(instance, personal_status_sex_override=gender_override, age_override=race_override)
            else:
                prompt = prompt_module.build_shap_prompt(instance, gender_override=gender_override, race_override=race_override)
        elif prompt_type == "cf":
            if dataset == "law":
                prompt = prompt_module.build_cf_prompt(instance, gender_override=gender_override, race_override=race_override)
            elif dataset == "credit":
                prompt = prompt_module.build_cf_prompt(instance, personal_status_sex_override=gender_override, age_override=race_override)
            else:
                prompt = prompt_module.build_cf_prompt(instance, gender_override=gender_override, race_override=race_override)
        else:
            print(f"ERROR: Unknown prompt type: {prompt_type}")
            return None
        
        # Generate narrative using generate_text function
        messages = [{"role": "user", "content": prompt}]
        narrative = generate_text(messages, provider=provider, model=model)
        
        # Save to JSON file structure
        output_dir = Path(f"results/narratives/{dataset}/narratives/{prompt_type}/{provider}/{model}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"instance_{instance}.json"
        
        data = {
            "dataset": dataset,
            "instance_idx": instance,
            "prompt_type": prompt_type,
            "provider": provider,
            "model": model,
            "status": "success",
            "narrative": narrative,
            "timestamp": datetime.now().isoformat(),
            "error": None
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to: {output_file}")
        return data
    
    except Exception as e:
        print(f"ERROR: Failed to generate narrative: {e}")
        return None


def display_narrative(dataset, instance, prompt_type, provider="openai", model=None, save_to=None):
    """Display a narrative in readable format and optionally save to file."""
    data = load_narrative(dataset, instance, prompt_type, provider, model)
    
    if data is None:
        print(f"Narrative not found: {dataset}/{prompt_type}/{provider}/{model or 'any_model'}/instance_{instance}")
        return
    
    # Build output text for console display (plain text format)
    output = []
    output.append("="*80)
    output.append(f"NARRATIVE: {data['dataset'].upper()} | Instance {data['instance_idx']} | {data['prompt_type'].upper()}")
    output.append("="*80)
    output.append(f"Provider: {data['provider']}")
    output.append(f"Model: {data['model']}")
    output.append(f"Status: {data['status']}")
    output.append(f"Timestamp: {data['timestamp']}")
    
    if data['status'] != 'success':
        output.append(f"Error: {data['error']}")
    else:
        output.append("")
        output.append("-"*80)
        output.append("NARRATIVE:")
        output.append("-"*80)
        output.append(data['narrative'])
    
    output.append("")
    output.append("="*80)
    output.append("")
    
    text = "\n".join(output)
    
    # Display to console
    print(text)
    
    # Save to file if requested
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        
        # Use markdown formatting if saving to .md file
        if save_to.endswith('.md'):
            markdown_text = f"""# Narrative: {data['dataset'].upper()} | Instance {data['instance_idx']} | {data['prompt_type'].upper()}

**Provider:** {data['provider']}  
**Model:** {data['model']}  
**Status:** {data['status']}  
**Timestamp:** {data['timestamp']}

"""
            if data['status'] != 'success':
                markdown_text += f"**Error:** {data['error']}\n"
            else:
                markdown_text += f"""## Explanation

{data['narrative']}
"""
            with open(save_to, "a", encoding="utf-8") as f:
                f.write(markdown_text)
        else:
            # Plain text format for non-markdown files
            with open(save_to, "a", encoding="utf-8") as f:
                f.write(text)
        print(f"✓ Saved to: {save_to}")
    
    return text


def main():
    """Main function - runs quick generation mode."""
    if not QUICK_GENERATE_ENABLED:
        print("Quick generation is disabled. Set QUICK_GENERATE_ENABLED = True in the script to generate narratives.")
        return
    
    print("\n" + "="*80)
    print("NARRATIVE GENERATION")
    print("="*80)
    print(f"Dataset: {QUICK_GENERATE_CONFIG['dataset']}")
    print(f"Instance: {QUICK_GENERATE_CONFIG['instance']}")
    print(f"Prompt Type: {QUICK_GENERATE_CONFIG['prompt_type']}")
    print(f"Provider: {QUICK_GENERATE_CONFIG['provider']}")
    print(f"Model: {QUICK_GENERATE_CONFIG['model']}")
    print("="*80 + "\n")
    
    # Generate narrative
    data = generate_narrative_quick(
        dataset=QUICK_GENERATE_CONFIG["dataset"],
        instance=QUICK_GENERATE_CONFIG["instance"],
        prompt_type=QUICK_GENERATE_CONFIG["prompt_type"],
        provider=QUICK_GENERATE_CONFIG["provider"],
        model=QUICK_GENERATE_CONFIG["model"],
        gender_override=QUICK_GENERATE_CONFIG.get("gender_override"),
        race_override=QUICK_GENERATE_CONFIG.get("race_override"),
    )
    
    if data is None:
        print("ERROR: Failed to generate narrative")
        return
    
    # Display if requested
    if QUICK_GENERATE_CONFIG["display"]:
        display_narrative(
            dataset=data["dataset"],
            instance=data["instance_idx"],
            prompt_type=data["prompt_type"],
            provider=data["provider"],
            model=data["model"],
            save_to=QUICK_GENERATE_CONFIG.get("save_to_file")
        )
    elif QUICK_GENERATE_CONFIG.get("save_to_file"):
        # Save without displaying
        display_narrative(
            dataset=data["dataset"],
            instance=data["instance_idx"],
            prompt_type=data["prompt_type"],
            provider=data["provider"],
            model=data["model"],
            save_to=QUICK_GENERATE_CONFIG.get("save_to_file")
        )
    
    print("\n✓ Done!\n")


if __name__ == "__main__":
    main()
