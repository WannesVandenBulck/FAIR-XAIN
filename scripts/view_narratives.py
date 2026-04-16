"""
View generated narratives in a readable format.

Usage:
    python scripts/view_narratives.py --dataset credit --prompt-type shap --instances 67
    python scripts/view_narratives.py --dataset credit --prompt-type cf --instances 67 --save narratives.txt
    python scripts/view_narratives.py --dataset saudi --prompt-type shap --all-instances --save all_narratives.txt
    python scripts/view_narratives.py --list  # Show all available narratives
"""

import json
import argparse
from pathlib import Path


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


def list_all_narratives():
    """List all available narratives."""
    results_dir = Path("results/narratives")
    
    if not results_dir.exists():
        print("No narratives found. Run make_narratives_new.py first.")
        return
    
    narratives = []
    for dataset_dir in results_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        narratives_dir = dataset_dir / "narratives"
        if not narratives_dir.exists():
            continue
        
        for prompt_type_dir in narratives_dir.iterdir():
            if not prompt_type_dir.is_dir():
                continue
            
            for provider_dir in prompt_type_dir.iterdir():
                if not provider_dir.is_dir():
                    continue
                
                for model_dir in provider_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    
                    for json_file in model_dir.glob("instance_*.json"):
                        with open(json_file, "r") as f:
                            data = json.load(f)
                        
                        narratives.append({
                            "Dataset": data["dataset"],
                            "Instance": data["instance_idx"],
                            "Type": data["prompt_type"],
                            "Provider": data["provider"],
                            "Model": data["model"],
                            "Status": data["status"],
                            "Length": len(data.get("narrative", "")) if data["status"] == "success" else 0
                        })
    
    if narratives:
        print(f"\nFound {len(narratives)} narratives:\n")
        print(f"{'Dataset':<12} {'Instance':<10} {'Type':<6} {'Provider':<12} {'Model':<20} {'Status':<10} {'Length':<8}")
        print("-" * 90)
        for n in narratives:
            print(f"{n['Dataset']:<12} {n['Instance']:<10} {n['Type']:<6} {n['Provider']:<12} {n['Model']:<20} {n['Status']:<10} {n['Length']:<8}")
    else:
        print("No narratives found.")


def display_narrative(dataset, instance, prompt_type, provider="openai", model=None, save_to=None):
    """Display a narrative in readable format and optionally save to file."""
    data = load_narrative(dataset, instance, prompt_type, provider, model)
    
    if data is None:
        print(f"Narrative not found: {dataset}/{prompt_type}/{provider}/{model or 'any_model'}/instance_{instance}")
        return
    
    # Build output text
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
        with open(save_to, "a", encoding="utf-8") as f:
            f.write(text)
        print(f"✓ Saved to: {save_to}")
    
    return text


def main():
    parser = argparse.ArgumentParser(description="View generated narratives")
    
    parser.add_argument("--dataset", choices=["saudi", "credit", "law", "student"],
                        help="Dataset name")
    parser.add_argument("--prompt-type", choices=["shap", "cf"],
                        help="Prompt type")
    parser.add_argument("--instances", nargs="+", type=int,
                        help="Instance indices to view")
    parser.add_argument("--all-instances", action="store_true",
                        help="View all instances for dataset/prompt-type")
    parser.add_argument("--provider", default="openai",
                        help="LLM provider (default: openai)")
    parser.add_argument("--model", default=None,
                        help="Model name (e.g., gpt-4o, grok-3-mini). If not specified, uses first available.")
    parser.add_argument("--list", action="store_true",
                        help="List all available narratives")
    parser.add_argument("--save", type=str, default=None,
                        help="Save narratives to text file (e.g., 'narratives_credit_67.txt')")
    
    args = parser.parse_args()
    
    # List all narratives
    if args.list:
        list_all_narratives()
        return
    
    # Validate arguments
    if not args.dataset or not args.prompt_type:
        if not args.list:
            parser.error("Must specify --dataset and --prompt-type, or use --list")
            return
    
    # Get instances to view
    if args.all_instances:
        if args.model:
            model_path = Path(f"results/narratives/{args.dataset}/narratives/{args.prompt_type}/{args.provider}/{args.model}")
        else:
            # Find first available model under provider
            provider_path = Path(f"results/narratives/{args.dataset}/narratives/{args.prompt_type}/{args.provider}")
            if provider_path.exists():
                models = [d.name for d in provider_path.iterdir() if d.is_dir()]
                if models:
                    model_path = provider_path / models[0]
                else:
                    print(f"No models found for {args.dataset}/{args.prompt_type}/{args.provider}")
                    return
            else:
                print(f"No narratives found for {args.dataset}/{args.prompt_type}/{args.provider}")
                return
        
        if not model_path.exists():
            print(f"No narratives found at {model_path}")
            return
        
        instances = []
        for json_file in sorted(model_path.glob("instance_*.json")):
            instance_num = int(json_file.stem.split("_")[1])
            instances.append(instance_num)
    elif args.instances:
        instances = args.instances
    else:
        parser.error("Must specify --instances or --all-instances")
        return
    
    # Create output file if saving
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear file if it exists
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("")
    
    # Display narratives
    for instance in instances:
        display_narrative(args.dataset, instance, args.prompt_type, args.provider, args.model, save_to=args.save)


if __name__ == "__main__":
    main()
