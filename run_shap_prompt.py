#!/usr/bin/env python
"""
Simple script to generate SHAP prompts for credit dataset instances.

Usage:
    python run_shap_prompt.py                 # Default: instance 438
    python run_shap_prompt.py 89              # Specific instance
    python run_shap_prompt.py 438 --save      # Save to file
"""

import sys
from pathlib import Path

# Add prompts directory to path
sys.path.insert(0, str(Path(__file__).parent / "llm_tools" / "prompts"))

from prompt_credit import build_shap_prompt


def main():
    # Parse arguments
    instance_index = 438  # Default
    save_to_file = False
    
    if len(sys.argv) > 1:
        try:
            instance_index = int(sys.argv[1])
        except ValueError:
            print(f"Error: instance_index must be an integer, got '{sys.argv[1]}'")
            sys.exit(1)
    
    if len(sys.argv) > 2 and sys.argv[2] == "--save":
        save_to_file = True
    
    # Generate prompt
    print(f"Generating SHAP prompt for instance {instance_index}...\n")
    
    try:
        prompt = build_shap_prompt(instance_index)
        
        # Save or print
        if save_to_file:
            filename = f"shap_prompt_{instance_index}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"Saved to: {filename}")
            print(f"Prompt length: {len(prompt)} characters")
        else:
            print(prompt)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure CSV files exist in datasets_prep/data/")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
