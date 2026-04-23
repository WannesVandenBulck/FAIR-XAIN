#!/usr/bin/env python3
import json
import sys
sys.path.insert(0, '.')

from scripts.compute_faithfulness import (
    extract_ground_truth, 
    extract_from_narrative,
    compute_shap_faithfulness
)

# Load narrative
with open('results/narratives/credit/narratives/shap/grok_female_married_age35/grok-4-1-fast-non-reasoning/instance_0.json') as f:
    data = json.load(f)
    narrative = data['narrative']

print('NARRATIVE LENGTH:', len(narrative), 'chars')
print('NARRATIVE (first 500 chars):')
print(narrative[:500])
print('...\n')

# Extract using the LLM
print('EXTRACTING FROM NARRATIVE...')
extracted = extract_from_narrative(narrative, 'shap', provider='openai', model='gpt-4o')

print('EXTRACTED:')
for feat_name, feat_dict in extracted.items():
    if feat_name not in ['instance_features', 'averages_mentioned']:
        print(f"  {feat_name}: rank={feat_dict.get('rank')}, sign={feat_dict.get('sign')}, value={feat_dict.get('value')}")

print(f"\nTotal extracted features: {len([k for k in extracted if k not in ['instance_features', 'averages_mentioned']])}\n")

# Extract ground truth from the original prompts by loading them from the results
# We'll use the compare narratives approach
from pathlib import Path
import os

# Find the prompt file
prompt_files = list(Path('results/comparisons').glob('*credit*.json'))
if not prompt_files:
    print("No prompt files found, trying alternate method...")
    # Just try to extract from file structure
    print("\nNote: Ground truth extraction requires access to original prompt construction")
    
    # Try to manually construct the expected ground truth
    from llm_tools.prompts.prompt_credit import build_shap_prompt
    
    # Use numeric values for overrides
    # From context, female_married is status code we need to find
    # age_override=35
    try:
        prompt = build_shap_prompt(instance_index=0)
        print("\n✓ Using base prompt without overrides")
    except Exception as e:
        print(f"✗ Error: {e}")
else:
    print(f"Found {len(prompt_files)} prompt files")
