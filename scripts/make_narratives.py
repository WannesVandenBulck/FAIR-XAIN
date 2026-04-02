import os
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
ROOT = current_file.parent if (current_file.parent / "data").exists() else current_file.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_tools.prompts.prompt_adult import (
    DATASET_DESCRIPTION, 
    SHAP_EXPLANATION, 
    SHAP_PROMPT_INSTRUCTIONS,
    describe_instance
)
from llm_tools.llm_client import generate_text

# --- CONFIG ---
DATA_DIR = os.path.join(ROOT, "data")

GENERATE_SHAP_NARRATIVES = True
GENERATE_CF_NARRATIVES = False

# Process all groups: original + counterfactual attribute swaps
GROUPS = [
    "minority",
    "majority", 
    "minority_with_majority_race",
    "majority_with_minority_race"
]

SHAP_NARRATIVES_DIRS = {
    group: os.path.join(DATA_DIR, "shap_narratives", group) 
    for group in GROUPS
}

for dir_path in SHAP_NARRATIVES_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o"

def build_shap_prompt(shap_data, instance_idx, instance_data):
    shap_vals = shap_data['shap_values']
    feature_names = shap_data['feature_names']
    base_value = shap_data['base_value']
    instance_shap = shap_vals[instance_idx]
    
    shap_table_data = []
    for feat_name, feat_val, shap_val in zip(feature_names, instance_data.values, instance_shap):
        shap_arr = np.asarray(shap_val).flatten()
        shap_table_data.append({
            "feature": feat_name,
            "value": round(float(feat_val), 3),
            "shap_value": round(float(shap_arr[0]), 4)
        })
    
    shap_df = pd.DataFrame(shap_table_data)
    
    # Extract scalar base value
    base_val = np.asarray(base_value).flatten()[0]
    
    try:
        instance_desc = describe_instance(instance_data)
    except KeyError:
        instance_desc = "Instance features:\n" + "\n".join(
            f"- {name}: {round(float(val), 3)}" for name, val in zip(feature_names, instance_data.values)
        )
    
    prompt = f"""{DATASET_DESCRIPTION}

{SHAP_EXPLANATION}

{instance_desc}

SHAP values:
{shap_df.to_string(index=False)}

Base value: {round(float(base_val), 4)}

{SHAP_PROMPT_INSTRUCTIONS}
"""
    return prompt

if __name__ == "__main__":
    for group in GROUPS:
        SHAP_PATH = os.path.join(DATA_DIR, "shap", group, "shap_values.pkl")
        SHAP_NARRATIVES_DIR = SHAP_NARRATIVES_DIRS[group]
        
        if not os.path.exists(SHAP_PATH):
            print(f"Skipping {group}: no SHAP file found")
            continue
        
        with open(SHAP_PATH, "rb") as f:
            shap_data = pickle.load(f)
        
        X_explain = pd.DataFrame(shap_data['X_explain'], columns=shap_data['feature_names'])
        num_instances = shap_data['num_instances']
        
        for idx in range(num_instances):
            instance_data = X_explain.iloc[idx]
            
            if GENERATE_SHAP_NARRATIVES:
                prompt = build_shap_prompt(shap_data, idx, instance_data)
                
                messages = [
                    {"role": "system", "content": "You are an expert at explaining machine learning predictions to non-technical users."},
                    {"role": "user", "content": prompt},
                ]
                
                narrative = generate_text(messages, provider=LLM_PROVIDER, model=LLM_MODEL)
                
                out_path = os.path.join(SHAP_NARRATIVES_DIR, f"instance_{idx}.md")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(f"# Instance {idx}\n\n{narrative}")
