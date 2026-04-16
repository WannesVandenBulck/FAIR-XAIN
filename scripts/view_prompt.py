import sys
import os

# Add workspace root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_tools.prompts import prompt_saudi, prompt_credit, prompt_law, prompt_student
import pandas as pd

# Dataset mapping
datasets = {
    "1": ("saudi", prompt_saudi, "datasets_prep/data/saudi_dataset/saudi_shap.csv", "datasets_prep/data/saudi_dataset/saudi_counterfactual.csv"),
    "2": ("credit", prompt_credit, "datasets_prep/data/credit_dataset/credit_shap.csv", "datasets_prep/data/credit_dataset/credit_counterfactual.csv"),
    "3": ("law", prompt_law, "datasets_prep/data/law_dataset/law_shap.csv", "datasets_prep/data/law_dataset/law_counterfactual.csv"),
    "4": ("student", prompt_student, "datasets_prep/data/student_dataset/student_shap.csv", "datasets_prep/data/student_dataset/student_counterfactual.csv"),
}

# Select dataset
print("Datasets: 1=saudi, 2=credit, 3=law, 4=student")
dataset_choice = input("Choose dataset (1-4): ").strip()
if dataset_choice not in datasets:
    print("Invalid choice"); exit()

name, prompt_module, shap_path, cf_path = datasets[dataset_choice]

# Show available instances
shap_df = pd.read_csv(shap_path)
available = sorted(shap_df['instance_index'].unique())
print(f"Available instances for SHAP: {available[:20]}... (showing first 20)")

# Select type first
print("\nPrompt types:")
print("  s = Standard SHAP")
print("  n = Narrative SHAP (story-focused)")
print("  c = Counterfactual")
prompt_choice = input("Choose prompt type (s/n/c): ").strip().lower()

if prompt_choice == 's':
    prompt_type = 'shap'
    available_idx = sorted(shap_df['instance_index'].unique())
elif prompt_choice == 'n':
    prompt_type = 'narrative'
    available_idx = sorted(shap_df['instance_index'].unique())
elif prompt_choice == 'c':
    prompt_type = 'counterfactual'
    cf_df = pd.read_csv(cf_path)
    available_idx = sorted(cf_df['instance_index'].unique())
else:
    print("Invalid type"); exit()

print(f"Available instances for {prompt_type}: {available_idx}")

idx = int(input(f"Choose instance: "))

if idx not in available_idx:
    print(f"Instance {idx} not available. Choose from: {available_idx}"); exit()

# Build full prompt using the module's functions
try:
    if prompt_type == 'shap':
        full_prompt = prompt_module.build_shap_prompt(idx)
    elif prompt_type == 'narrative':
        # Check if module has narrative variant function
        if hasattr(prompt_module, 'build_shap_prompt_narrative'):
            full_prompt = prompt_module.build_shap_prompt_narrative(idx)
        else:
            print(f"Error: Narrative SHAP prompt not available for {name} dataset"); exit()
    else:  # counterfactual
        full_prompt = prompt_module.build_cf_prompt(idx)
except Exception as e:
    print(f"Error: {e}"); exit()

# Save to file
type_suffix = {'shap': 's', 'narrative': 'n', 'counterfactual': 'c'}[prompt_type]
filename = f"prompt_{name}_{idx}_{type_suffix}.txt"
with open(filename, 'w', encoding='utf-8') as f:
    f.write(full_prompt)

print(full_prompt)
print(f"\n{'='*80}")
print(f"Prompt saved to: {filename}")
