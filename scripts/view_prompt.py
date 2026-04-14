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
prompt_type = input("SHAP (s) or Counterfactual (c)? ").strip().lower()
if prompt_type not in ['s', 'c']:
    print("Invalid type"); exit()

# Show available based on type
if prompt_type == 's':
    available_idx = sorted(shap_df['instance_index'].unique())
else:
    cf_df = pd.read_csv(cf_path)
    available_idx = sorted(cf_df['instance_index'].unique())

print(f"Available instances for {('SHAP' if prompt_type == 's' else 'Counterfactual')}: {available_idx}")

idx = int(input(f"Choose instance: "))

if idx not in available_idx:
    print(f"Instance {idx} not available. Choose from: {available_idx}"); exit()

# Build full prompt using the module's functions
try:
    if prompt_type == 's':
        full_prompt = prompt_module.build_shap_prompt(idx)
    else:
        full_prompt = prompt_module.build_cf_prompt(idx)
except Exception as e:
    print(f"Error: {e}"); exit()

# Save to file
filename = f"prompt_{name}_{idx}_{prompt_type}.txt"
with open(filename, 'w', encoding='utf-8') as f:
    f.write(full_prompt)

print(full_prompt)
print(f"\n{'='*80}")
print(f"Prompt saved to: {filename}")
