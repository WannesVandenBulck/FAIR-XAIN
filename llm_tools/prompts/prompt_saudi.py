import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Load dataset_info from pickle file
DATASET_INFO_PATH = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "saudi_dataset" / "dataset_info"

def load_dataset_info():
    """Load dataset info from pickle file"""
    with open(DATASET_INFO_PATH, 'rb') as f:
        return pickle.load(f)

DATASET_INFO = load_dataset_info()

def get_dataset_description():
    """Generate dataset description from loaded info"""
    if DATASET_INFO is None:
        return ""
    
    desc = DATASET_INFO.get("dataset_description", "")
    target = DATASET_INFO.get("target_description", "")
    task = DATASET_INFO.get("task_description", "")
    
    return f"""{desc}

Target: {target}

ML Task: {task}"""

def create_instance_description_from_row(row):
    """
    Create instance description using actual feature names and descriptions from dataset_info.
    
    Parameters:
    - row: pandas Series with feature values
    """
    if DATASET_INFO is None:
        # Fallback for when dataset_info isn't available
        feature_lines = [f"- {col} = {row[col]}" for col in row.index]
    else:
        feature_df = DATASET_INFO.get("feature_description")
        feature_lines = []
        
        for col in row.index:
            value = row[col]
            # Find feature description
            feature_info = feature_df[feature_df['feature_name'] == col]
            if not feature_info.empty:
                desc = feature_info.iloc[0]['feature_desc']
                avg = feature_info.iloc[0]['feature_average']
                # Handle numeric and non-numeric values
                try:
                    value_str = f"{float(value):.2f}"
                    avg_str = f"{float(avg):.2f}"
                    feature_lines.append(f"- {col} = {value_str} (avg: {avg_str}) - {desc}")
                except (ValueError, TypeError):
                    feature_lines.append(f"- {col} = {value} - {desc}")
            else:
                feature_lines.append(f"- {col} = {value}")
    
    instance_desc = f"""The model is making a prediction for a job applicant.

Feature values:
{chr(10).join(feature_lines)}

"""
    return instance_desc

def describe_instance(row):
    """Generate instance description from row"""
    return create_instance_description_from_row(row)

# ===== PROMPT TEMPLATES =====

PROMPT_PREAMBLE_SHAP = """
A machine learning model predicted that an employee will LEAVE their position and therefore a promotion offer was REJECTED.

YOUR TASK: Translate the following technical information into a clear, non-technical narrative explanation that helps the applicant understand:
- Why the model made this prediction
- Which factors were most important in this decision
- How their specific situation compared to other employees

INFORMATION YOU WILL RECEIVE:
1. DATASET INFORMATION: Context about the dataset, target variable and ML task used to train the model
2. TECHNICAL EXPLANATION METHOD: How we measure feature importance (SHAP values)
3. APPLICANT PROFILE: The employee's specific feature values with comparisons to dataset averages
4. FEATURE IMPORTANCE ANALYSIS: SHAP values showing which features most influenced the decision
5. CLEAR INSTRUCTIONS: What narrative you should write
"""

PROMPT_PREAMBLE_CF = """
A machine learning model predicted that an employee will LEAVE their position and therefore a promotion offer was REJECTED.

YOUR TASK: Summarize the following counterfactual scenarios into a clear, non-technical narrative explanation that helps the applicant understand:
- Why the model made this prediction
- Which factors were most important in this decision
- How their specific situation compared to other employees
- What changes would be needed to flip the prediction to "will stay" and get the job offer accepted

INFORMATION YOU WILL RECEIVE:
1. DATASET INFORMATION: Context about the dataset, target variable and ML task used to train the model
2. COUNTERFACTUAL EXPLANATION: Information about what counterfactuals are and how to interpret them
3. APPLICANT PROFILE: The employee's specific feature values with comparisons to dataset averages and the model's predicted probability of leaving
4. COUNTERFACTUAL SCENARIOS TABLE: A table showing the original instance and multiple counterfactual scenarios with feature changes that would flip the prediction
5. CLEAR INSTRUCTIONS: What narrative you should write
"""

DATASET_EXPLANATION = """
1. DATASET INFORMATION
"""

APPLICANT_INFORMATION = """
3. APPLICANT PROFILE 
"""

SHAP_VALUES_SECTION = """
4. FEATURE IMPORTANCE ANALYSIS (Ranked by Influence)
"""

INSTRUCTIONS_SECTION = """
5. YOUR NARRATIVE TASK
"""

COUNTERFACTUAL_EXPLANATION_DETAILS = """
2. COUNTERFACTUAL ANALYSIS (Alternative Scenarios)

You are given a table comparing the employee's current situation (original) with alternative scenarios (counterfactuals cf_1, cf_2, etc.):

- The 'original' row shows the employee's actual feature values and the model's actual prediction (will leave).
- Each 'cf_k' row shows what would happen if certain features changed - representing scenarios where the model WOULD predict staying.

In other words: Each counterfactual is a "what if" scenario showing the minimum feature changes needed to flip the model's prediction from "will leave" to "will stay". This helps answer: "What would need to be different for the job offer to be accepted?"

Compensation, work environment, job satisfaction, and career development opportunities can vary. The counterfactuals show which combinations of changes would be most effective in improving employee retention prospects.
"""

SHAP_EXPLANATION = """
2. TECHNICAL EXPLANATION: SHAP VALUES

You are given SHAP values for this applicant's prediction.

SHAP values explain how much each feature contributes to the model's prediction for this specific applicant.
Each feature has a SHAP value that tells you:
- How much that feature influenced the model's decision for this applicant.
- Whether it pushed the prediction toward "will leave" (positive contribution) or "will stay" (negative contribution).
- Larger absolute values indicate features with stronger influence on the prediction.

Features are ranked by their absolute SHAP values, with the most influential features listed first.
Features with positive SHAP values contributed toward a "will leave" prediction.
Features with negative SHAP values contributed toward a "will stay" prediction.
"""

SHAP_PROMPT_INSTRUCTIONS = """
TASK:
Your goal is to generate a textual explanation or narrative explaining why the model made this prediction for this applicant.

Write a detailed narrative explanation for a non-technical reader that explains:
1) The model's predicted probability of leaving the job and what this means for the applicant.
2) Which features were most important in driving this prediction.
3) How each important feature contributed (either pushing toward leaving or toward staying).
4) The relative importance ranking of features based on their influence.
5) A summary of the overall story: which factors are the primary drivers of this prediction.

CONSTRAINTS:
- Only use information from the dataset description, instance description, and SHAP values table.
- Do NOT invent new SHAP values or numerical values.
- Do not use the numeric SHAP values in your answer. Instead, discuss the ranking and direction of influence.
- Do not talk about model internals, algorithms, or training details.
- Focus on the direction (positive/negative contribution) and relative ranking of features.

STYLE:
- Length: 12-15 sentences.
- Tone: neutral, explanatory, respectful.
- Write a coherent narrative without bullet points or tables. The goal is to have a narrative/story.
- Directly address the applicant. Be empathic and respectful.
- Consider the relative magnitude of feature importance when discussing their roles in the decision.
"""

COUNTERFACTUAL_PROMPT_INSTRUCTIONS = """
TASK:
- You are given a table of counterfactuals for the same original applicant instance.
- Summarize what the model considers important for changing the prediction outcome.
- Provide concrete, actionable insights about which feature changes would shift the prediction.
- Discuss patterns: which features consistently need to change together to flip the prediction.

Write a detailed narrative explanation for a non-technical reader:
1) Briefly summarize the current prediction and employment situation.
2) Explain what counterfactuals represent: "what if" scenarios that would change the prediction.
3) Specify which features always changed (MUST changes) and which never changed.
4) Quantified summary: how many counterfactuals were generated, which features changed most often.
5) Describe which features changed together - this shows effective combination patterns.
6) End with actionable guidance: based on patterns, which features are realistic to change and by approximately how much.

CONSTRAINTS:
- Only use information from dataset description, instance description, and counterfactual table.
- Do NOT invent new feature values or examples.
- Do NOT refer to individual counterfactuals; discuss overall patterns only.
- Do not talk about model internals or training details.
- Use realistic ranges when discussing feature changes.

STYLE:
- Length: 15-18 sentences.
- Tone: neutral, explanatory, helpful.
- Write a coherent narrative without bullet points or tables.
- Directly address the applicant. Be respectful and empathic.
- Focus on actionable insights the applicant can implement.
"""


def build_shap_prompt(instance_index, shap_csv_path: str = None) -> str:
    """
    Build a SHAP explanation prompt by loading from the SHAP CSV.
    
    Parameters:
    - instance_index: the instance index to explain (e.g., 10, 25, etc.)
    - shap_csv_path: path to the SHAP CSV file (defaults to saudi_dataset/saudi_shap.csv)
    
    Returns:
    - Full prompt string ready for LLM
    """
    if shap_csv_path is None:
        shap_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "saudi_dataset" / "saudi_shap.csv"
    
    # Load SHAP values (instance_index is now an explicit column)
    shap_df = pd.read_csv(shap_csv_path)
    shap_row = shap_df[shap_df['instance_index'] == instance_index]
    
    if shap_row.empty:
        raise ValueError(f"Instance {instance_index} not found in SHAP CSV")
    
    shap_values = shap_row.iloc[0]
    
    # Extract predicted_probability
    predicted_probability = shap_values.get('predicted_probability', np.nan)
    
    # Load corresponding original data
    test_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "saudi_dataset" / "saudi_adverse.csv"
    adverse_df = pd.read_csv(test_csv_path)
    adverse_row = adverse_df[adverse_df['instance_index'] == instance_index]
    
    if adverse_row.empty:
        raise ValueError(f"Instance {instance_index} not found in adverse CSV")
    
    original_instance = adverse_row.iloc[0]
    prediction = original_instance['predicted_class']
    
    # Extract SHAP values (remove instance_index and SHAP_ prefix)
    shap_dict = {}
    for col in shap_values.index:
        if col.startswith('SHAP_'):
            feature_name = col[5:]  # Remove 'SHAP_' prefix
            shap_dict[feature_name] = shap_values[col]
    
    # Create instance description (excluding metadata columns)
    feature_cols = [col for col in original_instance.index if col not in 
                    ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target']]
    instance_row = original_instance[feature_cols]
    
    instance_desc = describe_instance(instance_row)
    
    # Create SHAP table as simple text
    shap_table_df = pd.DataFrame({
        'Feature': list(shap_dict.keys()),
        'SHAP_Value': list(shap_dict.values())
    }).sort_values('SHAP_Value', key=abs, ascending=False)
    
    shap_table = shap_table_df.to_string(index=False)
    
    # Format predicted_probability for display
    pred_prob_str = f"{predicted_probability:.1%}" if not np.isnan(predicted_probability) else "N/A"
    
    # Get fresh dataset description (compute dynamically instead of using module-level variable)
    dataset_desc = get_dataset_description()
    
    prompt = f"""{PROMPT_PREAMBLE_SHAP}
{DATASET_EXPLANATION}
{dataset_desc}

{SHAP_EXPLANATION}

{APPLICANT_INFORMATION}
{instance_desc}

The model's prediction:
- Predicted probability of leaving: {pred_prob_str}

{SHAP_VALUES_SECTION}
{shap_table}

{INSTRUCTIONS_SECTION}
{SHAP_PROMPT_INSTRUCTIONS}
"""
    return prompt


def build_cf_prompt(instance_index, cf_csv_path: str = None, adverse_csv_path: str = None, shap_csv_path: str = None) -> str:
    """
    Build a counterfactual prompt by loading from the CSV files.
    
    Parameters:
    - instance_index: the instance index to explain (e.g., 10, 25, etc.)
    - cf_csv_path: path to counterfactual CSV (defaults to saudi_dataset/saudi_counterfactual.csv)
    - adverse_csv_path: path to adverse CSV (defaults to saudi_dataset/saudi_adverse.csv)
    - shap_csv_path: path to SHAP CSV for predicted_probability (defaults to saudi_dataset/saudi_shap.csv)
    
    Returns:
    - Full prompt string ready for LLM
    """
    if cf_csv_path is None:
        cf_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "saudi_dataset" / "saudi_counterfactual.csv"
    if adverse_csv_path is None:
        adverse_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "saudi_dataset" / "saudi_adverse.csv"
    if shap_csv_path is None:
        shap_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "saudi_dataset" / "saudi_shap.csv"
    
    # Load original instance (instance_index is now an explicit column)
    adverse_df = pd.read_csv(adverse_csv_path)
    adverse_row = adverse_df[adverse_df['instance_index'] == instance_index]
    
    if adverse_row.empty:
        raise ValueError(f"Instance {instance_index} not found in adverse CSV")
    
    original = adverse_row.iloc[0]
    prediction = original['predicted_class']
    
    # Load predicted_probability from SHAP CSV
    shap_df = pd.read_csv(shap_csv_path)
    shap_row = shap_df[shap_df['instance_index'] == instance_index]
    if shap_row.empty:
        predicted_probability = np.nan
    else:
        predicted_probability = shap_row.iloc[0]['predicted_probability']
    
    # Load counterfactuals for this instance (use integer comparison, no float conversion)
    cf_df = pd.read_csv(cf_csv_path)
    cf_rows = cf_df[cf_df['instance_index'] == instance_index]
    
    if cf_rows.empty:
        raise ValueError(f"No counterfactuals found for instance {instance_index}")
    
    # Create table with original + counterfactuals
    # Extract feature columns only (exclude metadata like instance_index, original_test_index, CF_number, distance_to_original)
    feature_cols = [col for col in adverse_df.columns 
                   if col not in ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target']]
    
    table_data = []
    
    # Add original row
    original_features = original[feature_cols]
    table_data.append(dict(row_type='original', **original_features))
    
    # Add counterfactuals
    for idx, cf_row in cf_rows.iterrows():
        cf_data = {'row_type': f"cf_{int(cf_row['CF_number'])}"}
        for col in feature_cols:
            cf_data[col] = cf_row[col]
        table_data.append(cf_data)
    
    table_df = pd.DataFrame(table_data).set_index('row_type')
    
    # Create instance description (excluding metadata)
    instance_features = original[[c for c in original.index if c not in 
                                   ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target']]]
    instance_desc = describe_instance(instance_features)
    
    table_str = table_df.to_string()

    # Format predicted_probability for display
    pred_prob_str = f"{predicted_probability:.1%}" if not np.isnan(predicted_probability) else "N/A"
    
    # Get fresh dataset description (compute dynamically instead of using module-level variable)
    dataset_desc = get_dataset_description()
    
    prompt = f"""{PROMPT_PREAMBLE_CF}
{DATASET_EXPLANATION}
{dataset_desc}

{COUNTERFACTUAL_EXPLANATION_DETAILS}

{APPLICANT_INFORMATION}
{instance_desc}

The model's prediction:
- Predicted probability of leaving: {pred_prob_str}

4. COUNTERFACTUAL SCENARIOS TABLE

{table_str}

{INSTRUCTIONS_SECTION}
{COUNTERFACTUAL_PROMPT_INSTRUCTIONS}
"""
    return prompt


