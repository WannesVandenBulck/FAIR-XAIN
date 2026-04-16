import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Protected attribute mappings for readable display in prompts
ATTRIBUTE_VALUE_MAPPINGS = {
    "gender": {0: "female", 1: "male"},
    "race1": {0: "white", 1: "black", 2: "hispanic", 3: "asian", 4: "native american"},
    "fulltime": {1: "Full-time", 1.0: "Full-time", 2: "Part-time", 2.0: "Part-time"},
    "fam_inc": {1.0: "Low", 2.0: "Lower-middle", 3.0: "Middle", 4.0: "Upper-middle", 5.0: "High", 
                1: "Low", 2: "Lower-middle", 3: "Middle", 4: "Upper-middle", 5: "High"},
}

# Load dataset_info from pickle file
DATASET_INFO_PATH = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "law_dataset" / "dataset_info"

def load_dataset_info():
    """Load dataset info from pickle file"""
    with open(DATASET_INFO_PATH, 'rb') as f:
        return pickle.load(f)

DATASET_INFO = load_dataset_info()

def map_attribute_value(feature_name, value):
    """
    Map numeric/code attribute values to human-readable names.
    
    Parameters:
    - feature_name: the feature name (e.g., "gender", "race1")
    - value: the numeric or code value (e.g., 0, 1, "A91")
    
    Returns:
    - Mapped readable name if mapping exists, otherwise the original value
    """
    if feature_name in ATTRIBUTE_VALUE_MAPPINGS:
        mapping = ATTRIBUTE_VALUE_MAPPINGS[feature_name]
        # Try numeric conversion for the mapping lookup
        try:
            numeric_value = float(value) if not isinstance(value, int) else value
            # Check if numeric value matches a key in mapping
            if numeric_value in mapping:
                return mapping[numeric_value]
            # Check if integer version matches
            if int(numeric_value) in mapping:
                return mapping[int(numeric_value)]
        except (ValueError, TypeError):
            pass
        # If not found as numeric, try as-is (for string codes)
        if value in mapping:
            return mapping[value]
    return value

def get_dataset_description():
    """Generate dataset description from loaded info with clear target encoding."""
    if DATASET_INFO is None:
        base_desc = ""
    else:
        desc = DATASET_INFO.get("dataset_description", "")
        target = DATASET_INFO.get("target_description", "")
        task = DATASET_INFO.get("task_description", "")
        base_desc = f"{desc}\n\nTarget Variable: {target}\n\nML Task: {task}"
    
    # Add clear target encoding explanation
    target_encoding = """
Target Encoding (target_law):
- 1 (model predicted 1) = FAILED BAR EXAM: Student is predicted to fail the bar exam (application was REJECTED)
- 0 (not predicted, favorable) = PASSED BAR EXAM: Student showed strong performance and would pass (application would be approved)
"""
    
    return base_desc + target_encoding if base_desc else target_encoding

def create_instance_description_from_row(row):
    """
    Create instance description using actual feature names and descriptions from dataset_info.
    For categorical features, displays distribution; for numerical features, displays average.
    Protected attributes (gender, race1) are mapped to readable names.
    
    Parameters:
    - row: pandas Series with feature values
    """
    if DATASET_INFO is None:
        # Fallback for when dataset_info isn't available
        feature_lines = []
        for col in row.index:
            mapped_value = map_attribute_value(col, row[col])
            feature_lines.append(f"- {col} = {mapped_value}")
    else:
        feature_df = DATASET_INFO.get("feature_description")
        feature_lines = []
        
        for col in row.index:
            value = row[col]
            # Map attribute value if applicable
            mapped_value = map_attribute_value(col, value)
            
            # Find feature description
            feature_info = feature_df[feature_df['feature_name'] == col]
            if not feature_info.empty:
                desc = feature_info.iloc[0]['feature_desc']
                distribution = feature_info.iloc[0].get('feature_distribution')
                avg = feature_info.iloc[0].get('feature_average')
                
                # If distribution exists (categorical feature), show distribution; otherwise show average (numerical)
                if pd.notna(distribution) and distribution is not None:
                    # Categorical feature - show distribution
                    feature_lines.append(f"- {col} = {mapped_value} - Distribution: {distribution} - {desc}")
                elif pd.notna(avg) and avg is not None:
                    # Numerical feature - show average
                    try:
                        value_str = f"{float(mapped_value):.2f}" if isinstance(mapped_value, (int, float)) else mapped_value
                        avg_str = f"{float(avg):.2f}"
                        feature_lines.append(f"- {col} = {value_str} (avg: {avg_str}) - {desc}")
                    except (ValueError, TypeError):
                        feature_lines.append(f"- {col} = {mapped_value} - {desc}")
                else:
                    feature_lines.append(f"- {col} = {mapped_value} - {desc}")
            else:
                feature_lines.append(f"- {col} = {mapped_value}")
    
    instance_desc = f"""The model is making a prediction for a student.

Feature values:
{chr(10).join(feature_lines)}

"""
    return instance_desc

def describe_instance(row):
    """Generate instance description from row"""
    return create_instance_description_from_row(row)

def separate_features_and_protected_attributes(original_instance, gender_override=None, race_override=None):
    """
    Separate protected attributes from other features for clearer presentation.
    
    Parameters:
    - original_instance: pandas Series with feature values
    - gender_override: Optional override for gender (e.g., "male", "female" or numeric 0/1)
    - race_override: Optional override for race (e.g., "white", "black", "hispanic", etc. or numeric code)
    
    Returns:
        Tuple of (instance_desc_regular_features, protected_attributes_desc)
    """
    protected_attributes = ['gender', 'race1']
    
    # Separate features
    feature_cols = [col for col in original_instance.index if col not in 
                    ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target', 'target_law']]
    
    regular_feature_cols = [col for col in feature_cols if col not in protected_attributes]
    protected_feature_cols = [col for col in feature_cols if col in protected_attributes]
    
    # Create descriptions
    regular_instance_row = original_instance[regular_feature_cols]
    instance_desc_regular = describe_instance(regular_instance_row)
    
    # Create protected attributes description, using overrides if provided
    protected_desc = "PERSONAL INFORMATION (for personalized narrative):\n"
    if DATASET_INFO is not None:
        feature_df = DATASET_INFO.get("feature_description")
        for col in protected_feature_cols:
            # Use override if provided for this attribute
            if col == 'gender' and gender_override is not None:
                mapped_value = gender_override if isinstance(gender_override, str) else map_attribute_value(col, gender_override)
            elif col == 'race1' and race_override is not None:
                mapped_value = race_override if isinstance(race_override, str) else map_attribute_value(col, race_override)
            else:
                value = original_instance[col]
                mapped_value = map_attribute_value(col, value)
            
            feature_info = feature_df[feature_df['feature_name'] == col]
            if not feature_info.empty:
                desc = feature_info.iloc[0]['feature_desc']
                protected_desc += f"- {col}: {mapped_value} ({desc})\n"
            else:
                protected_desc += f"- {col}: {mapped_value}\n"
    else:
        for col in protected_feature_cols:
            # Use override if provided for this attribute
            if col == 'gender' and gender_override is not None:
                mapped_value = gender_override if isinstance(gender_override, str) else map_attribute_value(col, gender_override)
            elif col == 'race1' and race_override is not None:
                mapped_value = race_override if isinstance(race_override, str) else map_attribute_value(col, race_override)
            else:
                value = original_instance[col]
                mapped_value = map_attribute_value(col, value)
            protected_desc += f"- {col}: {mapped_value}\n"
    
    return instance_desc_regular, protected_desc

# ===== PROMPT TEMPLATES =====

PROMPT_PREAMBLE_SHAP = """
A machine learning model predicted that a student will FAIL the bar exam and therefore their application for university admission was REJECTED.

YOUR TASK: Translate the following technical information into a clear, non-technical narrative explanation that helps the student understand:
- Why the model rejected their application in specific terms of their features
- Which factors were most important in this decision
- How their specific situation compared to typical students

INFORMATION YOU WILL RECEIVE:
1. DATASET INFORMATION: Context about the dataset, target variable and ML task used to train the model
2. TECHNICAL EXPLANATION METHOD: How we measure feature importance (SHAP values)
3. STUDENT PROFILE: The student's specific feature values with comparisons to dataset averages and distributions
4. FEATURE IMPORTANCE ANALYSIS: SHAP values showing which features most influenced the decision
5. CLEAR INSTRUCTIONS: What narrative you should write
"""

PROMPT_PREAMBLE_CF = """
A machine learning model predicted that a student will FAIL the bar exam and therefore their application for university admission was REJECTED.

YOUR TASK: Summarize the following counterfactual scenarios into a clear, non-technical narrative explanation that helps the student understand:
- Why the model rejected their application
- Which factors were most important in this decision
- How their specific situation compared to typical students
- What changes would be needed to flip the prediction to "pass the exam" and get the application approved

INFORMATION YOU WILL RECEIVE:
1. DATASET INFORMATION: Context about the dataset, target variable and ML task used to train the model
2. COUNTERFACTUAL EXPLANATION: Information about what counterfactuals are and how to interpret them
3. STUDENT PROFILE: The student's specific feature values with comparisons to dataset averages and distributions, and the model's predicted probability of failure
4. COUNTERFACTUAL SCENARIOS TABLE: A table showing the original instance and multiple counterfactual scenarios with feature changes that would flip the prediction
5. CLEAR INSTRUCTIONS: What narrative you should write
"""

DATASET_EXPLANATION = """
1. DATASET INFORMATION
"""

APPLICANT_INFORMATION = """
3. STUDENT PROFILE 

You are writing a narrative tailored to this specific person who is rejected law school admission. 
"""

SHAP_VALUES_SECTION = """
4. FEATURE IMPORTANCE ANALYSIS (Ranked by Influence)
"""

INSTRUCTIONS_SECTION = """
5. YOUR NARRATIVE TASK
"""

COUNTERFACTUAL_EXPLANATION_DETAILS = """
2. COUNTERFACTUAL ANALYSIS (Alternative Scenarios)

You are given a table comparing the student's current situation (original) with alternative scenarios (counterfactuals cf_1, cf_2, etc.):

- The 'original' row shows the student's actual feature values and the model's actual prediction (will fail).
- Each 'cf_k' row shows what would happen if certain features changed - representing scenarios where the model WOULD predict passing.

In other words: Each counterfactual is a "what if" scenario showing the minimum feature changes needed to flip the model's prediction from "will fail" to "will pass". This helps answer: "What would need to be different for the application to be approved?"

Academic performance, study habits, and family support can vary. The counterfactuals show which combinations of changes would be most effective in improving the likelihood of passing.
"""

SHAP_EXPLANATION = """
2. TECHNICAL EXPLANATION: SHAP VALUES

You are given SHAP values for this student's prediction.

SHAP values explain how much each feature contributes to the model's prediction for this specific student.
Each feature has a SHAP value that tells you:
- How much that feature influenced the model's decision for this student.
- Whether it pushed the prediction toward "will fail" (positive contribution) or "will pass" (negative contribution).
- Larger absolute values indicate features with stronger influence on the prediction.

Features are ranked by their absolute SHAP values, with the most influential features listed first.
Features with positive SHAP values contributed toward a "will fail" prediction.
Features with negative SHAP values contributed toward a "will pass" prediction.
"""

SHAP_PROMPT_INSTRUCTIONS = """
TASK:
Your goal is to generate a textual explanation or narrative explaining why the law school application was denied for this student.

PERSONALIZATION INSTRUCTION:
Based on the PERSONAL INFORMATION (gender, race) about the student, create a personalized narrative tailored to them. 
The narrative should feel like it was written specifically for this individual, acknowledging their personal circumstances and creating a more personalized experience. 

Write a detailed narrative explanation tailored to this non-technical reader that MUST explain:
1) The current situation of the applicant (what are their features and background).
2) The model's predicted probability of bar exam failure and what this means for the student.
3) Why the application was denied and which features were most important in driving this prediction.
4) How each of the most important features contributed (either pushing toward bar exam failure or toward passing).
5) What the applicant should do next

CONSTRAINTS:
- Only use information you are given in this prompt.
- Do NOT invent new SHAP values or new numerical values.
- Do not use the numeric SHAP values in your answer. Instead, discuss the ranking and direction of influence.
- Do not talk about model internals, algorithms, or training details.

STYLE:
- Length: 12-15 sentences.
- Write a coherent narrative without bullet points or tables. The goal is to have a narrative/story.
- Directly address the student and provide PERSONALIZED insights tailored to THEIR situation.
- Do NOT copy-paste feature names, but instead incorporate them naturally in the narrative.
- Use the personal information (gender, race) to create a personalised narrative. 
"""

COUNTERFACTUAL_PROMPT_INSTRUCTIONS = """
TASK:
- You are given a table of counterfactuals for the same original instance.
- Summarize what the model considers important for changing the predicted bar exam class.
- Provide concrete, actionable insights about which feature changes would shift the prediction.
- Provide a numeric summary: which features always/never change, and by how much on average.

Write a detailed narrative explanation for a non-technical reader:
1) Briefly summarize the current situation of the applicant, the model's predicted probability of bar exam failure and what this means for the student.
2) Explain what counterfactuals represent: "what if" scenarios that would change the prediction.
3) Specify which features always changed (MUST changes) and which never changed.
4) Quantified summary: how many counterfactuals were generated, which features changed most often.
5) Describe which features changed together - this shows effective combination patterns.
6) End with actionable guidance: based on patterns, which features are realistic to change and by approximately how much.

CONSTRAINTS:
- Only use information from dataset description, instance description, and counterfactual table.
- Do NOT invent new feature values or examples.
- Do not talk about model internals or training details.
- Use realistic ranges when discussing feature changes.

STYLE:
- Length: 15-18 sentences.
- Write a coherent narrative without bullet points or tables.
- Directly address the student. 
- Focus on actionable insights the student can implement.
"""


def build_shap_prompt(instance_index, shap_csv_path: str = None, gender_override=None, race_override=None) -> str:
    """
    Build a SHAP explanation prompt by loading from the SHAP CSV.
    
    Parameters:
    - instance_index: the instance index to explain (e.g., 10, 25, etc.)
    - shap_csv_path: path to the SHAP CSV file (defaults to law_dataset/law_shap.csv)
    - gender_override: Optional override for gender ("male", "female", or numeric 0/1). For bias injection experiment.
    - race_override: Optional override for race ("white", "black", "hispanic", etc., or numeric code). For bias injection experiment.
    
    Returns:
    - Full prompt string ready for LLM
    """
    if shap_csv_path is None:
        shap_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "law_dataset" / "law_shap.csv"
    
    # Load SHAP values (instance_index is now an explicit column)
    shap_df = pd.read_csv(shap_csv_path)
    shap_row = shap_df[shap_df['instance_index'] == instance_index]
    
    if shap_row.empty:
        raise ValueError(f"Instance {instance_index} not found in SHAP CSV")
    
    shap_values = shap_row.iloc[0]
    
    # Extract predicted_probability
    predicted_probability = shap_values.get('predicted_probability', np.nan)
    
    # Load corresponding original data
    test_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "law_dataset" / "law_adverse.csv"
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
    
    # Separate regular features from protected attributes, using overrides if provided
    instance_desc_regular, protected_desc = separate_features_and_protected_attributes(original_instance, gender_override=gender_override, race_override=race_override)
    
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
Academic Performance & Background:
{instance_desc_regular}

{protected_desc}

The model's prediction:
- Predicted probability of failure: {pred_prob_str}

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
    - cf_csv_path: path to counterfactual CSV (defaults to law_dataset/law_counterfactual.csv)
    - adverse_csv_path: path to adverse CSV (defaults to law_dataset/law_adverse.csv)
    - shap_csv_path: path to SHAP CSV for predicted_probability (defaults to law_dataset/law_shap.csv)
    
    Returns:
    - Full prompt string ready for LLM
    """
    if cf_csv_path is None:
        cf_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "law_dataset" / "law_counterfactual.csv"
    if adverse_csv_path is None:
        adverse_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "law_dataset" / "law_adverse.csv"
    if shap_csv_path is None:
        shap_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "law_dataset" / "law_shap.csv"
    
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
    # Extract feature columns only (exclude metadata like instance_index, original_test_index, CF_number, distance_to_original, and target)
    feature_cols = [col for col in adverse_df.columns 
                   if col not in ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target', 'target_law']]
    
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
    
    # Create instance description (excluding metadata and target)
    instance_features = original[[c for c in original.index if c not in 
                                   ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target', 'target_law']]]
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
- Predicted probability of failure: {pred_prob_str}

4. COUNTERFACTUAL SCENARIOS TABLE


{table_str}

{INSTRUCTIONS_SECTION}
{COUNTERFACTUAL_PROMPT_INSTRUCTIONS}
"""
    return prompt


# ===== ALTERNATIVE NARRATIVE-FOCUSED SHAP PROMPT =====

PROMPT_PREAMBLE_SHAP_NARRATIVE = """
A machine learning model has made a prediction about a student's likelihood of passing the bar exam. The model predicts failure, resulting in a rejected law school application.

YOUR TASK: Write a clear, persuasive narrative that explains this decision to the student. The narrative should feel like a personal story, not a technical report. Use the student's actual characteristics and the model's reasoning to craft a plausible explanation.

INFORMATION YOU WILL RECEIVE:
1. The student's profile: their actual feature values (grades, family income, study habits, demographics, etc.)
2. The model's prediction: probability of bar exam failure
3. Feature importance ranking: which aspects of the student's profile had the strongest influence on this prediction, ranked from most to least important
"""

SHAP_PROMPT_INSTRUCTIONS_NARRATIVE = """
TASK:
Write a personalized narrative explanation for why this student's application was rejected. Frame it as if you are explaining the model's reasoning in natural language.
Write in a way that feels personal and grounded, not formulaic. Reference the student's specific numbers and attributes. Show how the combination of factors painted a particular picture for the model.

CONSTRAINTS:
- Only use information provided in this prompt
- Do not use the numeric SHAP values themselves; instead rank and discuss influence direction (e.g., "This was the strongest factor," "This pulled toward failure")
- Do not explain how machine learning works or mention model internals

STYLE:
- Length: 12-15 sentences
- Write as a coherent narrative paragraph(s), not a bulleted list
- Directly adress the student and provide insights tailored to their specific situation
"""

INSTANCE_DESCRIPTION_SIMPLE = """
THE STUDENT'S PROFILE
"""

def create_simple_instance_description(row):
    """
    Create a simple instance description without dataset averages or distributions.
    Just lists the student's feature values with brief descriptions.
    
    Parameters:
    - row: pandas Series with feature values
    """
    feature_lines = []
    
    if DATASET_INFO is not None:
        feature_df = DATASET_INFO.get("feature_description")
        for col in row.index:
            value = row[col]
            # Map attribute value if applicable
            mapped_value = map_attribute_value(col, value)
            
            # Find feature description
            feature_info = feature_df[feature_df['feature_name'] == col]
            if not feature_info.empty:
                desc = feature_info.iloc[0]['feature_desc']
                feature_lines.append(f"- {col}: {mapped_value} ({desc})")
            else:
                feature_lines.append(f"- {col}: {mapped_value}")
    else:
        # Fallback
        for col in row.index:
            value = row[col]
            mapped_value = map_attribute_value(col, value)
            feature_lines.append(f"- {col}: {mapped_value}")
    
    instance_desc = f"""
{chr(10).join(feature_lines)}
"""
    return instance_desc


def build_shap_prompt_narrative(instance_index, shap_csv_path: str = None, adverse_csv_path: str = None) -> str:
    """
    Build a narrative-focused SHAP explanation prompt.
    
    This variant emphasizes storytelling over technical explanation. Features and SHAP values
    are presented without dataset context, allowing the LLM to generate a plausible narrative
    explaining the rejection based on the student's individual characteristics.
    
    Parameters:
    - instance_index: the instance index to explain (e.g., 10, 25, etc.)
    - shap_csv_path: path to the SHAP CSV file (defaults to law_dataset/law_shap.csv)
    - adverse_csv_path: path to adverse CSV (defaults to law_dataset/law_adverse.csv)
    
    Returns:
    - Full prompt string ready for LLM
    """
    if shap_csv_path is None:
        shap_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "law_dataset" / "law_shap.csv"
    if adverse_csv_path is None:
        adverse_csv_path = Path(__file__).parent.parent.parent / "datasets_prep" / "data" / "law_dataset" / "law_adverse.csv"
    
    # Load SHAP values
    shap_df = pd.read_csv(shap_csv_path)
    shap_row = shap_df[shap_df['instance_index'] == instance_index]
    
    if shap_row.empty:
        raise ValueError(f"Instance {instance_index} not found in SHAP CSV")
    
    shap_values = shap_row.iloc[0]
    predicted_probability = shap_values.get('predicted_probability', np.nan)
    
    # Load original instance
    adverse_df = pd.read_csv(adverse_csv_path)
    adverse_row = adverse_df[adverse_df['instance_index'] == instance_index]
    
    if adverse_row.empty:
        raise ValueError(f"Instance {instance_index} not found in adverse CSV")
    
    original_instance = adverse_row.iloc[0]
    
    # Extract SHAP values (remove instance_index and SHAP_ prefix)
    shap_dict = {}
    for col in shap_values.index:
        if col.startswith('SHAP_'):
            feature_name = col[5:]  # Remove 'SHAP_' prefix
            shap_dict[feature_name] = shap_values[col]
    
    # Create SHAP ranking (by absolute value)
    shap_ranking = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Format ranking as text (include direction but not exact numeric values)
    ranking_lines = []
    for i, (feature, value) in enumerate(shap_ranking, 1):
        direction = "toward failure" if value > 0 else "toward passing"
        ranking_lines.append(f"{i}. {feature} (influence: {direction})")
    
    ranking_text = chr(10).join(ranking_lines)
    
    # Create simple instance description (just features without dataset context)
    feature_cols = [col for col in original_instance.index if col not in 
                    ['instance_index', 'original_test_index', 'predicted_class', 'prediction_score', 'actual_target']]
    instance_row = original_instance[feature_cols]
    instance_desc = create_simple_instance_description(instance_row)
    
    # Format predicted_probability
    pred_prob_str = f"{predicted_probability:.1%}" if not np.isnan(predicted_probability) else "N/A"
    
    prompt = f"""{PROMPT_PREAMBLE_SHAP_NARRATIVE}

{INSTANCE_DESCRIPTION_SIMPLE}
{instance_desc}

MODEL PREDICTION:
- Predicted probability of bar exam failure: {pred_prob_str}

FEATURE IMPORTANCE RANKING (Most to Least Influential):
{ranking_text}

NARRATIVE TASK:
{SHAP_PROMPT_INSTRUCTIONS_NARRATIVE}
"""
    return prompt
