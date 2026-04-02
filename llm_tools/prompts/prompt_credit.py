DATASET_DESCRIPTION = """
You are working with a binary classification model that predicts whether a person has an annual income above or below 50K USD.
The dataset consists of adult individuals from the US Census.

Each row in the data is one person. The target variable is:
- income = 0: predicted income <= 50K (50,000 USD)
- income = 1: predicted income > 50K (50,000 USD)

The features are:
- age: Age of the individual in years (continuous).
- education-num: The highest level of education achieved, represented as a numerical ordinal value:
    1: Preschool, 2: 1st-4th, 3: 5th-6th, 4: 7th-8th, 5: 9th, 
    6: 10th, 7: 11th, 8: 12th, 9: High School Grad, 10: Some College, 
    11: Associate Degree (Vocational), 12: Associate Degree (Academic), 
    13: Bachelor's, 14: Master's, 15: Professional School, 16: Doctorate.
- hours-per-week: The average number of hours worked per week (continuous).
- capital-gain: Annual income from investment sources (capital gains) in USD (continuous).
- capital-loss: Annual loss from investment sources (capital losses) in USD (continuous).
- workclass: The type of employment or employment status, represented as a numerical value:
    0: Other/Unknown (missing data)
    1: Federal government
    2: Local government
    3: Never worked
    4: Private sector
    5: Self-employed with incorporation
    6: Self-employed without incorporation
    7: State government
    8: Without pay
- occupation: The primary occupation or job type, represented as a numerical value:
    0: Other/Unknown (missing data)
    1: Administrative and clerical
    2: Armed forces
    3: Craft and repair
    4: Executive and managerial
    5: Farming and fishing
    6: Handlers and cleaners
    7: Machine operators and inspectors
    8: Other service
    9: Private household service
    10: Professional specialty
    11: Protective services
    12: Sales
    13: Technical support
    14: Transportation and moving
- sex: Biological sex, where 0 = Female and 1 = Male.
- race: The individual's reported racial/ethnic background, represented as a numerical value:
    0: American Indian or Alaska Native
    1: Asian or Pacific Islander
    2: Black or African American
    3: Other
    4: White
- married: A binary indicator derived from marital status.
    1 = Married with spouse present (includes "Married-civ-spouse" and "Married-AF-spouse").
    0 = Not currently married (includes "Never-married", "Divorced", "Separated", "Widowed", and "Married-spouse-absent").
- relationship: The family relationship of the individual, represented as a numerical value:
    0: Husband
    1: Not in family
    2: Other relative
    3: Own child
    4: Unmarried partner
    5: Wife
- native-country: The country of origin or residence, represented as a numerical value:
    0: Other/Unknown (missing data)
    Countries are numbered from 1 onwards (e.g., 39: United States, 2: Canada, 3: China, etc.)

In this project, age, sex, and race are considered fixed characteristics that cannot be changed.
All other features are considered potentially changeable in the counterfactuals.
"""


INSTANCE_DESCRIPTION_TEMPLATE = """
The model is making a prediction for a single person.

For this person:
- age = {age}
- education-num = {education_num}
- hours-per-week = {hours_per_week}
- capital-gain = {capital_gain}
- capital-loss = {capital_loss}
- workclass = {workclass_label}
- occupation = {occupation_label}
- sex = {sex_label}
- race = {race_label}
- married = {married_label}
- relationship = {relationship_label}
- native-country = {native_country_label}

The model's prediction for this person is:
- income = {income_pred}  (0 = income <= 50K, 1 = income > 50K)
"""


def describe_instance(row):
    sex_label = "male" if row["sex"] == 1 else "female"
    married_label = "married" if row["married"] == 1 else "not married"
    
    race_mapping = {
        0: "American Indian or Alaska Native",
        1: "Asian or Pacific Islander",
        2: "Black or African American",
        3: "Other",
        4: "White"
    }
    race_label = race_mapping.get(int(row["race"]), "Unknown")
    
    workclass_mapping = {
        0: "Unknown", 1: "Federal government", 2: "Local government", 
        3: "Never worked", 4: "Private", 5: "Self-employed (inc)", 
        6: "Self-employed (no inc)", 7: "State government", 8: "Without pay"
    }
    workclass_label = workclass_mapping.get(int(row["workclass"]), "Unknown")
    
    occupation_mapping = {
        0: "Unknown", 1: "Admin/Clerical", 2: "Armed Forces", 3: "Craft/Repair",
        4: "Executive", 5: "Farming", 6: "Handlers", 7: "Machine ops",
        8: "Service", 9: "Household service", 10: "Professional", 11: "Protective",
        12: "Sales", 13: "Tech support", 14: "Transportation"
    }
    occupation_label = occupation_mapping.get(int(row["occupation"]), "Unknown")
    
    relationship_mapping = {
        0: "Husband", 1: "Not in family", 2: "Other relative",
        3: "Own child", 4: "Unmarried partner", 5: "Wife"
    }
    relationship_label = relationship_mapping.get(int(row["relationship"]), "Unknown")
    
    # Simple country mapping (most common countries)
    country_mapping = {
        0: "Unknown", 1: "Cambodia", 2: "Canada", 3: "China", 4: "Columbia",
        5: "Cuba", 6: "Dominican-Republic", 7: "Ecuador", 8: "El-Salvador",
        9: "England", 10: "France", 11: "Germany", 12: "Greece", 13: "Guatemala",
        14: "Haiti", 15: "Holand-Netherlands", 16: "Honduras", 17: "Hong Kong",
        18: "Hungary", 19: "India", 20: "Iran", 21: "Ireland", 22: "Italy",
        23: "Jamaica", 24: "Japan", 25: "Laos", 26: "Mexico", 27: "Nicaragua",
        28: "Outlying-US", 29: "Peru", 30: "Philippines", 31: "Poland",
        32: "Portugal", 33: "Puerto-Rico", 34: "Scotland", 35: "South",
        36: "Taiwan", 37: "Thailand", 38: "Trinidad&Tobago", 39: "United-States",
        40: "Vietnam", 41: "Yugoslavia"
    }
    native_country_label = country_mapping.get(int(row["native-country"]), "Unknown")

    return INSTANCE_DESCRIPTION_TEMPLATE.format(
        age=int(row["age"]),
        education_num=int(row["education-num"]),
        hours_per_week=float(row["hours-per-week"]),
        capital_gain=float(row["capital-gain"]),
        capital_loss=float(row["capital-loss"]),
        workclass_label=workclass_label,
        occupation_label=occupation_label,
        sex_label=sex_label,
        race_label=race_label,
        married_label=married_label,
        relationship_label=relationship_label,
        native_country_label=native_country_label,
        income_pred=int(row["income"]),
    )



COUNTERFACTUAL_EXPLANATION = """
You are also given a small table with one row called 'original' and several rows called 'cf_1', 'cf_2', etc.

- The 'original' row shows this person's current feature values and the model's current prediction.
- Each 'cf_k' row is a counterfactual version of this person:
  it shows a change to some of the features that would flip the model's prediction to the opposite class.
  Refer to cf_k as "counterfactual k" in your explanation.

In other words:
- The original row corresponds to the current situation.
- Each counterfactual row corresponds to a "what if" scenario where some change in the person's situation would be enough for the model to predict an income greater than 50K instead of smaller or equal than 50K (or vice versa).

The features age, sex, and race must be treated as fixed personal attributes.
They are not allowed to change in the counterfactuals and you must not suggest changing them.
You do not need to mention the unchangeability of age, sex, or race. 
Changes are only allowed in education-num, hours-per-week, capital-gain, capital-loss, married, workclass, occupation, relationship, or native-country.
Refer to these features as "education level", "hours worked per week", "capital gains", "capital losses", "marital status", "type of employment", "occupation", "family relationship", and "country of residence" in your explanation.
Or use similar clear, non-technical language.
"""


SHAP_EXPLANATION = """
You are also given SHAP values for this person's prediction.

SHAP values are a way to understand how much each feature contributes to the model's prediction for this specific person.
Think of it as a breakdown showing which features pushed the prediction up or down compared to the average prediction.

Each feature has a SHAP value that tells you:
- How much that feature influenced the model's decision for this person.
- Whether it pushed the prediction toward income > 50K (positive contribution) or income <= 50K (negative contribution).
- The larger the absolute value, the more important that feature was for this particular prediction.

For this person:
- The base value (average prediction) is the starting point.
- Each feature's SHAP value is added or subtracted from this base.
- The sum of all SHAP values plus the base value equals the model's final prediction for this person.

Features with large positive SHAP values are the main reasons the model predicted income > 50K.
Features with large negative SHAP values are the main reasons the model predicted income <= 50K.

The features age, sex, and race are treated as fixed characteristics and cannot be changed to improve the prediction.
"""


SHAP_PROMPT_INSTRUCTIONS = """
TASK:
- You are given SHAP values that explain the model's prediction for this specific person.
- It is your job to summarize which features had the strongest influence on the prediction.
- Provide an end user with a clear understanding of why the model made this prediction.
- Identify which features pushed the prediction in the positive direction (toward income > 50K) and which pushed it in the negative direction (toward income <= 50K).
- You must talk in terms of feature importance and directional influence. Quantify the relative strength of each feature's contribution.

Write a detailed narrative explanation for a non-technical reader that explains:
1) The current prediction for this person and what their overall situation looks like.
2) The base value (average prediction) and how this person's features compare to that baseline.
3) Which features had the strongest positive influence on the prediction (pushing it toward income > 50K). Explain why these features are important.
4) Which features had the strongest negative influence on the prediction (pushing it toward income <= 50K). Explain why these features are important.
5) A summary of the overall story: which factors are the primary drivers of this prediction, and how they interact.

CONSTRAINTS:
- Only use information from the dataset description, the original instance description, and the SHAP values.
- Do NOT invent new SHAP values or new feature contributions.
- Do NOT suggest that age, sex, or race should be changed. 
- Do not talk about model internals, algorithms, probabilities, loss functions, or training details.
- Rank features by the absolute magnitude of their SHAP values when describing importance.

STYLE:
- Length: aim for around 12-15 sentences.
- Tone: neutral, explanatory, patient, and educational.
- Focus on translating SHAP values into human-understandable feature importance rankings.
- Do not use bullet points or tables; write a coherent, flowing narrative.
- Don't refer to categories of features as their number, but use real-world names. For example, say "PhD" instead of education level 13.
- You are directly talking to the person in the original instance. Be polite, empathic, and respectful.
- Emphasize that this explanation shows what the model learned, not a judgment of the person.

"""

MANY_CF_PROMPT_INSTRUCTIONS = """
TASK:
- You are given a table of counterfactuals for the same original instance.
- It is your job to summarize what the model seems to consider important for changing the prediction. If a change to a feature in order to change the target class is counterintuitive and seems wrong, don't mention it. 
- Provide an end user with actionable insights that do change the predicted class.
- At the same time, provide a numeric summary of the counterfactuals. This means saying how many counterfactuals changed which features, and by how much on average.
- You must talk in explicit percentages (in %) and averages where relevant. Don't refer to individual counterfactuals, except for in the last phrase.

Write a detailed narrative explanation for a non-technical reader that explains:
1) The current prediction for the original person and what their situation looks like.
2) Quantified summary over all counterfactuals: mention how many total counterfactuals the were generated,
and extract the most relevant features. If relevant, include which features were often changed together as this might contain extra information. 
3) Based on the situation of the current person and the quantified summary, provide the best actionable insights for this person to actually change the predicted class. 
Make sure that your suggestions change the target class and explicitely mention that following these suggestions does change the target class. 
4) Provide, based on your insights, the best counterfactual given this context in the last phrase. Explicitely refer to this counterfactual.  

CONSTRAINTS:
- Only use information from the dataset description, the original instance description, and the counterfactual table.
- Do NOT invent new feature values or new counterfactual examples.
- Do NOT suggest changing age, sex, or race; mention explicitly that these are treated as fixed characteristics if relevant.
- Only discuss changes in education-num, hours-per-week, capital-gain, capital-loss, married, workclass, occupation, relationship, or native-country.
- If several counterfactuals show a similar pattern (e.g. education-num is always higher), highlight that pattern explicitly.
- Do not talk about model internals, algorithms, probabilities, loss functions, or training details.
- DO NOT refer to individual counterfactuals, except for the last phrase; focus on the overall patterns you see in the data.
- Use a maximum of 20 sentences. 

STYLE:
- Length: aim for around 10-15 sentences.
- Tone: neutral, explanatory, and patient.
- Focus on making the link between the specific numbers in the table and the qualitative story.
- Do not use bullet points or tables; write a coherent, flowing narrative.
- Do not refer to cf_k in technical terms; refer to them as "counterfactual k".
- Don't refer to categories of features as their number, but use real-world names. For example, say "PhD" instead of education level 13. 
- You are directly talking to the person in the original instance. Be polite, empathic, and respectful.

"""

NEW_CF_PROMPT_INSTRUCTIONS = """
TASK:
- You are given a table of counterfactuals for the same original instance.
- It is your job to summarize what the model seems to consider important for changing the predicted target class. 
- The end user will use this narrative (summary) and interact with it to find out which changed work best for them to alter the predicted class. 
- It's very important to keep in mind that the user will use the provided information to base their preferences on. 
- Provide an end user with concrete actionable insights that do change the predicted class.
- At the same time, provide a numeric summary of the counterfactuals. This means saying how many counterfactuals changed which features and by how much on average. 
- You must talk in explicit percentages (in %) and averages where relevant. Don't refer to individual counterfactuals. 

Write a detailed narrative explanation for a non-technical reader that explains:
1) Start by shortly going over the current prediction for the original instance and what their situation looks like.
2) Explain in one phrase what it means that the counterfactuals were generated. You are talking to lay-people. 
3) Mention briefly, if applicable, which features always changed in the counterfactuals, and which features never changed. Be explicit about this and convey to the user that they HAVE (or not) to change this feature in order to flip the predicted class. 
4) Quantified summary over all counterfactuals: mention how many total counterfactuals the were generated, and extract the most relevant features. 
5) Specifically mention which features always (never) change, be clear that these are MUST (NOT) changes.
6) Specifically mention which features always or often changed together in order to change the predicted class.  
7) Finally, ask the end user based on the summary which features they seem fit to change and, in the case of continuous features, by how much. 

CONSTRAINTS:
- Only use information from the dataset description, the original instance description, and the counterfactual table.
- Do NOT invent new feature values or new counterfactual examples.
- Do NOT suggest changing age, sex, or race; mention explicitly that these are treated as fixed characteristics if relevant.
- Only discuss changes in education-num, hours-per-week, capital-gain, capital-loss, married, workclass, occupation, relationship, or native-country.
- If several counterfactuals show a similar pattern (e.g. education-num is always higher), highlight that pattern explicitly.
- Do not talk about model internals, algorithms, probabilities, loss functions, or training details.
- DO NOT refer to individual counterfactuals, focus on the overall patterns you see in the data.
- Use a maximum of 20 sentences. 

STYLE:
- Length: aim for around 15 sentences.
- Tone: neutral, explanatory, and patient.
- Focus on making the link between the specific numbers in the table and the qualitative story.
- Do not use bullet points or tables; write a coherent, flowing narrative.
- Do not refer to cf_k in technical terms; refer to them as "counterfactual k".
- Don't refer to categories of features as their number, but use real-world names. For example, say "PhD" instead of education level 13. 
- You are directly talking to the person in the original instance. Be polite, empathic, and respectful. Use "you" statements.
- You are expecting a user prompt including user preferences after reading your narrative. Keep this in mind when writing the narrative, but don't mention it explicitly.

"""
PROMPT_INSTRUCTIONS = {
    "many": MANY_CF_PROMPT_INSTRUCTIONS,  
    "new": NEW_CF_PROMPT_INSTRUCTIONS,
}
def build_cf_prompt(cf_table, prompt_type: str = "many") -> str:
    """
    Build a counterfactual prompt for a given cf_table and prompt type.

    prompt_type must be one of the keys in PROMPT_INSTRUCTIONS
    """
    if prompt_type not in PROMPT_INSTRUCTIONS:
        raise ValueError(
            f"Unknown prompt_type '{prompt_type}'. "
            f"Available types: {list(PROMPT_INSTRUCTIONS.keys())}"
        )

    # original row
    original = cf_table.loc["original"]
    instance_desc = describe_instance(original)

    # counterfactual table as Markdown
    table_str = cf_table.to_markdown(index=True)

    instructions = PROMPT_INSTRUCTIONS[prompt_type]

    prompt = f"""
{DATASET_DESCRIPTION}

{instance_desc}

{COUNTERFACTUAL_EXPLANATION}

Here is the table with the original instance and its counterfactuals:

{table_str}

{instructions}
    """.strip()

    return prompt
