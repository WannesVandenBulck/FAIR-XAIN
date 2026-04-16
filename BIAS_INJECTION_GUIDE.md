# Bias Injection Experiment Guide

## Overview

This guide explains how to run the bias injection experiment to test for LLM narration bias based on protected attributes (gender and race). The experiment generates narratives for the same instances with different protected attribute values to compare how the LLM's tone and sentiment changes.

## Experiment Design

The bias injection experiment compares LLM-generated narratives across three conditions:

### Batch 1: Baseline (Original Attributes)
- Uses the real protected attributes from the law_adverse.csv dataset
- Reference condition to understand original LLM behavior
- Expected composition: Mix of genders and races as they appear in data

### Batch 2: All White Males
- Override all instances to appear as **white males** 
- Tests if LLM provides different narratives when attributes are masculine + white
- 433 instances, all with: gender="male", race="white"

### Batch 3: All Black Females
- Override all instances to appear as **black females**
- Tests if LLM provides different narratives when attributes are feminine + black
- 433 instances, all with: gender="female", race="black"

After generating all three batches, you can compare sentiment, tone, empathy, and other narrative characteristics using the `compare_narratives.py` script with the HuggingFace sentiment models.

## Quick Start

### Step 1: Generate Baseline Narratives (Optional Reference)

Generate narratives with original attributes (no overrides) for a subset to use as reference:

```bash
python scripts/bias_injection_experiment.py \
  --dataset law \
  --all-instances \
  --provider grok \
  --model grok-3-mini \
  --batch-name baseline \
  --output-dir results/narratives
```

This will generate narratives saved in:
```
results/narratives/law/narratives/shap/grok_baseline/grok-3-mini/instance_*.json
```

### Step 2: Generate White Male Batch

Generate narratives with all instances as white males:

```bash
python scripts/bias_injection_experiment.py \
  --dataset law \
  --all-instances \
  --provider grok \
  --model grok-3-mini \
  --gender-override male \
  --race-override white \
  --batch-name white_male \
  --output-dir results/narratives
```

Output saved in:
```
results/narratives/law/narratives/shap/grok_white_male/grok-3-mini/instance_*.json
```

### Step 3: Generate Black Female Batch

Generate narratives with all instances as black females:

```bash
python scripts/bias_injection_experiment.py \
  --dataset law \
  --all-instances \
  --provider grok \
  --model grok-3-mini \
  --gender-override female \
  --race-override black \
  --batch-name black_female \
  --output-dir results/narratives
```

Output saved in:
```
results/narratives/law/narratives/shap/grok_black_female/grok-3-mini/instance_*.json
```

## Advanced Usage

### Test with Specific Instances First

Before running all 433 instances, test with a smaller subset:

```bash
python scripts/bias_injection_experiment.py \
  --dataset law \
  --instances 0 1 2 3 4 5 6 7 8 9 \
  --provider grok \
  --model grok-3-mini \
  --gender-override male \
  --race-override white \
  --batch-name white_male_test
```

### Use Different LLM Providers

```bash
# Test with OpenAI
python scripts/bias_injection_experiment.py \
  --dataset law --all-instances \
  --provider openai --model gpt-4o \
  --gender-override male --race-override white \
  --batch-name white_male_openai

# Test with Anthropic
python scripts/bias_injection_experiment.py \
  --dataset law --all-instances \
  --provider anthropic --model claude-3-opus-20240229 \
  --gender-override male --race-override white \
  --batch-name white_male_claude
```

### Dry-Run Mode

Preview what would be done without calling the LLM:

```bash
python scripts/bias_injection_experiment.py \
  --dataset law --all-instances \
  --gender-override male --race-override white \
  --batch-name white_male \
  --dry-run
```

## Analyzing Results

### Using Direct Command Line

After generating narratives, analyze a single batch:

```bash
python scripts/make_narratives.py \
  --dataset law \
  --prompt-type shap \
  --all-instances \
  --gender-override male \
  --race-override white \
  --output-dir results_comparison_white_male
```

### Using Compare Script (Recommended)

Modify [scripts/compare_narratives.py](../scripts/compare_narratives.py) to load from the bias experiment directories:

```python
# Edit compare_narratives.py USER CONFIGURATION section:

# Load narratives from different batches
NARRATIVE_DIR_1 = "results/narratives/law/narratives/shap/grok_baseline/grok-3-mini/"
NARRATIVE_DIR_2 = "results/narratives/law/narratives/shap/grok_white_male/grok-3-mini/"
NARRATIVE_DIR_3 = "results/narratives/law/narratives/shap/grok_black_female/grok-3-mini/"

# Run sentiment analysis
# The script will analyze tone, empathy, and sentiment across the three conditions
```

## Output Directory Structure

After running all batches, your results will be organized as:

```
results/narratives/law/narratives/shap/
├── grok_baseline/
│   └── grok-3-mini/
│       ├── instance_0.json
│       ├── instance_1.json
│       └── ... (433 total)
├── grok_white_male/
│   └── grok-3-mini/
│       ├── instance_0.json
│       ├── instance_1.json
│       └── ... (433 total)
└── grok_black_female/
    └── grok-3-mini/
        ├── instance_0.json
        ├── instance_1.json
        └── ... (433 total)
```

## What Gets Stored in Each JSON File

Each generated narrative includes:

```json
{
  "dataset": "law",
  "instance_idx": 0,
  "prompt_type": "shap",
  "provider": "grok",
  "model": "grok-3-mini",
  "bias_batch": "white_male",
  "gender_override": "male",
  "race_override": "white",
  "narrative": "The generated narrative text...",
  "status": "success",
  "timestamp": "2024-04-16T12:34:56.789012"
}
```

## Interpreting Results

After analysis with sentiment models, look for:

1. **Tone Differences**: Does the LLM use warmer language for certain demographic groups?
2. **Empathy Variations**: Are narratives more empathetic to some groups than others?
3. **Blame Attribution**: Does the LLM frame failures differently based on protected attributes?
4. **Recommendations**: Are suggestions for improvement more hopeful/actionable for some groups?

## Protected Attribute Values

Valid values for gender and race overrides:

### Gender Values
- `male` → 1 in the data
- `female` → 0 in the data

### Race Values
- `white` → 0 in the data (white/Caucasian)
- `black` → 1 in the data (Black/African American)
- `hispanic` → 2 in the data
- `asian` → 3 in the data
- `native american` → 4 in the data

## Troubleshooting

### Q: "Instance not found" errors
**A**: Ensure all instances are available in law_adverse.csv. The experiment expects valid instance indices 0-432.

### Q: API Rate Limiting
**A**: If using OpenAI/Anthropic and hitting rate limits, add delays or reduce batch size:
```bash
# Process in smaller batches
python scripts/bias_injection_experiment.py \
  --dataset law \
  --instances 0 1 2 3 4 5 \
  --batch-name white_male_batch1
```

### Q: How to resume partial batches?
**A**: The script skips instances that already exist, so you can safely re-run the same command if it was interrupted.

### Q: Can I use custom protected attribute mapping?
**A**: Currently, gender and race are hardcoded to use the standard mappings. To use different mappings, modify the ATTRIBUTE_VALUE_MAPPINGS in [llm_tools/prompts/prompt_law.py](../llm_tools/prompts/prompt_law.py).

## Example: Complete Workflow

```bash
# 1. Generate baseline with first 50 instances (for testing)
python scripts/bias_injection_experiment.py \
  --dataset law --instances 0-49 \
  --batch-name baseline_test

# 2. Generate white male batch (full)
python scripts/bias_injection_experiment.py \
  --dataset law --all-instances \
  --gender-override male --race-override white \
  --batch-name white_male

# 3. Generate black female batch (full)
python scripts/bias_injection_experiment.py \
  --dataset law --all-instances \
  --gender-override female --race-override black \
  --batch-name black_female

# 4. Analyze with sentiment models
python scripts/compare_narratives.py  # (modify to load bias experiment batches)
```

## Technical Details

### How Bias Injection Works

1. The `build_shap_prompt()` function now accepts `gender_override` and `race_override` parameters
2. When overrides are provided, they replace the protected attributes from the original data
3. The SHAP values and prediction probabilities remain the same (same model predictions)
4. Only the demographics presented to the LLM change
5. Different narratives indicate the LLM is responding to demographic cues

### Architecture

- **[scripts/bias_injection_experiment.py](../scripts/bias_injection_experiment.py)**: Main orchestration script
- **[scripts/make_narratives.py](../scripts/make_narratives.py)**: Updated to support overrides
- **[llm_tools/prompts/prompt_law.py](../llm_tools/prompts/prompt_law.py)**: Updated functions with override support
  - `build_shap_prompt(instance_idx, gender_override, race_override)`
  - `separate_features_and_protected_attributes(row, gender_override, race_override)`

## Next Steps

After running the bias injection experiment:

1. Use `compare_narratives.py` to run HuggingFace sentiment models (DistilRoBERTa, Go Emotions, Empathy Classifier)
2. Aggregate sentiment scores by demographic batch
3. Statistical testing to determine if differences are significant
4. Document findings in a bias report
5. Consider additional experiments with other demographic attributes or LLM models

---

**Last Updated**: April 16, 2026
