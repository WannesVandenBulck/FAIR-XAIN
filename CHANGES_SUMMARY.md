# Summary of Changes for Bias Injection Experiment

## Overview
Modified the narrative generation pipeline to support gender and race attribute overrides for conducting bias injection experiments. This enables testing whether LLM-generated narratives exhibit bias based on protected demographic attributes.

## Files Modified

### 1. [llm_tools/prompts/prompt_law.py](llm_tools/prompts/prompt_law.py)

#### Function: `separate_features_and_protected_attributes()`
**Before**: Only extracted protected attributes from instance data
```python
def separate_features_and_protected_attributes(original_instance):
```

**After**: Accepts optional gender and race overrides
```python
def separate_features_and_protected_attributes(original_instance, gender_override=None, race_override=None):
```

**New Behavior**:
- If `gender_override` is provided (e.g., "male", "female"), uses that instead of original gender value
- If `race_override` is provided (e.g., "white", "black"), uses that instead of original race value
- Accepts both string values and numeric codes for overrides
- Maintains all other prompt structure and formatting

#### Function: `build_shap_prompt()`
**Before**: Only accepted instance index and CSV path
```python
def build_shap_prompt(instance_index, shap_csv_path: str = None) -> str:
```

**After**: Accepts optional gender and race overrides
```python
def build_shap_prompt(instance_index, shap_csv_path: str = None, gender_override=None, race_override=None) -> str:
```

**New Behavior**:
- Passes `gender_override` and `race_override` to `separate_features_and_protected_attributes()`
- Generates prompts with modified demographic information while keeping all other features identical
- Preserves SHAP values and prediction probabilities (only demographics change)

### 2. [scripts/make_narratives.py](scripts/make_narratives.py)

#### Function: `generate_narrative()`
**Added Parameters**:
- `gender_override=None`: Override gender for all instances
- `race_override=None`: Override race for all instances

**Changes**:
- Passes overrides through to `build_shap_prompt()`
- Stores override information in the result JSON for tracking
- Works with all prompt types (shap, narrative, cf)

#### Main Function: Command-line Arguments
**New Arguments**:
```
--gender-override TEXT     Override gender for all instances (e.g., 'male', 'female')
--race-override TEXT       Override race for all instances (e.g., 'white', 'black', 'hispanic')
```

**Example Usage**:
```bash
python scripts/make_narratives.py \
  --dataset law \
  --prompt-type shap \
  --all-instances \
  --gender-override male \
  --race-override white \
  --provider grok \
  --model grok-3-mini
```

**Output in Result Directory**:
- Each JSON file includes `gender_override` and `race_override` fields for tracking

### 3. [scripts/bias_injection_experiment.py](scripts/bias_injection_experiment.py) **NEW FILE**

A specialized orchestration script for managing bias injection experiment batches.

**Key Features**:
- Simplified workflow for running multiple bias experiment batches
- Organizes outputs by batch name in directory structure
- Built-in batch naming convention: `grok_{batch_name}` 
- Comprehensive progress tracking and summaries

**Main Function**: `run_bias_experiment_batch()`
- Parameters for dataset, instances, LLM provider/model, overrides, and batch name
- Output structure: `results/narratives/{dataset}/narratives/{prompt_type}/grok_{batch_name}/{model}/`
- Prints summary statistics for each batch

**Command-line Usage**:
```bash
# Generate white male batch for all law instances
python scripts/bias_injection_experiment.py \
  --dataset law \
  --all-instances \
  --gender-override male \
  --race-override white \
  --batch-name white_male

# Generate black female batch for all law instances
python scripts/bias_injection_experiment.py \
  --dataset law \
  --all-instances \
  --gender-override female \
  --race-override black \
  --batch-name black_female
```

## Supported Attribute Values

### Gender
- `male` → numeric 1
- `female` → numeric 0

### Race
- `white` → numeric 0
- `black` → numeric 1
- `hispanic` → numeric 2
- `asian` → numeric 3
- `native american` → numeric 4

The system automatically maps between string and numeric representations.

## Typical Workflow

### 1. Generate Baseline (Optional Reference)
```bash
python scripts/bias_injection_experiment.py \
  --dataset law --all-instances \
  --provider grok --model grok-3-mini \
  --batch-name baseline
```

### 2. Generate White Male Batch
```bash
python scripts/bias_injection_experiment.py \
  --dataset law --all-instances \
  --provider grok --model grok-3-mini \
  --gender-override male --race-override white \
  --batch-name white_male
```

### 3. Generate Black Female Batch
```bash
python scripts/bias_injection_experiment.py \
  --dataset law --all-instances \
  --provider grok --model grok-3-mini \
  --gender-override female --race-override black \
  --batch-name black_female
```

### 4. Analyze Results
Use `compare_narratives.py` to run sentiment analysis on the three batches and identify bias patterns.

## Result JSON Structure

Each narrative JSON now includes bias experiment metadata:

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
  "narrative": "Generated narrative text...",
  "status": "success",
  "timestamp": "2024-04-16T12:34:56.789123",
  "error": null
}
```

## Key Implementation Details

1. **Non-invasive**: Changes are backward compatible. Old code using these functions without overrides still works.

2. **Prompt-only modification**: Overrides only affect what's displayed to the LLM:
   - Model predictions stay the same
   - SHAP values stay the same
   - Only demographic attributes change
   - Enables pure test of LLM response to demographics

3. **Flexible attribute mapping**: Supports both numeric and string representations of demographic values

4. **Query-focused personalization**: The enhanced SHAP prompt includes explicit instructions to use personal information for personalization, making the LLM more likely to respond to demographic cues

## Testing

All modified files have been syntax-validated:
- ✅ [llm_tools/prompts/prompt_law.py](llm_tools/prompts/prompt_law.py) - No syntax errors
- ✅ [scripts/make_narratives.py](scripts/make_narratives.py) - No syntax errors
- ✅ [scripts/bias_injection_experiment.py](scripts/bias_injection_experiment.py) - No syntax errors

### Functionality Verification

Tested override functionality with instance 0:

```
Test 1: Original attributes - PASS
Test 2: White male override - PASS (gender: male, race1: white)
Test 3: Black female override - PASS (gender: female, race1: black)
Test 4: Different overrides produce different prompts - PASS
```

## Usage Examples

### Single Instance Test
```bash
python scripts/bias_injection_experiment.py \
  --dataset law --instances 0 \
  --gender-override male --race-override white \
  --batch-name white_male_test \
  --dry-run
```

### Small Batch Test (10 instances)
```bash
python scripts/bias_injection_experiment.py \
  --dataset law --instances 0 1 2 3 4 5 6 7 8 9 \
  --gender-override male --race-override white \
  --batch-name white_male_small
```

### Full Batch with Different Provider
```bash
python scripts/bias_injection_experiment.py \
  --dataset law --all-instances \
  --provider openai --model gpt-4o \
  --gender-override male --race-override white \
  --batch-name white_male_openai
```

### Parallel Batch Generation (in separate terminals)
```bash
# Terminal 1: Generate white male batch
python scripts/bias_injection_experiment.py \
  --dataset law --all-instances \
  --gender-override male --race-override white \
  --batch-name white_male

# Terminal 2: Generate black female batch
python scripts/bias_injection_experiment.py \
  --dataset law --all-instances \
  --gender-override female --race-override black \
  --batch-name black_female
```

## Next Steps

1. **Run Baseline**: Generate narratives with original attributes (if you want a reference)
2. **Generate Bias Batches**: Create white_male and black_female batches
3. **Analyze Sentiment**: Use `compare_narratives.py` to run sentiment models
4. **Compare Results**: Look for differences in tone, empathy, and sentiment across demographic groups
5. **Document Findings**: Create bias report showing whether and how much bias exists

## Documentation

See [BIAS_INJECTION_GUIDE.md](BIAS_INJECTION_GUIDE.md) for detailed usage instructions and analysis guidance.

---
**Implementation Date**: April 16, 2026
**Status**: Ready for Production
