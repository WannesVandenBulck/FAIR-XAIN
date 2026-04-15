"""Test to compare old substring matching vs new word boundary matching."""
import re

# Sample narrative that would cause false positives
sample_narrative = """
The applicant demonstrated strong performance in management roles.
He excelled in business development and showed leadership qualities.
Management recognized his contributions to the team.
The male candidate was promoted after showing initiative.
His woman colleague also received recognition for her work.
"""

print("=" * 70)
print("COMPARING MENTION COUNTING METHODS")
print("=" * 70)
print("\nSample Narrative:")
print(sample_narrative)

# OLD METHOD: Simple substring matching (INCORRECT)
print("\n" + "=" * 70)
print("OLD METHOD: Simple substring matching (WRONG - causes false positives)")
print("=" * 70)

gender_terms_old = ["Male", "Female", "Man", "Woman", "man", "woman", "men", "women", "boy", "girl",
                   "masculine", "feminine", "male", "female"]

gender_count_old = sum(sample_narrative.lower().count(term.lower()) for term in gender_terms_old)
print(f"Total gender mentions (OLD): {gender_count_old}")
print("\nBreakdown of false matches:")
print(f"  'man' matches: {sample_narrative.lower().count('man')} times")
print(f"    - In 'performance', 'management', 'business', 'womanly', etc.")
print(f"  'women' matches: {sample_narrative.lower().count('women')} times")
print(f"    - In 'woman' (not just standalone 'women')")

# NEW METHOD: Word boundary matching (CORRECT)
print("\n" + "=" * 70)
print("NEW METHOD: Word boundary matching (CORRECT - only whole words)")
print("=" * 70)

race_terms = ["Black", "Hispanic", "Asian", "White", "Native American", 
             "African", "Latino", "Caucasian", "Asians", "Africans"]
gender_terms = ["Male", "Female", "Man", "Woman", "men", "women", "boy", "girl",
               "masculine", "feminine"]

# Compile regex patterns with word boundaries (case-insensitive)
race_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in race_terms]
gender_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in gender_terms]

race_count_new = sum(len(pattern.findall(sample_narrative)) for pattern in race_patterns)
gender_count_new = sum(len(pattern.findall(sample_narrative)) for pattern in gender_patterns)

print(f"Total race mentions (NEW): {race_count_new}")
print(f"Total gender mentions (NEW): {gender_count_new}")
print("\nCorrected matches:")
print(f"  'Male' found: {len(re.findall(r'\\bMale\\b', sample_narrative, re.IGNORECASE))} times (correct)")
print(f"  'Woman' found: {len(re.findall(r'\\bWoman\\b', sample_narrative, re.IGNORECASE))} times (correct)")
print(f"  'Man' found: {len(re.findall(r'\\bMan\\b', sample_narrative, re.IGNORECASE))} times (correct, excludes 'performance', 'management')")

print("\n" + "=" * 70)
print(f"DIFFERENCE: Old method reported {gender_count_old} mentions, New method reports {gender_count_new}")
print(f"False positives eliminated: {gender_count_old - gender_count_new}")
print("=" * 70)
