import pandas as pd
import os

# Check all Excel files in results directory
results_dir = 'results'
excel_files = []
for root, dirs, files in os.walk(results_dir):
    for f in files:
        if f.endswith('.xlsx'):
            excel_files.append(os.path.join(root, f))

print(f"Found {len(excel_files)} Excel files:")
for ef in excel_files:
    print(f"  - {ef}")
    try:
        df = pd.read_excel(ef, sheet_name='Detailed Results')
        print(f"    Rows: {len(df)}")
        if len(df) > 0:
            print(f"    Columns: {df.columns.tolist()}")
            if 'sign_accuracy' in df.columns and 'value_accuracy' in df.columns:
                print(f"    Sign Accuracy Values: {df['sign_accuracy'].tolist()[:5]}")
                print(f"    Value Accuracy Values: {df['value_accuracy'].tolist()[:5]}")
    except Exception as e:
        print(f"    Error reading: {e}")
print()
