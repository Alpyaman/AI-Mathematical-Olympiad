import pandas as pd

df = pd.read_csv('evaluation-results-phase2/detailed_results.csv')

print("\n" + "="*70)
print("TEST RESULTS ANALYSIS")
print("="*70 + "\n")

for i, row in df.iterrows():
    print(f"{i+1}. Problem: {row['problem']}")
    print(f"   Generated: {row['generated'][:200]}")
    print(f"   Extracted Answer: {row['extracted_answer']}")
    print()
