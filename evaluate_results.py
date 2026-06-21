import pandas as pd
import re


def normalize(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9\s]", "", text)

    text = text.strip()

    return text


df = pd.read_csv("results_no_rag.csv")

exact_matches = 0

for _, row in df.iterrows():

    pred = normalize(row["model_output"])

    gt = normalize(row["ground_truth"])

    print("\nGROUND TRUTH:", gt)
    print("PREDICTION:", pred)

    if gt in pred:
        exact_matches += 1


em_score = exact_matches / len(df)

print("\n===== RESULTS =====\n")

print(f"Exact Match: {em_score:.4f}")