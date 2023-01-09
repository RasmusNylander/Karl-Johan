import os

import pandas as pd

CSV_PATH = "./datasets/sorted_downscaled/train.csv"
SUBSET_PATH = "./datasets/sorted_downscaled/validation.csv"
DIFFERENCE_PATH = "./datasets/sorted_downscaled/new_train.csv"
SAMPLE_PERCENTAGE = 0.20
SEED = 9000 + 4206969420

# Avoid accidentally overwriting data
error: str = ""
if os.path.exists(SUBSET_PATH):
    error = f"Subset path {SUBSET_PATH} already exists. "
if os.path.exists(DIFFERENCE_PATH):
    error += f"Difference path {DIFFERENCE_PATH} already exists. "

if not os.path.exists(CSV_PATH):
    error += f"CSV path {CSV_PATH} does not exist. "

if SAMPLE_PERCENTAGE < 0 or SAMPLE_PERCENTAGE > 1:
    error += f"Sample percentage {SAMPLE_PERCENTAGE} is not between 0 and 1. "

if error:
    raise ValueError(error.strip())

original_csv = pd.read_csv(CSV_PATH)
labels = original_csv['0'].map(lambda x: x[0:2])
grouped = original_csv.groupby(labels.values)
csv_subset = grouped.sample(frac=SAMPLE_PERCENTAGE, random_state=SEED)
difference = original_csv.index.difference(csv_subset.index)
original_without_subset = original_csv.loc[difference]
csv_subset.to_csv(SUBSET_PATH, index=False)
original_without_subset.to_csv(DIFFERENCE_PATH, index=False)

