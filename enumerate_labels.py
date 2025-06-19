import os
import pandas as pd

DATA_FOLDER = "gesture_data"

labels = ['4', '8', 'alpha', 'double', 'flick', 'junk']
label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    filename = f"{label}_normalized.csv"
    filepath = os.path.join(DATA_FOLDER, filename)

    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        continue

    df = pd.read_csv(filepath)

    df['label'] = df['label'].map(label_map)

    if df['label'].isnull().any():
        print(f"Warning: Some labels in {filename} were not recognized and became NaN")

    df.to_csv(filepath, index=False)
    print(f"Updated: {filename}")