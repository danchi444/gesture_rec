import os
import pandas as pd

DATA_FOLDER = "gesture_data"

labels = ['double', 'flick', 'infinity', 'junk', 'kiss']
label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    in_name = f"{label}_normalized.csv"
    out_name = f"{label}_final.csv"
    in_path = os.path.join(DATA_FOLDER, in_name)
    out_path = os.path.join(DATA_FOLDER, out_name)

    if not os.path.isfile(in_path):
        print(f"File not found: {in_path}")
        continue

    df = pd.read_csv(in_path)

    df['label'] = df['label'].map(label_map)

    if df['label'].isnull().any():
        print(f"Warning: Some labels in {in_name} were not recognized and became NaN")

    df.to_csv(out_path, index=False)
    print(f"Finsihed: {out_name}")