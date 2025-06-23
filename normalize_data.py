import os
import pandas as pd
import numpy as np

DATA_DIR = "gesture_data"
GESTURES = ['infinity', 'kiss', 'double', 'flick', 'junk']
SENSOR_COLUMNS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

all_data = []

for gesture in GESTURES:
    file_path = os.path.join(DATA_DIR, f"{gesture}_labeled.csv")
    df = pd.read_csv(file_path)
    all_data.append(df[SENSOR_COLUMNS])

combined_df = pd.concat(all_data)
means = combined_df.mean()
stds = combined_df.std()

for gesture in GESTURES:
    file_path = os.path.join(DATA_DIR, f"{gesture}_labeled.csv")
    df = pd.read_csv(file_path)
    
    df_normalized = df.copy()
    for col in SENSOR_COLUMNS:
        df_normalized[col] = (df[col] - means[col]) / stds[col]
    
    out_path = os.path.join(DATA_DIR, f"{gesture}_normalized.csv")
    df_normalized.to_csv(out_path, index=False)

with open("norm_data.txt", "w") as f:
    f.write("const float means[6] = {")
    f.write(", ".join(f"{means[col]:.6f}" for col in SENSOR_COLUMNS))
    f.write("};\n")

    f.write("const float stds[6] = {")
    f.write(", ".join(f"{stds[col]:.6f}" for col in SENSOR_COLUMNS))
    f.write("};\n")