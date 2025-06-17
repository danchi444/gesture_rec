import pandas as pd
import os

folder = "gesture_data"

file_path = os.path.join(folder, f"flick_raw_data.csv")

df = pd.read_csv(file_path)

offset = df.iloc[0, 0]
df.iloc[:, 0] = df.iloc[:, 0] - offset

df.to_csv(file_path, index=False)