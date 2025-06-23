import os
import pandas as pd

data_dir = "gesture_data"
gesture_names = ['double', 'flick', 'infinity', 'kiss']

for gesture in gesture_names:
    print(f"\n--- {gesture} ---")
    raw_path = os.path.join(data_dir, f"{gesture}_raw_data.csv")
    if not os.path.exists(raw_path):
        print(f"  {raw_path} not found.")
        continue
    df_raw = pd.read_csv(raw_path)
    if 'timestamp' not in df_raw.columns or len(df_raw) < 2:
        print("  Not enough data or no timestamp column.")
    else:
        ts = df_raw['timestamp'].values
        intervals = (ts[1:] - ts[:-1]) / 1000.0
        avg_interval = intervals.mean()
        sample_rate = 1 / avg_interval if avg_interval > 0 else float('nan')
        print(f"  Raw data: {len(ts)} samples")
        print(f"  Average interval: {avg_interval:.4f} s")
        print(f"  Estimated sample rate: {sample_rate:.2f} Hz")

    labeled_path = os.path.join(data_dir, f"{gesture}_labeled.csv")
    if not os.path.exists(labeled_path):
        print(f"  {labeled_path} not found.")
        continue
    df_labeled = pd.read_csv(labeled_path)
    if 'label' not in df_labeled.columns:
        print("  No label column.")
        continue

    lengths = []
    current_len = 0
    for label in df_labeled['label']:
        if label == gesture:
            current_len += 1
        else:
            if current_len > 0:
                lengths.append(current_len)
                current_len = 0
    if current_len > 0:
        lengths.append(current_len)

    if lengths:
        avg_len = sum(lengths) / len(lengths)
        print(f"  Avg. consecutive '{gesture}' window: {avg_len:.2f} samples")
        if 'timestamp' in df_raw.columns and len(df_raw) > 1:
            print(f"  Avg. consecutive '{gesture}' window: {avg_len * avg_interval:.2f} seconds")
    else:
        print(f"  No consecutive '{gesture}' windows found.")