import os
import pandas as pd

gestures = ['8', 'double', 'flick', '4', 'alpha']
folder = "gesture_data"

for gesture in gestures:
    raw_path = os.path.join(folder, f"{gesture}_raw_data.csv")
    interval_path = os.path.join(folder, f"{gesture}_intervals.txt")
    output_path = os.path.join(folder, f"{gesture}_labeled.csv")

    df = pd.read_csv(raw_path)
    intervals = pd.read_csv(interval_path)

    df['label'] = 'junk'

    current_index = 0
    num_intervals = len(intervals)

    for i, row in df.iterrows():
        t = row['timestamp']

        # preskoci nerelvantne intervale
        while current_index < num_intervals and t > intervals.loc[current_index, 'end_time'] * 1000:
            current_index += 1

        if current_index < num_intervals:
            start = intervals.loc[current_index, 'start_time'] * 1000
            end = intervals.loc[current_index, 'end_time'] * 1000
            if start <= t <= end:
                df.at[i, 'label'] = gesture

    df = df.drop(columns=['timestamp'])
    df.to_csv(output_path, index=False) # index=False da ne doda br. reda