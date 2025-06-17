import os
import csv

data_dir = "gesture_data"
gesture_names = ['8', 'double', 'flick', '4', 'alpha']

bins = [
    (None, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
    (1.0, 1.1),
    (1.1, 1.2),
    (1.2, 1.3)
]

for gesture in gesture_names:
    path = os.path.join(data_dir, f"{gesture}_intervals.txt")
    counts = [0 for _ in bins]
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = float(row['start_time'])
            end = float(row['end_time'])
            length = end - start
            for i, (bin_start, bin_end) in enumerate(bins):
                if bin_start is None:
                    if length < bin_end:
                        counts[i] += 1
                        break
                else:
                    if bin_start <= length < bin_end:
                        counts[i] += 1
                        break
    print(f"\n{gesture}:")
    for i, (bin_start, bin_end) in enumerate(bins):
        if bin_start is None:
            label = f"<{bin_end:.1f}"
        else:
            label = f"{bin_start:.1f}-{bin_end:.1f}"
        print(f"  {label}: {counts[i]}")