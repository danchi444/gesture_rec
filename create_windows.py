import os
import csv
import random
from collections import defaultdict # ne trazi provjeru kljuca

data_dir = "gesture_data"
gesture_names = ['8', 'double', 'alpha', 'flick', '4']

sensor_sample_rate = 60
window_size = int(0.8 * sensor_sample_rate)
stride = int(0.4 * sensor_sample_rate)

gesture_ratio_thresholds = {
    'double': 0.6, # 0.65 (avg trajanje geste u s)
    'flick': 0.55, # 0.6
    'alpha': 0.75, # 0.86
    '4': 0.65, # 0.76
    '8': 0.65 # 0.76
}

test_win_per_class = 25
train_data = 'train_dataset.csv'
test_data = 'test_dataset.csv'

sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

def load_csv(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f) # mogu selektirat stupce po imenu (umjesto po indeksu)
        return list(reader)
    
def save_dataset(windows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{col}{i}' for i in range(window_size) for col in sensor_cols]
        writer.writerow(header)
        for label, features in windows:
            writer.writerow([label] + features)

def extract_windows(data, gesture_name=None): 
    windows = []
    n = len(data)
    for start in range(0, n - window_size + 1, stride):
        window_rows = data[start:start+window_size]
        label_counter = defaultdict(int)
        for row in window_rows:
            label_counter[row['label']] += 1

        if gesture_name is not None:
            min_gesture_ratio = gesture_ratio_thresholds.get(gesture_name, 0.5)
            gesture_count = label_counter[gesture_name]
            if gesture_count / window_size >= min_gesture_ratio:
                label = gesture_name
            else:
                label = 'junk'
        else:
            label = 'junk'

        flat_features = []
        for row in window_rows:
            for col in sensor_cols:
                flat_features.append(float(row[col]))

        windows.append((label, flat_features))
    return windows

all_windows = defaultdict(list)

for gesture in gesture_names:
    path = os.path.join(data_dir, f"{gesture}_labeled.csv")
    data = load_csv(path)
    windows = extract_windows(data, gesture_name=gesture)
    for label, features in windows:
        all_windows[label].append((label, features))

junk_path = os.path.join(data_dir, "junk_labeled.csv")
junk_data = load_csv(junk_path)
junk_windows = extract_windows(junk_data, gesture_name=None)
for label, feats in junk_windows:
    all_windows['junk'].append((label, feats))

testing_data = []
for gesture in gesture_names + ['junk']:
    items = all_windows[gesture]
    random.shuffle(items)
    test_items = items[:test_win_per_class]
    testing_data.extend(test_items)
    all_windows[gesture] = items[test_win_per_class:]

gesture_counter = [len(all_windows[g]) for g in gesture_names]
max_gesture_count = max(gesture_counter)

training_data = []
for gesture in gesture_names:
    items = all_windows[gesture]
    if len(items) < max_gesture_count:
        items = items + random.choices(items, k=max_gesture_count - len(items))
    training_data.extend(items)

junk_items = all_windows['junk']
if len(junk_items) > max_gesture_count:
    junk_items = random.sample(junk_items, max_gesture_count)
elif len(junk_items) < max_gesture_count:
    junk_items = junk_items + random.choices(junk_items, k=max_gesture_count - len(junk_items))
training_data.extend(junk_items)

random.shuffle(training_data)
random.shuffle(testing_data)

save_dataset(training_data, train_data)
save_dataset(testing_data, test_data)

print("✅ window processing complete.")
print(f"🧪 train samples: {len(training_data)}")
print(f"🔬 test samples: {len(testing_data)}")