import os
import csv
import random
import numpy as np
from collections import defaultdict # ne trazi provjeru kljuca
from collections import Counter

data_dir = "gesture_data"
label_list = ['4', '8', 'alpha', 'double', 'flick', 'junk']
label_map = {label: idx for idx, label in enumerate(label_list)}
gesture_names = label_list[:-1]

sensor_sample_rate = 60
window_size = int(0.8 * sensor_sample_rate)
stride = int(0.4 * sensor_sample_rate)

gesture_ratio_thresholds = {
    'double': 0.65, # 0.65 (avg trajanje geste u s)
    'flick': 0.6, # 0.6
    'alpha': 0.8, # 0.86
    '4': 0.7, # 0.76
    '8': 0.7 # 0.76
}

test_win_per_class = 50
train_data = 'train_dataset.npz'
test_data = 'test_dataset.npz'

sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

def load_csv(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f) # mogu selektirat stupce po imenu (umjesto po indeksu)
        return list(reader)

def extract_windows(data, gesture_name=None):
    windows = []
    labels = []
    n = len(data)
    for start in range(0, n - window_size + 1, stride):
        window_rows = data[start:start+window_size]
        label_counter = defaultdict(int)
        for row in window_rows:
            label_counter[row['label']] += 1

        if gesture_name is not None:
            min_gesture_ratio = gesture_ratio_thresholds.get(gesture_name, 0.5)
            gesture_count = label_counter[str(label_map[gesture_name])]
            if gesture_count / window_size >= min_gesture_ratio:
                label = label_map[gesture_name]
            else:
                label = label_map['junk']
        else:
            label = label_map['junk']

        features = []
        for row in window_rows:
            features.append([float(row[col]) for col in sensor_cols])

        windows.append(features)
        labels.append(label)
    return windows, labels

all_windows = defaultdict(list)
all_labels = defaultdict(list)

for gesture in gesture_names:
    path = os.path.join(data_dir, f"{gesture}_normalized.csv")
    data = load_csv(path)
    windows, labels = extract_windows(data, gesture_name=gesture)

    gesture_label = label_map[gesture]
    for win, lab in zip(windows, labels):
        if lab == gesture_label:
            all_windows[gesture].append(win)
            all_labels[gesture].append(lab)
        else:
            all_windows['junk'].append(win)
            all_labels['junk'].append(lab)

junk_path = os.path.join(data_dir, "junk_normalized.csv")
junk_data = load_csv(junk_path)
junk_windows, junk_labels = extract_windows(junk_data, gesture_name=None)
all_windows['junk'].extend(junk_windows)
all_labels['junk'].extend(junk_labels)

print("\n🧩 Extracted windows per gesture:")
for g in all_windows:
    print(f"{g}: {len(all_windows[g])}")

testing_data = []
testing_labels = []
for gesture in gesture_names + ['junk']:
    items = list(zip(all_windows[gesture], all_labels[gesture]))
    random.shuffle(items)
    test_items = items[:test_win_per_class]
    for win, lab in test_items:
        testing_data.append(win)
        testing_labels.append(lab)
    all_windows[gesture] = [x[0] for x in items[test_win_per_class:]]
    all_labels[gesture] = [x[1] for x in items[test_win_per_class:]]

gesture_counts = [len(all_windows[g]) for g in gesture_names]
max_gesture_count = max(gesture_counts)

training_data = []
training_labels = []
for gesture in gesture_names:
    items = list(zip(all_windows[gesture], all_labels[gesture]))
    if len(items) < max_gesture_count:
        items = items + random.choices(items, k=max_gesture_count - len(items))
    else:
        items = random.sample(items, max_gesture_count)
    for win, lab in items:
        training_data.append(win)
        training_labels.append(lab)

junk_items = list(zip(all_windows['junk'], all_labels['junk']))
if len(junk_items) > max_gesture_count:
    junk_items = random.sample(junk_items, max_gesture_count)
elif len(junk_items) < max_gesture_count:
    junk_items = junk_items + random.choices(junk_items, k=max_gesture_count - len(junk_items))
for win, lab in junk_items:
    training_data.append(win)
    training_labels.append(lab)

combined = list(zip(training_data, training_labels))
random.shuffle(combined)
training_data, training_labels = zip(*combined)

X_train = np.array(training_data, dtype=np.float32)
y_train = np.array(training_labels)
X_test = np.array(testing_data, dtype=np.float32)
y_test = np.array(testing_labels)

np.savez(train_data, X=X_train, y=y_train)
np.savez(test_data, X=X_test, y=y_test)

print("✅ window processing complete.")
print(f"🧪 train samples: {len(X_train)}")
print(f"🔬 test samples: {len(X_test)}")
print("📊 train windows by class:")
print(Counter(y_train))
print("📊 test windows by class:")
print(Counter(y_test))