import os
import csv
import random
import numpy as np
from collections import defaultdict # ne trazi provjeru kljuceva
from collections import Counter
import pandas as pd 

data_dir = "gesture_data"

label_list = ['double', 'flick', 'infinity', 'junk', 'kiss']
label_map = {label: idx for idx, label in enumerate(label_list)}
gesture_names = ['double', 'flick', 'infinity', 'kiss'] 

junk_label_idx = label_map['junk'] 

sensor_sample_rate = 88
window_size_seconds = 3 
window_size = int(window_size_seconds * sensor_sample_rate)
junk_stride = int(1 * sensor_sample_rate) 

MAX_WINDOWS_PER_TRAIN_CLASS = 40 
JUNK_MULTIPLIER = 1              

test_win_per_class = 10          
train_data = 'train_dataset.npz'
test_data = 'test_dataset.npz'

sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

def load_csv_as_dataframe(path, has_timestamp=True):
    df = pd.read_csv(path)
    if has_timestamp and 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'] / 1000.0
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int).astype(str)
    return df

def load_intervals(path):
    intervals = []
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if 'start_time' not in reader.fieldnames or 'end_time' not in reader.fieldnames:
                raise ValueError(f"Intervals file {path} must have 'start_time' and 'end_time' columns.")
            for row in reader:
                intervals.append({
                    'start_time': float(row['start_time']),
                    'end_time': float(row['end_time'])
                })
    except FileNotFoundError:
        print(f"Error: Intervals file not found at {path}")
        return []
    except Exception as e:
        print(f"Error reading intervals from {path}: {e}")
        return []
    return intervals

def inject_junk_between_intervals(data_df_full, intervals, junk_label_str_val):
    modified_rows = []
    
    intervals = sorted(intervals, key=lambda x: x['start_time'])

    last_idx_processed_in_original_df = 0

    for i in range(len(intervals)):
        interval = intervals[i]
        
        interval_data_start_idx = data_df_full['timestamp'].searchsorted(interval['start_time'], side='left')
        if interval_data_start_idx > last_idx_processed_in_original_df:
            modified_rows.extend(data_df_full.iloc[last_idx_processed_in_original_df : interval_data_start_idx].to_dict('records'))

        interval_data_end_idx = data_df_full['timestamp'].searchsorted(interval['end_time'], side='right') - 1
        modified_rows.extend(data_df_full.iloc[interval_data_start_idx : interval_data_end_idx + 1].to_dict('records'))
        
        last_idx_processed_in_original_df = interval_data_end_idx + 1

        if i < len(intervals) - 1:
            next_interval = intervals[i+1]
            
            gap_duration_seconds = next_interval['start_time'] - interval['end_time']
            
            if gap_duration_seconds > 0:
                mid_time_between = (interval['end_time'] + next_interval['start_time']) / 2
                
                closest_sample_idx = (data_df_full['timestamp'] - mid_time_between).abs().idxmin()
                middle_junk_sample_data = data_df_full.iloc[closest_sample_idx].copy()
                
                num_samples_in_gap = int(gap_duration_seconds * sensor_sample_rate)
                num_junk_samples_to_inject = min(num_samples_in_gap, 312)
                
                if num_junk_samples_to_inject > 0:
                    injected_junk_rows = []
                    current_junk_timestamp = interval['end_time'] + (1.0 / sensor_sample_rate) 
                    
                    for _ in range(num_junk_samples_to_inject):
                        junk_row = middle_junk_sample_data.to_dict()
                        junk_row['label'] = junk_label_str_val 
                        junk_row['timestamp'] = current_junk_timestamp 
                        injected_junk_rows.append(junk_row)
                        current_junk_timestamp += (1.0 / sensor_sample_rate) 
                        
                    modified_rows.extend(injected_junk_rows)
    
    if last_idx_processed_in_original_df < len(data_df_full):
        modified_rows.extend(data_df_full.iloc[last_idx_processed_in_original_df:].to_dict('records'))

    df_modified = pd.DataFrame(modified_rows, columns=data_df_full.columns).reset_index(drop=True)
    
    return df_modified

def extract_gesture_windows(data_df_full, intervals, gesture_name_str):
    windows = []
    labels = []
    
    if gesture_name_str not in label_map:
        print(f"Warning: Gesture '{gesture_name_str}' not found in label_map. Skipping.")
        return [], []

    intended_gesture_label_str = str(label_map[gesture_name_str])
    
    for interval in intervals: 
        interval_start_t = interval['start_time']
        interval_end_t = interval['end_time']
        
        interval_data_start_idx = data_df_full['timestamp'].searchsorted(interval_start_t, side='left')
        interval_data_end_idx = data_df_full['timestamp'].searchsorted(interval_end_t, side='right') - 1

        if interval_data_start_idx >= len(data_df_full) or interval_data_start_idx > interval_data_end_idx:
            continue
        
        trigger_found_at_idx = -1
        
        for idx_in_full_df in range(interval_data_start_idx, interval_data_end_idx + 1):
            if data_df_full.iloc[idx_in_full_df]['label'] == intended_gesture_label_str:
                gx, gy, gz = data_df_full.iloc[idx_in_full_df]['gx'], data_df_full.iloc[idx_in_full_df]['gy'], data_df_full.iloc[idx_in_full_df]['gz']
                
                if abs(gx) >= 1.0 or abs(gy) >= 1.0 or abs(gz) >= 1.0:
                    trigger_found_at_idx = idx_in_full_df
                    break

        if trigger_found_at_idx != -1:
            window_start_idx = trigger_found_at_idx
            window_end_idx = window_start_idx + window_size - 1

            if window_end_idx < len(data_df_full):
                window_df = data_df_full.loc[data_df_full.index[window_start_idx] : data_df_full.index[window_end_idx]].copy()
                
                if len(window_df) != window_size:
                    continue 

                final_cleaned_window_df = window_df

                if not all(col in final_cleaned_window_df.columns for col in sensor_cols):
                    print(f"Warning: Missing sensor columns in window for {gesture_name_str}. Skipping this window.")
                    continue
                
                features = final_cleaned_window_df[sensor_cols].astype(np.float32).values.tolist()
                
                windows.append(features)
                labels.append(label_map[gesture_name_str]) 
        
    return windows, labels


def extract_junk_windows(data_df_junk):
    windows = []
    labels = []
    
    junk_label_idx_val = label_map['junk']

    n = len(data_df_junk)
    
    effective_stride = int(1 * sensor_sample_rate) 

    for start_idx in range(0, n - window_size + 1, effective_stride):
        window_df = data_df_junk.iloc[start_idx : start_idx + window_size]
        
        if len(window_df) != window_size:
            continue

        if not all(col in window_df.columns for col in sensor_cols):
            print(f"Warning: Missing sensor columns in junk window. Skipping this window.")
            continue

        features = window_df[sensor_cols].astype(np.float32).values.tolist()
        windows.append(features)
        labels.append(junk_label_idx_val)
        
    return windows, labels

all_windows = defaultdict(list)
all_labels = defaultdict(list)

for gesture in gesture_names:
    final_csv_path = os.path.join(data_dir, f"{gesture}_final.csv")
    intervals_txt_path = os.path.join(data_dir, f"{gesture}_intervals.txt")

    try:
        data_df_full = load_csv_as_dataframe(final_csv_path, has_timestamp=True)
        intervals = load_intervals(intervals_txt_path)
        
        df_modified_with_injected_junk = inject_junk_between_intervals(data_df_full, intervals, str(junk_label_idx))

        windows, labels = extract_gesture_windows(
            df_modified_with_injected_junk, intervals, gesture
        )
        
        all_windows[gesture].extend(windows)
        all_labels[gesture].extend(labels)

        print(f"Processed {len(windows)} windows for gesture '{gesture}'.")

    except FileNotFoundError:
        print(f"Error: Missing files for '{gesture}' gesture. Ensure '{final_csv_path}' and '{intervals_txt_path}' exist. Skipping.")
    except Exception as e:
        print(f"Error processing '{gesture}' gesture data: {e}. Skipping this gesture.")


junk_final_csv_path = os.path.join(data_dir, "junk_final.csv") 
try:
    junk_data_df = load_csv_as_dataframe(junk_final_csv_path, has_timestamp=False)
    junk_windows, junk_labels = extract_junk_windows(junk_data_df)
    all_windows['junk'].extend(junk_windows)
    all_labels['junk'].extend(junk_labels)
    print(f"Processed {len(junk_windows)} windows for 'junk'.")
except FileNotFoundError:
    print(f"Error: Missing junk data file at '{junk_final_csv_path}'. No junk windows will be included.")
except Exception as e:
    print(f"Error processing junk data: {e}. Skipping junk data.")


print("\n🧩 Extracted windows per gesture (raw counts):")
for g in all_windows:
    print(f"{g}: {len(all_windows[g])}")

testing_data = []
testing_labels = []
training_data = []
training_labels = []

for label_name in label_list: 
    if label_name not in all_windows: 
        print(f"Warning: No data found for '{label_name}' to create test set. This class will be empty in test set.")
        continue
    items = list(zip(all_windows[label_name], all_labels[label_name]))
    random.shuffle(items)
    
    actual_test_count = min(test_win_per_class, len(items))
    
    test_items = items[:actual_test_count]
    for win, lab in test_items:
        testing_data.append(win)
        testing_labels.append(lab)
    
    remaining_items = items[actual_test_count:]
    all_windows[label_name] = [x[0] for x in remaining_items]
    all_labels[label_name] = [x[1] for x in remaining_items]


for gesture in gesture_names: 
    if gesture not in all_windows:
        print(f"Warning: No data found for '{gesture}' to create train set. This class will be empty in train set.")
        continue
    items = list(zip(all_windows[gesture], all_labels[gesture]))
    if len(items) < MAX_WINDOWS_PER_TRAIN_CLASS:
        items = items + random.choices(items, k=MAX_WINDOWS_PER_TRAIN_CLASS - len(items))
    else:
        items = random.sample(items, MAX_WINDOWS_PER_TRAIN_CLASS)
    for win, lab in items:
        training_data.append(win)
        training_labels.append(lab)

junk_items = list(zip(all_windows['junk'], all_labels['junk']))
target_junk_count = int(MAX_WINDOWS_PER_TRAIN_CLASS * JUNK_MULTIPLIER)
if len(junk_items) > target_junk_count:
    junk_items = random.sample(junk_items, target_junk_count)
elif len(junk_items) < target_junk_count:
    junk_items = junk_items + random.choices(junk_items, k=target_junk_count - len(junk_items))
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