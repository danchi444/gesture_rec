import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv # To load intervals as CSV
from collections import Counter # For final counts

# --- Configuration ---
RAW_DATA_FILE = os.path.join('gesture_data', 'novi_raw_data.csv')
INTERVALS_FILE = os.path.join('gesture_data', 'novi_intervals.txt') # It's a TXT file but CSV format
QUANTIZED_MODEL_PATH = 'model_quantized.tflite'

SENSOR_SAMPLE_RATE_HZ = 88 # From your new Arduino code
WINDOW_SIZE = int(3 * SENSOR_SAMPLE_RATE_HZ) # 3 seconds * 88 Hz = 264 samples
NUM_FEATURES = 6 # ax, ay, az, gx, gy, gz

# Normalization constants from your provided Arduino code
MEANS = np.array([0.157700, -0.303717, 0.655390, 1.477335, -2.782316, -7.167620])
STDS = np.array([0.314980, 0.295075, 0.531993, 98.251744, 59.885257, 32.191821])

# --- Label Mapping ---
# IMPORTANT: This must match the label_map created by your create_windows.py
# based on its label_list = ['double', 'flick', 'infinity', 'junk', 'kiss']
# So the order here maps to the integer indices 0, 1, 2, 3, 4
CLASS_NAMES = ['double', 'flick', 'infinity', 'junk', 'kiss']
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
JUNK_LABEL_INDEX = LABEL_MAP['junk'] # Should be 3
sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

# --- Interval Segmentation for 'novi' data ---
# As specified:
INTERVAL_SEGMENTS = {
    'double': {'start_row': 0, 'count': 4},
    'flick': {'start_row': 4, 'count': 5},
    'infinity': {'start_row': 4 + 5, 'count': 5},
    'kiss': {'start_row': 4 + 5 + 5, 'count': 6},
    'junk_segment': {'start_row': 4 + 5 + 5 + 6, 'count': 1} # Special handling for the last row (junk interval)
}

# --- Data Loading and Preprocessing Functions ---

def load_raw_data_as_dataframe(path):
    df = pd.read_csv(path)
    # Convert timestamp from microseconds to seconds
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['timestamp'] = df['timestamp'] / 1000000.0
    
    # Convert sensor columns to numeric, handling potential strings
    for col in sensor_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            raise ValueError(f"Missing sensor column: {col} in {path}")
    df.dropna(subset=sensor_cols, inplace=True) # Drop rows with any NaN after conversion

    return df

def load_intervals_from_txt(path):
    intervals = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            intervals.append({'start_time': float(row['start_time']), 'end_time': float(row['end_time'])})
    return intervals

def normalize_sensor_data(df, means, stds, sensor_cols):
    df_norm = df.copy()
    for i, col in enumerate(sensor_cols):
        df_norm[col] = (df_norm[col] - means[i]) / stds[i]
    return df_norm

def extract_gesture_window(full_df_normalized, interval, gesture_type_name):
    """
    Extracts a single window for a gesture based on its interval.
    Finds the first gyro trigger within the interval and takes WINDOW_SIZE samples from there.
    """
    windows = []
    labels = []

    # Get start and end indices of the interval in the full dataframe
    interval_data_start_idx = full_df_normalized['timestamp'].searchsorted(interval['start_time'], side='left')
    interval_data_end_idx = full_df_normalized['timestamp'].searchsorted(interval['end_time'], side='right') - 1

    if interval_data_start_idx >= len(full_df_normalized) or interval_data_start_idx > interval_data_end_idx:
        print(f"Warning: Interval for {gesture_type_name} ({interval['start_time']:.2f}-{interval['end_time']:.2f}s) is out of data range or empty. Skipping.")
        return [], []

    # Scan for the first gyro trigger *within this specific interval's data segment*
    trigger_start_idx = -1
    
    # Iterate only through the samples within the current interval's bounds
    for idx_in_full_df in range(interval_data_start_idx, interval_data_end_idx + 1):
        # We assume the label in the data is correct within the interval, but only check gyro if it's the intended gesture.
        # This check is crucial for finding the *activity peak* within the labelled segment.
        gx, gy, gz = full_df_normalized.iloc[idx_in_full_df]['gx'], full_df_normalized.iloc[idx_in_full_df]['gy'], full_df_normalized.iloc[idx_in_full_df]['gz']
        
        if abs(gx) >= 1.0 or abs(gy) >= 1.0 or abs(gz) >= 1.0:
            trigger_start_idx = idx_in_full_df # Found trigger point
            break # Stop scanning this interval once trigger is found

    if trigger_start_idx != -1: # If a valid gyro trigger was found within the interval
        window_start_idx = trigger_start_idx
        window_end_idx = window_start_idx + WINDOW_SIZE - 1

        if window_end_idx < len(full_df_normalized):
            window_df = full_df_normalized.loc[full_df_normalized.index[window_start_idx] : full_df_normalized.index[window_end_idx]].copy()
            
            if len(window_df) != WINDOW_SIZE:
                print(f"Warning: Extracted window for {gesture_type_name} (starting {full_df_normalized.iloc[window_start_idx]['timestamp']:.2f}s) has incorrect size ({len(window_df)} instead of {WINDOW_SIZE}). Skipping.")
                return [], [] # Return empty if window size is incorrect

            # Extract features (dropping timestamp and labels for model input)
            features = window_df[sensor_cols].astype(np.float32).values.tolist()
            
            windows.append(features)
            labels.append(LABEL_MAP[gesture_type_name])
    else:
        print(f"Warning: No gyro trigger (abs > 1.0) found within interval for {gesture_type_name} ({interval['start_time']:.2f}-{interval['end_time']:.2f}s). Skipping.")
        
    return windows, labels

def extract_junk_windows_from_interval(full_df_normalized, interval):
    """
    Extracts junk windows from a specific interval in the raw data.
    Uses a 1-second stride (60 samples) for junk windows.
    """
    windows = []
    labels = []

    # Get start and end indices of the junk interval in the full dataframe
    junk_data_start_idx = full_df_normalized['timestamp'].searchsorted(interval['start_time'], side='left')
    junk_data_end_idx = full_df_normalized['timestamp'].searchsorted(interval['end_time'], side='right') - 1

    if junk_data_start_idx >= len(full_df_normalized) or junk_data_start_idx > junk_data_end_idx:
        print(f"Warning: Junk interval ({interval['start_time']:.2f}-{interval['end_time']:.2f}s) is out of data range or empty. Skipping.")
        return [], []
    
    # Slice the DataFrame for the junk segment
    junk_segment_df = full_df_normalized.loc[full_df_normalized.index[junk_data_start_idx] : full_df_normalized.index[junk_data_end_idx]].copy()

    # Iterate through the junk segment with a 1-second stride
    n = len(junk_segment_df)
    effective_stride = int(1 * SENSOR_SAMPLE_RATE_HZ)

    for start_idx_in_segment in range(0, n - WINDOW_SIZE + 1, effective_stride):
        window_df = junk_segment_df.iloc[start_idx_in_segment : start_idx_in_segment + WINDOW_SIZE]
        
        if len(window_df) != WINDOW_SIZE:
            continue

        features = window_df[sensor_cols].astype(np.float32).values.tolist()
        windows.append(features)
        labels.append(JUNK_LABEL_INDEX) # Label as junk
        
    return windows, labels


# --- Main Script Execution ---

if __name__ == "__main__":
    print("Starting specialized 'novi' data processing for quantized model testing.")

    # 1. Load raw data and intervals
    try:
        raw_df = load_raw_data_as_dataframe(RAW_DATA_FILE)
        intervals = load_intervals_from_txt(INTERVALS_FILE)
        if not intervals:
            print("No intervals loaded. Exiting.")
            exit()
    except Exception as e:
        print(f"Failed to load raw data or intervals: {e}. Exiting.")
        exit()

    print(f"Raw data loaded. Total samples: {len(raw_df)}")
    print(f"Intervals loaded. Total intervals: {len(intervals)}")

    # 2. Normalize the entire raw DataFrame once
    normalized_df = normalize_sensor_data(raw_df, MEANS, STDS, sensor_cols)
    print("Raw data normalized.")

    # Prepare lists for all test windows and labels
    all_test_windows = []
    all_test_labels = []
    
    # 3. Extract gesture windows based on defined segments
    current_interval_row_idx = 0
    for gesture_name, segment_info in INTERVAL_SEGMENTS.items():
        if gesture_name == 'junk_segment':
            # Handle junk segment separately later
            continue 

        start_interval_idx = segment_info['start_row']
        end_interval_idx = start_interval_idx + segment_info['count']
        
        # Slice intervals for the current gesture
        gesture_intervals = intervals[start_interval_idx : end_interval_idx]
        
        print(f"\nExtracting windows for gesture: '{gesture_name}' ({len(gesture_intervals)} intervals)")
        for interval in gesture_intervals:
            windows, labels = extract_gesture_window(normalized_df, interval, gesture_name)
            all_test_windows.extend(windows)
            all_test_labels.extend(labels)
            
    # 4. Extract junk windows from the specified junk segment
    junk_segment_info = INTERVAL_SEGMENTS['junk_segment']
    junk_interval_row = intervals[junk_segment_info['start_row']]
    
    print(f"\nExtracting windows for junk from interval: {junk_interval_row['start_time']:.2f}-{junk_interval_row['end_time']:.2f}s")
    junk_windows, junk_labels = extract_junk_windows_from_interval(normalized_df, junk_interval_row)
    all_test_windows.extend(junk_windows)
    all_test_labels.extend(junk_labels)

    # Convert lists to numpy arrays for the model
    X_test_novi = np.array(all_test_windows, dtype=np.float32)
    y_test_novi = np.array(all_test_labels)

    print(f"\n--- Prepared 'novi' test dataset ---")
    print(f"X_test_novi shape: {X_test_novi.shape}")
    print(f"y_test_novi counts: {Counter(y_test_novi)}")


    # 5. Load and Test Quantized TFLite Model
    try:
        interpreter = tf.lite.Interpreter(model_path=QUANTIZED_MODEL_PATH)
        interpreter.allocate_tensors()
        print(f"\nQuantized model loaded from '{QUANTIZED_MODEL_PATH}'.")
    except FileNotFoundError:
        print(f"Error: Quantized model '{QUANTIZED_MODEL_PATH}' not found. Please run quantize_model.py first.")
        exit()
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        exit()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale = input_details['quantization_parameters']['scales'][0]
    input_zero_point = input_details['quantization_parameters']['zero_points'][0]
    output_scale = output_details['quantization_parameters']['scales'][0]
    output_zero_point = output_details['quantization_parameters']['zero_points'][0]

    print(f"Input Quantization: scale={input_scale}, zero_point={input_zero_point}")
    print(f"Output Quantization: scale={output_scale}, zero_point={output_zero_point}")

    # Run inference
    y_pred = []
    print("\nRunning inference on 'novi' test data...")
    for i in range(X_test_novi.shape[0]):
        input_data_float = X_test_novi[i:i+1] # Get one window (batch size of 1)
        
        # Quantize input data (float32 to int8)
        input_data_int8 = (input_data_float / input_scale + input_zero_point).astype(input_details['dtype'])

        interpreter.set_tensor(input_details['index'], input_data_int8)
        interpreter.invoke()
        output_data_int8 = interpreter.get_tensor(output_details['index'])

        # Dequantize output data (int8 to float32 probabilities)
        output_data_float = (output_data_int8 - output_zero_point) * output_scale

        predicted_class_idx = np.argmax(output_data_float[0])
        y_pred.append(predicted_class_idx)

    y_pred = np.array(y_pred)

    # 6. Evaluate and Plot Confusion Matrix
    print("\n--- Evaluation on 'novi' data (Quantized Model) ---")
    
    # Filter CLASS_NAMES to only include labels actually present in y_test_novi
    unique_labels_in_test = np.unique(y_test_novi)
    filtered_class_names = [name for idx, name in enumerate(CLASS_NAMES) if idx in unique_labels_in_test]

    print("\nClassification Report:")
    print(classification_report(y_test_novi, y_pred, target_names=filtered_class_names, labels=unique_labels_in_test))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_novi, y_pred, labels=unique_labels_in_test)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=filtered_class_names, yticklabels=filtered_class_names)
    plt.title("Confusion Matrix for 'novi' Data (Quantized Model)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_novi_quantized.png")
    plt.show()

    print("\n'novi' data testing complete. Confusion matrix saved as 'confusion_matrix_novi_quantized.png'.")