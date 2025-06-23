import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import numpy as np 

gestures = ['novi'] 

data_dir = "gesture_data"
for gesture in gestures:
    input_file_path = os.path.join(data_dir, f"{gesture}_raw_data.csv") 
    output_pdf_path = os.path.join(data_dir, f"{gesture}_graphs.pdf")

    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Warning: Raw data file not found for '{gesture}' at '{input_file_path}'. Skipping this gesture.")
        continue
    except Exception as e:
        print(f"Error reading data from '{input_file_path}': {e}. Skipping.")
        continue

    columns_to_convert = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in '{input_file_path}'.")

    df.dropna(subset=columns_to_convert, inplace=True)

    if 'timestamp' in df.columns:
        df["timestamp"] = df["timestamp"] / 1000000.0 
    else:
        df['sample_index'] = df.index
        print(f"Note: '{gesture}_raw_data.csv' has no 'timestamp' column. Plotting against sample index.")

    start_time = df["timestamp"].min() if 'timestamp' in df.columns else df["sample_index"].min()
    end_time = df["timestamp"].max() if 'timestamp' in df.columns else df["sample_index"].max()
    
    x_axis_col = "timestamp" if 'timestamp' in df.columns else "sample_index"

    window_size = 3 
    step_size = 2 
    with PdfPages(output_pdf_path) as pdf:
        current_time = start_time
        while current_time + window_size <= end_time:
            window_df = df[(df[x_axis_col] >= current_time) & (df[x_axis_col] < current_time + window_size)]

            if not window_df.empty:
                fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True) 
                
                fig.suptitle(f"Sensor Data for Gesture: {gesture.capitalize()} (Time Window: {current_time:.1f}s - {current_time + window_size:.1f}s)")

                axs[0].plot(window_df[x_axis_col], window_df["ax"], label="ax")
                axs[0].plot(window_df[x_axis_col], window_df["ay"], label="ay")
                axs[0].plot(window_df[x_axis_col], window_df["az"], label="az")
                axs[0].legend()
                axs[0].grid(True)
                axs[0].set_ylabel("Value")


                axs[1].plot(window_df[x_axis_col], window_df["gx"], label="gx")
                axs[1].plot(window_df[x_axis_col], window_df["gy"], label="gy")
                axs[1].plot(window_df[x_axis_col], window_df["gz"], label="gz")
                axs[1].set_xlabel("time (s)" if x_axis_col == "timestamp" else "Sample Index")
                axs[1].set_ylabel("Value")
                axs[1].legend()
                axs[1].grid(True)

                tick_start = window_df[x_axis_col].iloc[0]
                tick_end = tick_start + window_size
                if x_axis_col == "timestamp":
                    ticks = [round(t, 1) for t in list(pd.Series(np.arange(0, int(window_size * 10 + 1))).div(10) + tick_start)]
                else: 
                    ticks = np.arange(tick_start, tick_end + 1, (tick_end - tick_start) // 5 if (tick_end - tick_start) > 5 else 1)
                
                axs[1].set_xticks(ticks)
                axs[1].tick_params(axis='x', labelsize=8)
                plt.setp(axs[1].get_xticklabels(), rotation=75)

                pdf.savefig(fig)
                plt.close(fig)

            current_time += step_size
    print(f"Graphs generated and saved to '{output_pdf_path}' for gesture '{gesture}'.")

print("\nAll requested plots generated and saved.")