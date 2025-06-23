import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

files_to_plot = ['double', 'flick', 'infinity', 'junk', 'kiss']

data_dir = "gesture_data"

plotting_window_size = 1000

for file_prefix in files_to_plot:
    input_file_path = os.path.join(data_dir, f"{file_prefix}_final.csv")
    output_pdf_path = os.path.join(data_dir, f"{file_prefix}_norm_graphs.pdf")

    try:
        df = pd.read_csv(input_file_path)
        
    except FileNotFoundError:
        print(f"Error: Data file not found for '{file_prefix}' at '{input_file_path}'. Skipping.")
        continue
    except Exception as e:
        print(f"Error reading data from '{input_file_path}': {e}. Skipping.")
        continue

    with PdfPages(output_pdf_path) as pdf:
        print(f"Generating plots for {file_prefix}...")

        num_samples = len(df)
        for start_idx in range(0, num_samples, plotting_window_size):
            end_idx = min(start_idx + plotting_window_size, num_samples)
            window_df = df.iloc[start_idx:end_idx]

            if window_df.empty:
                continue

            fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True) 

            axs[0].plot(window_df.index, window_df["ax"], label="ax")
            axs[0].plot(window_df.index, window_df["ay"], label="ay")
            axs[0].plot(window_df.index, window_df["az"], label="az")
            axs[0].set_title(f"{file_prefix.capitalize()} - Acceleration (Samples {start_idx}-{end_idx-1})")
            axs[0].set_ylabel("Normalized Value")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(window_df.index, window_df["gx"], label="gx")
            axs[1].plot(window_df.index, window_df["gy"], label="gy")
            axs[1].plot(window_df.index, window_df["gz"], label="gz")
            axs[1].set_title(f"{file_prefix.capitalize()} - Gyroscope (Samples {start_idx}-{end_idx-1})")
            axs[1].set_xlabel("Sample Index")
            axs[1].set_ylabel("Normalized Value")
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Saved plots for '{file_prefix}' to '{output_pdf_path}'.")

print("\nAll requested plots generated and saved.")