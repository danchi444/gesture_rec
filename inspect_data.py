import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# Define your label map to translate integer labels back to names
# This must match the label_map used in your create_windows.py
# Assuming: {'double':0, 'flick':1, 'infinity':2, 'junk':3, 'kiss':4}
# Make sure these are the exact labels your model is trained on
label_names = ['double', 'flick', 'infinity', 'junk', 'kiss']
label_map_reverse = {idx: name for idx, name in enumerate(label_names)}

def plot_and_save_windows(npz_filepath, output_pdf_filepath, dataset_type):
    """
    Loads an NPZ file, plots each window's sensor data, and saves to a PDF.
    """
    try:
        data = np.load(npz_filepath)
        X = data['X'] # Features
        y = data['y'] # Labels

        print(f"\n--- Processing {npz_filepath} ({dataset_type} dataset) ---")
        print(f"Total windows: {X.shape[0]}")
        print(f"Window shape: {X.shape[1]} samples x {X.shape[2]} features")

        # Open a PDF file to save all plots
        with PdfPages(output_pdf_filepath) as pdf:
            for i in range(X.shape[0]):
                window_data = X[i]
                window_label_idx = y[i]
                window_label_name = label_map_reverse.get(window_label_idx, f"Unknown_Label_{window_label_idx}")

                fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Two subplots, share x-axis
                
                # Title for the entire figure
                fig.suptitle(f"{dataset_type} Dataset - Window {i+1}/{X.shape[0]} | Label: {window_label_name}", fontsize=14)

                # Plot Acceleration (ax, ay, az) on the top subplot
                axs[0].plot(window_data[:, 0], label="ax")
                axs[0].plot(window_data[:, 1], label="ay")
                axs[0].plot(window_data[:, 2], label="az")
                axs[0].set_title("Acceleration (Normalized)")
                axs[0].set_ylabel("Normalized Value")
                axs[0].legend(loc='upper right')
                axs[0].grid(True)

                # Plot Gyroscope (gx, gy, gz) on the bottom subplot
                axs[1].plot(window_data[:, 3], label="gx")
                axs[1].plot(window_data[:, 4], label="gy")
                axs[1].plot(window_data[:, 5], label="gz")
                axs[1].set_title("Gyroscope (Normalized)")
                axs[1].set_xlabel("Sample Index") # Only the bottom subplot needs x-label when shared
                axs[1].set_ylabel("Normalized Value")
                axs[1].legend(loc='upper right')
                axs[1].grid(True)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
                pdf.savefig(fig) # Save the current figure to the PDF
                plt.close(fig) # Close the figure to free memory

            print(f"Finished plotting for {npz_filepath}. Saved to '{output_pdf_filepath}'.")

    except FileNotFoundError:
        print(f"Error: {npz_filepath} not found. Please ensure the dataset file exists.")
    except Exception as e:
        print(f"An error occurred while processing {npz_filepath}: {e}")

# --- Main execution ---
if __name__ == "__main__":
    # Paths to your dataset files
    train_npz = 'train_dataset.npz'
    test_npz = 'test_dataset.npz'

    # Output PDF paths
    train_output_pdf = 'train_windows_plots.pdf'
    test_output_pdf = 'test_windows_plots.pdf'

    plot_and_save_windows(train_npz, train_output_pdf, "Train")
    plot_and_save_windows(test_npz, test_output_pdf, "Test")

    print("\nAll dataset window plots generated.")