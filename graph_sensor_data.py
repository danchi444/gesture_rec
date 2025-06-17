import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

df = pd.read_csv("gesture_data/alpha_raw_data.csv")

df["timestamp"] = df["timestamp"] / 1000.0

start_time = df["timestamp"].min()
end_time = df["timestamp"].max()

window_size = 3
step_size = 2
with PdfPages("gesture_data/alpha_graphs.pdf") as pdf:
    current_time = start_time
    while current_time + window_size <= end_time:
        window_df = df[(df["timestamp"] >= current_time) & (df["timestamp"] < current_time + window_size)]
        # npr. df[df['x'] > 1] je df[bool_list]

        if not window_df.empty:
            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True) # sharex da su poravnati grafovi

            axs[0].plot(window_df["timestamp"], window_df["ax"], label="ax")
            axs[0].plot(window_df["timestamp"], window_df["ay"], label="ay")
            axs[0].plot(window_df["timestamp"], window_df["az"], label="az")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(window_df["timestamp"], window_df["gx"], label="gx")
            axs[1].plot(window_df["timestamp"], window_df["gy"], label="gy")
            axs[1].plot(window_df["timestamp"], window_df["gz"], label="gz")
            axs[1].set_xlabel("time (s)")
            axs[1].legend()
            axs[1].grid(True)

            tick_start = window_df["timestamp"].iloc[0]
            tick_end = tick_start + window_size
            ticks = [round(t, 1) for t in list(pd.Series(range(0, int(window_size * 10 + 1))).div(10) + tick_start)]
            axs[1].set_xticks(ticks)
            axs[1].tick_params(axis='x', labelsize=8)
            plt.setp(axs[1].get_xticklabels(), rotation=75)

            pdf.savefig(fig)
            plt.close(fig)

        current_time += step_size