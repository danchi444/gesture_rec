import serial
import csv
import time
import os

port = 'COM9'
baud_rate = 115200
data_dir = 'gesture_data'
duration_seconds = 30

output_file = os.path.join(data_dir, 'junk_labeled.csv')

ser = serial.Serial(port, baud_rate, timeout=1)

with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label'])

    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        line = ser.readline().decode().strip()
        if line:
            row = line.split(',')
            writer.writerow(row + ['junk'])

ser.close()