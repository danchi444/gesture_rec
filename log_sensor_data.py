import serial
import csv
import os

port = 'COM9' 
baud_rate = 115200
data_dir = 'gesture_data'

file_name = os.path.join(data_dir, 'novi_raw_data.csv')

try:
    ser = serial.Serial(port, baud_rate, timeout=1) # timeout je za .readline()
    print(f"connected to {port} at {baud_rate} baud")
except serial.SerialException as e:
    print(e)
    exit()

with open(file_name, mode='w', newline='') as file: # newline='' zbog windows formatiranja
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])

    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                parts = line.split(',')
                writer.writerow(parts)
    finally:
        ser.close()
        print("serial connection closed")