import serial
import time
import csv
import re
from datetime import datetime

# Configure serial connection
ser = serial.Serial(port='COM4', baudrate=115200, timeout=1)

# Set up files
filenames = ["alexa.csv", "play.csv", "despacito.csv", "beethoven.csv"]

# Create expression to remove ANSI escape codes
ansi_escape = re.compile(r'(?:\x1B[@-_][0-?]*[ -/]*[@-~])')

# Define the regex patterns for accelerometer data
accel_pattern = re.compile(
    r'acce_x:\s*([-+]?\d*\.\d+|\d+)\s*,\s*acce_y:\s*([-+]?\d*\.\d+|\d+)\s*,\s*acce_z:\s*([-+]?\d*\.\d+|\d+)'
)

# Define the regex patterns for gyroscope data
gyro_pattern = re.compile(
    r'gyro_x:\s*([-+]?\d*\.\d+|\d+)\s*,\s*gyro_y:\s*([-+]?\d*\.\d+|\d+)\s*,\s*gyro_z:\s*([-+]?\d*\.\d+|\d+)'
)

# Function to extract accelerometer data
def extract_accel_data(data):
    match = accel_pattern.search(data)
    if match:
        return {
            'accel_x': match.group(1),
            'accel_y': match.group(2),
            'accel_z': match.group(3),
        }
    return None

# Function to extract gyroscope data
def extract_gyro_data(data):
    match = gyro_pattern.search(data)
    if match:
        return {
            'gyro_x': match.group(1),
            'gyro_y': match.group(2),
            'gyro_z': match.group(3),
        }
    return None

# Function to clean ANSI characters from data
def clean_data(data):
    return ansi_escape.sub('', data)

# Set up device to avoid late outputs
starttime = time.time()    
endtime = starttime + 1.0
while (time.time() < endtime):
    try:
        if ser.in_waiting > 0:  # Check if data is available
            # Read data
            data = ser.readline().decode('utf-8').strip()
    except KeyboardInterrupt:
        print("Terminating program")

# Run for all 4 motions/files
for i in range(4):
    # mode='w' will delete existing file data
    with open(filenames[i], mode='w', newline='') as file:
        # Create csv writer
        csv_writer = csv.writer(file)
        # Write header
        csv_writer.writerow(['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        print(f"Mouth the word {file.name} in 4 seconds")
        for t in reversed(range(2)):
            time.sleep(1)
            if t == 1:
                ser.reset_input_buffer() # resets input buffer so data read is correct data 
            print(f"{t} seconds")
        print(f"Mouth the word {file.name} over 4 seconds")
        # Set up timing for the while loop
        starttime = time.time()    
        endtime = starttime + 4.0
        #time.perf_counter()
        while (time.time() < endtime):
            try:
                if ser.in_waiting > 0:  # Check if data is available
                    # Read data
                    data = ser.readline().decode('utf-8').strip()
                    # Remove ANSI codes
                    data = clean_data(data)
                    # Check to see if gyro and/or accel data came in
                    accel_data = extract_accel_data(data)
                    gyro_data = extract_gyro_data(data) 
                    if accel_data or gyro_data:             
                        # Prepare data row
                        row = [
                            accel_data['accel_x'] if accel_data else '',
                            accel_data['accel_y'] if accel_data else '',
                            accel_data['accel_z'] if accel_data else '',
                            gyro_data['gyro_x'] if gyro_data else '',
                            gyro_data['gyro_y'] if gyro_data else '',
                            gyro_data['gyro_z'] if gyro_data else '',
                        ]
                        # Write to file
                        csv_writer.writerow(row)
                        file.flush()
            except KeyboardInterrupt:
                print("Terminating program")
            
        print("STOPPPP STOPP STOPPPPP")
        # wait a bit so I can reset my face
        time.sleep(2)        
print("Done!")
ser.close()