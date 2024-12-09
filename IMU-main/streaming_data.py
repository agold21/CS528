import pickle
import numpy as np
import sys
import time
import os
import pandas as pd
import serial
import re

with open('trained_svm_model.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)

words = {
    0: "Alexa",
    1: "Play",
    2: "Despacito",
    3: "Beethoven",
    4: "Still"
}

sequence_length = 60

def normalize_data(data):
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)
    return data

def preprocess_data(raw_data):
    raw_data = raw_data.astype(np.float32)
    raw_data = normalize_data(raw_data)

    if raw_data.shape[0] < sequence_length:
        padding = np.zeros((sequence_length - raw_data.shape[0], raw_data.shape[1]))
        raw_data = np.vstack((raw_data, padding))
    elif raw_data.shape[0] > sequence_length:
        raw_data = raw_data[:sequence_length]

    return raw_data.flatten().reshape(1, -1)

serial_port = '/dev/tty.usbserial-1120' 
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=2)

acce_pattern = r'accel_x:([\d.-]+), accel_y:([\d.-]+), accel_z:([\d.-]+)'
gyro_pattern = r'gyro_x:([\d.-]+), gyro_y:([\d.-]+), gyro_z:([\d.-]+)'

acc_data_buffer = []
gyro_data_buffer = []

latest_acce = None
latest_gyro = None

previous_gesture = None
hold_duration = 2.0

print("Starting real-time detection...")

try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        acce_match = re.search(acce_pattern, line)
        gyro_match = re.search(gyro_pattern, line)

        if acce_match:
            try:
                ax, ay, az = map(float, acce_match.groups())
                latest_acce = (ax, ay, az)
            except ValueError:
                latest_acce = None

        if gyro_match:
            try:
                gx, gy, gz = map(float, gyro_match.groups())
                latest_gyro = (gx, gy, gz)
            except ValueError:
                latest_gyro = None

        if latest_acce is not None and latest_gyro is not None:
            ax, ay, az = latest_acce
            gx, gy, gz = latest_gyro

            acc_data_buffer.append([ax, ay, az])
            gyro_data_buffer.append([gx, gy, gz])

            latest_acce = None
            latest_gyro = None

            if len(acc_data_buffer) >= sequence_length and len(gyro_data_buffer) >= sequence_length:
                acc_segment = np.array(acc_data_buffer[-sequence_length:])
                gyro_segment = np.array(gyro_data_buffer[-sequence_length:])
                imu_data = np.hstack((acc_segment, gyro_segment))

                features = preprocess_data(imu_data)
                prediction = svm_classifier.predict(features)[0]
                predicted_word = words[prediction]

                if predicted_word != "Still":
                    print(f"\rDetected Word: {predicted_word}  ", end='', flush=True)

                    time.sleep(hold_duration)

                    sys.stdout.write("\r" + " " * 50 + "\r")
                    sys.stdout.flush()

                    acc_data_buffer.clear()
                    gyro_data_buffer.clear()
                    predicted_word = None
                else:
                    if predicted_word is not None:
                        sys.stdout.write("\r" + " " * 50 + "\r")
                        sys.stdout.flush()
                        predicted_word = None

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nExiting real-time detection...")
    ser.close()
