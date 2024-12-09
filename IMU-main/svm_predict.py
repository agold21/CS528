import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle


labels = ["down", "left", "up", "right", "still"]

# Set up files with appropriate labels
all_data = []
for direction in os.listdir('csv_files/'):
    if len(direction) < 6: # avoids test.csv and train.csv
        for filename in os.listdir(f'csv_files/{direction}'):
            all_data.append([filename, direction]) 
        
files_df = pd.DataFrame(all_data)
files_df.columns = ['Filename', 'Direction']

# convert files to csv
files_df.to_csv("csv_files/train_svm.csv", index=False)

train_labels = pd.read_csv("csv_files/train_svm.csv") # Update directory

train_dir = "csv_files/" # Update directory

# Function to load dataset
def load_data(label_df, data_dir):
    # Empty lists to store features and labels
    features = []
    labels = []

    for _, row in label_df.iterrows():
        filename = os.path.join(data_dir, row['Direction'], row['Filename'])

        # Read file into pandas dataframe
        df = pd.read_csv(filename)

        # Keep only accelerometer and gyroscope signals
        data = df[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']].values.astype(np.float32)

        # Normalize data
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
   
        # Zero padding
        while len(data) < 500:
            zeros = np.array([[0, 0, 0, 0, 0, 0]])
            data = np.append(data, zeros, axis=0)
        
        # Populate lists with normalized data and labels
        features.append(data.flatten())
        labels.append(row['Direction'])

    return features, labels

def train(X_train, y_train):
    # Create the SVM classifier
    svm_classifier = SVC(kernel='rbf')

    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    return svm_classifier

# Create the train and test sets
X_train, y_train = load_data(train_labels, train_dir)

# Perform training and testing with SVM
svm = train(X_train, y_train)

def predict_svm(data):
    # Normalize data
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    # Flatten data
    flat_data = data.values.flatten()
    # Reshape for prediction (makes it 2D for 1 sample)
    flat_data = flat_data.reshape(1,-1)
    # # Make prediction Add if you want to see live visualizations
    # data.plot(y=["accel_x", "accel_y", "accel_z"], title="accel_data", xlabel="Time", ylabel="Amplitude")
    return data[0:0]

with open('trained_svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)