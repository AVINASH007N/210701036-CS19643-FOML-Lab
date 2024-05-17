# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/content/sensor_data.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
data.set_index('Timestamp', inplace=True)
data.dropna(inplace=True)  # Drop rows with NaT values

# Add a column to flag anomalies
data['anomaly'] = data['SensorValue'] > 800

# Normalize the sensor values
scaler = MinMaxScaler()
data['normalized_value'] = scaler.fit_transform(data['SensorValue'].values.reshape(-1, 1))

# Display the first few rows
print(data.head())

# Create sequences
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

sequence_length = 50
sequences = create_sequences(data['normalized_value'].values, sequence_length)

# Split the data into training and testing sets
train_size = int(0.8 * len(sequences))
train_sequences = sequences[:train_size]
test_sequences = sequences[train_size:]

# Prepare the training data
X_train = train_sequences[:, :-1]
y_train = train_sequences[:, -1]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Prepare the testing data
X_test = test_sequences[:, :-1]
y_test = test_sequences[:, -1]

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length-1, 1)))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Predict the values
predictions = model.predict(X_test)

# Rescale the predictions back to original values
predictions_rescaled = scaler.inverse_transform(predictions)

# Add predictions and flags to the original data
test_data = data.iloc[len(data) - len(y_test):].copy()
test_data['predicted'] = predictions_rescaled

# Identify actual anomalies based on the threshold
test_data['anomaly'] = test_data['SensorValue'] > 800

# Print detected anomalies
print(f"Anomalies detected: {test_data['anomaly'].sum()}")

# Optionally, you can visualize the anomalies
plt.figure(figsize=(15, 5))
plt.plot(data.index, data['SensorValue'], label='Sensor Value')
plt.plot(test_data.index, test_data['predicted'], label='Predicted Value', color='orange')
plt.scatter(test_data[test_data['anomaly']].index, test_data[test_data['anomaly']]['SensorValue'], color='red', label='Anomalies')
plt.axhline(y=800, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()
