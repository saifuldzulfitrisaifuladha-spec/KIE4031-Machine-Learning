import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =====================================================================
# 1. DATA COLLECTION & PREPROCESSING (Question 1)
# =====================================================================

# Download Tesla historical data (Using a stable 6-year window)
ticker = "TSLA"
data = yf.download(ticker, start="2020-01-01", end="2026-01-01")

# Extract the 'Close' column and drop any missing rows
df = data[['Close']].dropna()

# Normalize features to a range between 0 and 1 using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Structure lookback sequences using a custom 45-day window
sequence_length = 45
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# Split into Training (80%) and Testing (20%) datasets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input vectors to match the 3D tensor shape expected by LSTM layers
# Shape format: [Samples, Time Steps, Features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"Data ready. Training Shape: {X_train.shape} | Testing Shape: {X_test.shape}")

# =====================================================================
# 2. DESIGN & MODEL DEVELOPMENT (Question 2)
# =====================================================================

# Build a deep Recurrent Neural Network (LSTM Architecture)
model = Sequential([
    # First LSTM layer with a custom unit size of 64
    LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.25),  # Prevents training phase overfitting
    
    # Second LSTM layer (Flattens structural dimensionality)
    LSTM(units=32, return_sequences=False),
    Dropout(0.25),
    
    # Fully connected dense layers mapping hidden states down to output
    Dense(units=16),
    Dense(units=1)  # Single output node for continuous price prediction
])

# Compile the model optimizing for Mean Squared Error loss
model.compile(optimizer='adam', loss='mean_squared_error')

print("Starting training routine...")
# Train the model over 12 epochs using a batch size of 32
history = model.fit(
    X_train, y_train, 
    batch_size=32, 
    epochs=12, 
    validation_data=(X_test, y_test),
    verbose=1
)

# =====================================================================
# 3. EVALUATION & VISUALIZATION (Question 3)
# =====================================================================

# Generate predictions on the unseen testing set
predictions = model.predict(X_test)

# Reverse data normalization back to real dollar values
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate assessment metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)

print("\n=== MODEL PERFORMANCE METRICS ===")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Error (MAE):     ${mae:.2f}\n")

# Align split dates cleanly for plotting
train_df = df[:train_size+sequence_length]
valid_df = df[train_size+sequence_length:].copy()
valid_df['Predictions'] = predictions

# Render the final visualization
plt.figure(figsize=(14, 7))
plt.title('KIE4031 Final Assessment: Tesla ($TSLA$) Price Prediction Model', fontsize=14)
plt.xlabel('Timeline Date', fontsize=12)
plt.ylabel('Stock Closing Price (USD)', fontsize=12)

# Plot actual historical tracks and comparative predicted pathways
plt.plot(train_df['Close'], label='Historical Training Influx', color='gray', alpha=0.5)
plt.plot(valid_df['Close'], label='Actual Market Curve', color='crimson', linewidth=1.5)
plt.plot(valid_df['Predictions'], label='LSTM Sequence Prediction', color='darkblue', linestyle='--', linewidth=1.5)

plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Save the plot directly into your workspace folder to drop into Word/LaTeX
plt.savefig('tsla_lstm_prediction_plot.png', dpi=300)
plt.show()
