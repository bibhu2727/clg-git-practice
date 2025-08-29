import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Generate some dummy data (for regression)
# y = 3x + noise
np.random.seed(42)
X = np.random.rand(1000, 1)
y = 3 * X + np.random.randn(1000, 1) * 0.1

# Define the regression model
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(1,)),  
    layers.Dense(64, activation="relu"),
    layers.Dense(1)  
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train the model
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Plot training & validation loss
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

# Plot training & validation MAE
plt.plot(history.history['mae'], label="Training MAE")
plt.plot(history.history['val_mae'], label="Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.title("Training vs Validation MAE")
plt.show()

# Evaluate on training data
loss, mae = model.evaluate(X, y, verbose=0)
print(f"Final Mean Absolute Error: {mae:.4f}")

# Make predictions
X_test = np.array([[0.2], [0.5], [0.9]])
y_pred = model.predict(X_test)
print("Predictions:", y_pred.flatten())