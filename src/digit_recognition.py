import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical

# Load dataset
train_data = pd.read_csv('../data/Train.csv')
print("Shape of train_data:", train_data.shape)

X = train_data.iloc[:, 1:].values
y = train_data.iloc[:, 0].values

# Preprocess
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)
y = to_categorical(y, num_classes=10)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../results/accuracy_plot.png')
plt.show()

# Save model
model.save('../models/digit_model.h5')
print("Model saved to ../models/digit_model.h5")

# Predict if test.csv exists
try:
    test_data = pd.read_csv('../data/test.csv')
    X_test = test_data.values / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)

    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    for i in range(5):
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {predicted_labels[i]}")
        plt.axis('off')
        plt.show()
except FileNotFoundError:
    print("test.csv not found — skipping predictions.")
