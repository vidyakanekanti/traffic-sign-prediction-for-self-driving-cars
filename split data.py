import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
X = np.load("traffic_sign_images.npy")
y = np.load("traffic_sign_labels.npy")

# Normalize images
X = X / 255.0  

print("Dataset Loaded:", X.shape, y.shape)  # Check if data is loaded correctly

# Split the dataset (70% Train, 15% Validation, 15% Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Save the split datasets
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Data successfully split and saved!")
print(f"Training Set: {X_train.shape}, {y_train.shape}")
print(f"Validation Set: {X_val.shape}, {y_val.shape}")
print(f"Test Set: {X_test.shape}, {y_test.shape}")
