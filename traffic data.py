import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define dataset path
data_dir = r"C:\Users\sruja\Desktop\EDUCATION\vidya project\archive (3)\Indian-Traffic Sign-Dataset\Images"

# Initialize empty lists for images and labels
images = []
labels = []

# Loop through each subfolder (0 to 54)
for class_label in range(55):  # Assuming 55 classes (0 to 54)
    class_path = os.path.join(data_dir, str(class_label))  # Path to each class folder
    if not os.path.exists(class_path):
        print(f"Warning: Folder {class_path} not found.")
        continue
    
    # Read each image in the folder
    for img_file in tqdm(os.listdir(class_path), desc=f"Processing class {class_label}"):
        img_path = os.path.join(class_path, img_file)
        
        # Load image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to load {img_path}")
            continue
        
        # Resize image to 32x32 (dataset standard)
        img = cv2.resize(img, (32, 32))
        
        # Convert to RGB (if OpenCV loads as BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        images.append(img)
        labels.append(class_label)

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Save data for easy loading later
np.save("traffic_sign_images.npy", images)
np.save("traffic_sign_labels.npy", labels)

print("Dataset loaded successfully! Shape:")
print("Images:", images.shape)
print("Labels:", labels.shape)
