import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("traffic_sign_model.h5")
print("✅ Model loaded successfully!")

# Path to the test image
test_image_path = r"C:\Users\sruja\Desktop\EDUCATION\vidya project\archive (3)\istockphoto-1437819031-612x612.jpg"

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read the image
    if img is None:
        raise ValueError("❌ Error: Image not found or cannot be loaded!")
    
    img = cv2.resize(img, (32, 32))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model
    return img

# Load and preprocess the image
try:
    image = preprocess_image(test_image_path)
    print(f"📂 Loading image from: {test_image_path}")

    # Make prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    # Output result
    print(f"🚦 Predicted Traffic Sign Class: {predicted_class}")

except Exception as e:
    print(f"❌ Error: {e}")
