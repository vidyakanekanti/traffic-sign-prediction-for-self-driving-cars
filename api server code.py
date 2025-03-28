from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import cv2
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained model
try:
    model = tf.keras.models.load_model("traffic_sign_model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Function to preprocess the image
def preprocess_image(image_bytes):
    try:
        # Convert bytes to numpy array
        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # Decode the image

        if image is None:
            return None

        image = cv2.resize(image, (32, 32))  # Resize to match model input
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Expand dimensions for model
        return image
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None

# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = preprocess_image(file.read())

    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    predictions = model.predict(image)
    predicted_class = int(np.argmax(predictions))

    response = jsonify({"predicted_class": predicted_class})
    print(f"📡 Sending Response: {response.get_json()}")  # Debugging
    return response

# Serve the HTML file
@app.route("/")
def serve_frontend():
    html_path = os.path.join(os.getcwd(), "traffic-signs-for-selfdriving-vehicles.html")
    return send_file(html_path)

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
