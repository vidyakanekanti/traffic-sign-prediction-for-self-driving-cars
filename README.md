# Traffic Sign Prediction for Self-Driving Vehicles

## Overview
This project focuses on developing a deep learning model to accurately recognize and classify traffic signs, a crucial component in self-driving vehicle systems. The model is trained using a dataset of various traffic signs and leverages convolutional neural networks (CNNs) for efficient image recognition. By enabling autonomous vehicles to interpret traffic signs in real-time, this project contributes to safer and more intelligent transportation solutions.

## Dataset and Required Files
To test the model, you need several `.npy` files, including preprocessed training, validation, and test datasets. Due to GitHub's file size limitations, these files are hosted on Google Drive and can be accessed using the following link:

[Download Required `.npy` Files](https://drive.google.com/drive/folders/1kksTQMUjEXJnq0e0i2LI8t3-lAPRSJPi?usp=drive_link)

## Features
- Traffic sign classification using a deep learning model (CNN).
- Pretrained model for faster inference.
- Data preprocessing and augmentation for improved accuracy.
- Implementation in Python with TensorFlow/Keras.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/vidyakanekanti/traffic-sign-prediction-for-self-driving-cars.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the `.npy` files from the Google Drive link and place them in the appropriate directory.
4. Run the model training script:
   ```bash
   python train.py
   ```
5. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Model Architecture
The model is built using a CNN with the following layers:
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification

## Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request.


