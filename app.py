import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load model once
model = load_model('drowsiness_eye_model.h5')

# Load OpenCV eye cascade classifier
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_and_preprocess_eye(pil_image):
    # Convert PIL image to OpenCV BGR
    cv_img = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(eyes) == 0:
        return None  # No eyes detected
    
    # Crop the first detected eye
    (x, y, w, h) = eyes[0]
    eye_img = gray[y:y+h, x:x+w]

    # Resize to 24x24 (your model input size)
    eye_img = cv2.resize(eye_img, (24, 24))
    
    # Normalize
    eye_img = eye_img / 255.0
    
    # Reshape for model (batch, height, width, channels)
    eye_img = eye_img.reshape(1, 24, 24, 1)
    return eye_img

st.title("Sleep Detection (Eye Open/Closed)")
st.write("Upload or capture an image of your eyes to check if they're open or closed.")

img_file_buffer = st.camera_input("📷 Take a photo using your webcam or phone")
uploaded_file = st.file_uploader("📁 Or upload an image...", type=["jpg", "jpeg", "png"])

image = None
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)
    
    processed_eye = detect_and_preprocess_eye(image)
    
    if processed_eye is None:
        st.warning("Could not detect an eye in the image. Please try again with a clearer eye photo.")
    else:
        prediction = model.predict(processed_eye)[0][0]
        st.write(f"Prediction score: {prediction:.4f}")
        if prediction > 0.5:
            st.success("🟢 Eyes are OPEN")
        else:
            st.error("🔴 Eyes are CLOSED")
