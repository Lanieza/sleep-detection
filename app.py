import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2  # OpenCV for eye detection

# Load model
model = load_model('drowsiness_eye_model.h5')

# Load OpenCV's pre-trained eye detector
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_and_preprocess_eye(pil_image):
    # Convert PIL image to OpenCV format (RGB to BGR)
    cv_img = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(eyes) == 0:
        return None  # No eye detected

    # Crop first detected eye region
    (x, y, w, h) = eyes[0]
    eye_img = gray[y:y+h, x:x+w]

    # Resize and normalize
    eye_img = cv2.resize(eye_img, (24, 24))
    eye_img = eye_img / 255.0

    # Reshape to model input shape
    eye_img = eye_img.reshape(1, 24, 24, 1)
    return eye_img

st.title("Eye Detection (Open/Closed)")
st.write("Upload or capture an image of your eyes to check if they're open or closed.")

# Capture or upload
img_file_buffer = st.camera_input("📷 Take a photo using your webcam or phone")
uploaded_file = st.file_uploader("📁 Or upload an image...", type=["jpg", "jpeg", "png"])

# Choose input image
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
else:
    image = None

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    # Detect and preprocess eye region automatically
    processed_eye = detect_and_preprocess_eye(image)

    if processed_eye is None:
        st.warning("⚠️ Could not detect an eye. Please try a clearer image focusing on your eye.")
    else:
        prediction = model.predict(processed_eye)[0][0]
        if prediction > 0.5:
            st.success("🟢 Eyes are OPEN")
        else:
            st.error("🔴 Eyes are CLOSED")
