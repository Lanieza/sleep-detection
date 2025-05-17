import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model('drowsiness_eye_model.h5')

st.title("Drowsiness Detection (Eye Open/Closed)")
st.write("Upload or capture an image of your eyes to check if they're open or closed.")

# Option 1: Capture using camera
img_file_buffer = st.camera_input("📷 Take a photo using your webcam or phone")

# Option 2: Upload an image
uploaded_file = st.file_uploader("📁 Or upload an image...", type=["jpg", "jpeg", "png"])

# Use camera input if available, otherwise use uploaded image
if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert("L")
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
else:
    image = None

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    # Resize and preprocess
    image = image.resize((24, 24))
    img = np.array(image) / 255.0
    img = img.reshape(1, 24, 24, 1)

    # Predict
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        st.success("🟢 Eyes are OPEN")
    else:
        st.error("🔴 Eyes are CLOSED")
