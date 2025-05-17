import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model('drowsiness_eye_model.h5')

# Page title
st.title("Drowsiness Detection (Eye Open/Closed)")
st.write("Upload an image of your eyes OR take a photo using your webcam.")

# Let user choose input method
option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif option == "Use Webcam":
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    # Convert image to grayscale and resize to 24x24
    img = np.array(image.convert('L'))  # grayscale
    img = cv2.resize(img, (24, 24))     # resize to 24x24
    img = img / 255.0                   # normalize
    img = img.reshape(1, 24, 24, 1)     # reshape for model

    # Predict
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        st.success("🟢 Eyes are OPEN")
    else:
        st.error("🔴 Eyes are CLOSED")
