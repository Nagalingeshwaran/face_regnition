import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Face Recognition App", layout="centered")

MODEL_PATH = "face_model.h5"

# 🔹 Replace this with your Google Drive FILE ID
GDRIVE_FILE_ID = "https://drive.google.com/file/d/12dI4pxOr0IbFaufdNDy5bSl_zzbVNyXE/view?usp=sharing"
GDRIVE_URL = f"https://drive.google.com/drive/folders/1zmCsaBGlX1-JHSdTuTzdIJbo27o-RYgF={GDRIVE_FILE_ID}"

# 🔹 Class labels (CHANGE based on your model)
CLASS_NAMES = ["nagalingeshwaran", "vijay", "deepika"]

# -----------------------------
# DOWNLOAD MODEL FROM GDRIVE
# -----------------------------
@st.cache_resource
def load_face_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH)
    return model

model = load_face_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(img):
    img = img.resize((224, 224))   # change size if your model uses different input
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("👤 Face Recognition System")
st.write("Upload a face image to identify the person")

uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Recognize Face"):
        with st.spinner("Predicting..."):
            processed_img = preprocess_image(img)
            prediction = model.predict(processed_img)

            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

        st.success(f"✅ **Prediction:** {CLASS_NAMES[predicted_class]}")
        st.info(f"📊 **Confidence:** {confidence:.2f}%")
