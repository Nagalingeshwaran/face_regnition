import streamlit as st
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.set_page_config(page_title="Face Recognition App", layout="centered")
st.title("😀 Face Recognition App")
st.write("Upload a face image to identify the person")

# 🔗 Google Drive file ID
MODEL_ID = "12dI4pxOr0IbFaufdNDy5bSl_zzbVNyXE"
MODEL_PATH = "face_model.h5"

# 📥 Download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully")

# ✅ Load model
try:
    model = load_model(MODEL_PATH, compile=False)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Failed to load model 😕")
    st.exception(e)
    st.stop()

# ✅ Class names (must match training order)
CLASS_NAMES = ["Nagalingeshwaran", "Vijay", "Deepika"]

# 🖼 Upload image
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))  # adjust if different
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        preds = model.predict(img_array)
        class_index = np.argmax(preds)
        confidence = float(np.max(preds) * 100)

        st.subheader("Prediction Result")
        st.success(f"👤 Person: **{CLASS_NAMES[class_index]}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")
