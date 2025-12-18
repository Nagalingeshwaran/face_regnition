import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os

# Google Drive link
url = "https://drive.google.com/file/d/12dI4pxOr0IbFaufdNDy5bSl_zzbVNyXE"
MODEL_PATH = "FACEREGINITION.h5"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    gdown.download(url, MODEL_PATH, quiet=False)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (must match training folder names)
class_names = ['nagalingeshwaran', 'vijay', 'deepika']

st.title("🧑 Face Recognition App")
st.write("Upload a face image to predict the person")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]

    st.success(f"✅ Predicted Person: **{predicted_class}**")
