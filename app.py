import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page config
st.set_page_config(page_title="Face Recognition App", layout="centered")

st.title("😀 Face Recognition App")
st.write("Upload a face image to identify the person")

# Load model
@st.cache_resource
def load_face_model():
    return load_model("FACEREGNITION.h5")

model = load_face_model()
st.success("✅ Model loaded successfully")

# ✅ Person names (order MUST match model training)
CLASS_NAMES = [
    "Nagalingeshwaran",
    "Vijay",
    "Deepika"
]

# Upload image
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ⚠️ Change size if your model was trained with a different shape
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.subheader("Prediction Result")
        st.success(f"👤 Person: **{CLASS_NAMES[predicted_index]}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")
