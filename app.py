import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺")

st.title("🩺 Hybrid Skin Disease Screening System")
st.write("Upload a skin image to predict the skin condition.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

model = load_model()

# Load labels
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.asarray(img).astype(np.float32)
    img_array = (img_array / 127.5) - 1
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index] * 100

    st.success(f"Predicted Disease: {class_names[index]}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Symptoms")
    itching = st.checkbox("Itching")
    redness = st.checkbox("Redness")
    pain = st.checkbox("Pain")
    dry_skin = st.checkbox("Dry Skin")

    if st.button("Show Final Result"):
        st.write("### Final Screening Result")
        st.write(f"**Prediction:** {class_names[index]}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        if itching or redness or pain or dry_skin:
            st.warning("Symptoms are present. Please consult a dermatologist for proper diagnosis.")
        else:
            st.success("No symptoms selected. Continue monitoring and consult a doctor if symptoms develop.")
