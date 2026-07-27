import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import gdown
import urllib.request
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Klasifikasi Lumpy Skin Disease",
    page_icon="🐄",
    layout="centered"
)

st.title("🐄 Sistem Klasifikasi Lumpy Skin Disease (LSD)")
st.write(
    "Unggah gambar kulit sapi untuk mendeteksi apakah termasuk "
    "**Normal Skin** atau **Lumpy Skin Disease (LSD)**."
)

# ==============================
# GOOGLE DRIVE
# ==============================
FILE_ID = "1wkI0iK1BxGkLbqjjK5YaozQ7CzXsaqLa"
MODEL_PATH = "resnet50_baseline_best.h5"


# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    model_path = "model/resnet50_baseline_best.h5"
    url = "https://github.com/AnamRadentArjuna/projectskripsi-app/releases/download/V1.0/resnet50_baseline_best.h5"
    
    # Unduh file jika belum ada di folder
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        with st.spinner("Mengunduh model dari server..."):
            urllib.request.urlretrieve(url, model_path)
    
    # Load dan kembalikan model
    return tf.keras.models.load_model(model_path)

model = load_model()


# ==============================
# PREPROCESSING
# ==============================
def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    img = np.array(image).astype(np.float32)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# ==============================
# UPLOAD GAMBAR
# ==============================
uploaded_file = st.file_uploader(
    " Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    if st.button(" Klasifikasi"):

        with st.spinner("Melakukan prediksi..."):

            img = preprocess_image(image)

            prediction = model.predict(img, verbose=0)

            score = float(prediction[0][0])

        st.divider()

        st.subheader("Hasil Prediksi")

        if score >= 0.5:

            confidence = score * 100

            st.error(" **Lumpy Skin Disease (LSD)**")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        else:

            confidence = (1 - score) * 100

            st.success("✅ **Normal Skin**")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.progress(confidence / 100)

        st.info(
            "Prediksi ini merupakan hasil model ResNet50 "
            "dan hanya digunakan sebagai alat bantu. "
            "Keputusan diagnosis tetap memerlukan pemeriksaan oleh dokter hewan."
        )
