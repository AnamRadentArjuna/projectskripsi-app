import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import urllib.request
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

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
# KONFIGURASI MODEL
# ==============================
# !!! PENTING: sesuaikan dua hal ini dengan kondisi training kamu !!!

# 1) Cara preprocessing gambar saat training.
#    - Jika training pakai ImageDataGenerator(rescale=1./255) -> pakai "rescale"
#    - Jika training pakai tf.keras.applications.resnet50.preprocess_input -> pakai "resnet"
PREPROCESS_MODE = "rescale"  # ganti ke "resnet" jika perlu

# 2) Urutan label hasil sigmoid output model (index 0 vs 1).
#    Cek class_indices dari training kamu (flow_from_directory / image_dataset_from_directory).
#    Contoh: {'Lumpy Skin': 0, 'Normal Skin': 1} -> LABEL_FOR_SCORE_GE_0_5 = "Normal Skin"
LABEL_FOR_SCORE_GE_0_5 = "Lumpy Skin Disease (LSD)"
LABEL_FOR_SCORE_LT_0_5 = "Normal Skin"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_baseline_best.h5")
MODEL_URL = "https://github.com/AnamRadentArjuna/projectskripsi-app/releases/download/V1.0/resnet50_baseline_best.h5"


# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with st.spinner("Mengunduh model dari server..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    # compile=False -> lebih cepat & aman untuk inference-only
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


model = load_model()


# ==============================
# PREPROCESSING
# ==============================
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32)

    if PREPROCESS_MODE == "resnet":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        img = preprocess_input(img)
    else:
        img = img / 255.0

    img = np.expand_dims(img, axis=0)
    return img


# ==============================
# UPLOAD GAMBAR
# ==============================
uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(
        image,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    if st.button("Klasifikasi"):
        with st.spinner("Melakukan prediksi..."):
            img = preprocess_image(image)
            prediction = model.predict(img, verbose=0)
            score = float(prediction[0][0])

        st.divider()
        st.subheader("Hasil Prediksi")

        if score >= 0.5:
            confidence = score * 100
            label = LABEL_FOR_SCORE_GE_0_5
        else:
            confidence = (1 - score) * 100
            label = LABEL_FOR_SCORE_LT_0_5

        if label == LABEL_FOR_SCORE_GE_0_5:
            st.error(f"**{label}**")
        else:
            st.success(f"✅ **{label}**")

        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(confidence / 100)

        st.info(
            "Prediksi ini merupakan hasil model ResNet50 "
            "dan hanya digunakan sebagai alat bantu. "
            "Keputusan diagnosis tetap memerlukan pemeriksaan oleh dokter hewan."
        )
