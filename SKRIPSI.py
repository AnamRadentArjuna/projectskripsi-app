import os
import gdown
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
# GOOGLE DRIVE
# ==============================
FILE_ID = "1wkI0iK1BxGkLbqjjK5YaozQ7CzXsaqLa"
MODEL_PATH = "resnet50_baseline_best.h5"


# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():

    try:

        # Download model jika belum ada
        if not os.path.exists(MODEL_PATH):

            url = f"https://drive.google.com/uc?id={FILE_ID}"

            with st.spinner("Mengunduh model, mohon tunggu..."):

                gdown.download(
                    url,
                    MODEL_PATH,
                    quiet=False
                )

        model = tf.keras.models.load_model(MODEL_PATH)

        return model

    except Exception as e:
        st.error(f"Gagal memuat model.\n\n{e}")
        st.stop()


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
    "📷 Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    if st.button("🔍 Klasifikasi"):

        with st.spinner("Melakukan prediksi..."):

            img = preprocess_image(image)

            prediction = model.predict(img, verbose=0)

            score = float(prediction[0][0])

        st.divider()

        st.subheader("Hasil Prediksi")

        if score >= 0.5:

            confidence = score * 100

            st.error("🦠 **Lumpy Skin Disease (LSD)**")

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
