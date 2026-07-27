import streamlit as st
import os
import requests
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi LSD", layout="wide")

# 2. Setup Path Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = "https://github.com/AnamRadentArjuna/projectskripsi-app/releases/download/V1.0/resnet50_baseline_best.h5"
MODEL_LOCAL_PATH = os.path.join(BASE_DIR, "resnet50_baseline_best.h5")


@st.cache_resource
def get_model():
    """
    Download model dari GitHub Releases (jika belum ada secara lokal),
    lalu load dengan Keras. Model di-cache oleh Streamlit sehingga
    proses download & load hanya terjadi sekali per sesi server.
    """
    try:
        if not os.path.exists(MODEL_LOCAL_PATH):
            with st.spinner("Mengunduh model (hanya dilakukan sekali)..."):
                response = requests.get(MODEL_URL, stream=True, timeout=60)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                progress_bar = st.progress(0)
                downloaded = 0

                with open(MODEL_LOCAL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress_bar.progress(min(downloaded / total_size, 1.0))

                progress_bar.empty()

        return load_model(MODEL_LOCAL_PATH, compile=False)

    except requests.exceptions.RequestException as e:
        st.error(f"Gagal mengunduh model dari server: {e}")
        # Hapus file parsial jika download gagal di tengah jalan
        if os.path.exists(MODEL_LOCAL_PATH):
            os.remove(MODEL_LOCAL_PATH)
        return None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


model = get_model()

st.title("Klasifikasi Penyakit Lumpy Skin Disease (LSD) pada Sapi")

if model is None:
    st.warning("Model belum siap. Periksa koneksi internet atau ketersediaan file model, lalu muat ulang halaman.")
    st.stop()

# 3. Upload File
uploaded_file = st.file_uploader("Upload Gambar Sapi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    tab1, tab2 = st.tabs(["Preprocessing", "Prediksi"])

    # --- Preprocessing dihitung sekali di luar tab, disimpan di session_state
    # agar tidak hilang/undefined saat pengguna berpindah tab.
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    st.session_state["img_clahe"] = img_clahe

    # TAB 1: PREPROCESSING
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Asli")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Resize (224x224)")
            st.image(img_resized, use_container_width=True)

        with col3:
            st.subheader("CLAHE")
            st.image(img_clahe, use_container_width=True)

        # BAGIAN AUGMENTASI (hanya ilustrasi, tidak dipakai untuk prediksi)
        st.subheader("Contoh Augmentasi Data")
        st.caption("Ilustrasi variasi gambar yang digunakan saat training untuk mencegah overfitting. "
                    "Gambar-gambar ini TIDAK digunakan dalam proses prediksi di bawah.")
        col_a, col_b, col_c, col_d = st.columns(4)

        w, h = image.size

        with col_a:
            st.write("**Rotation Range**")
            st.image(image.rotate(30), use_container_width=True)
        with col_b:
            st.write("**Brightness Range**")
            enhancer = ImageEnhance.Brightness(image)
            st.image(enhancer.enhance(1.5), use_container_width=True)
        with col_c:
            st.write("**Zoom Range**")
            zoom = 1.2
            st.image(
                image.crop((
                    w * (1 - 1 / zoom) / 2,
                    h * (1 - 1 / zoom) / 2,
                    w - w * (1 - 1 / zoom) / 2,
                    h - h * (1 - 1 / zoom) / 2,
                )).resize((w, h)),
                use_container_width=True,
            )
        with col_d:
            st.write("**Fill Mode**")
            st.image(
                image.rotate(30, expand=True, fillcolor="black").resize((w, h)),
                use_container_width=True,
            )

    # TAB 2: PREDIKSI
    with tab2:
        if st.button("Jalankan Prediksi"):
            with st.spinner("Model sedang menganalisis fitur..."):
                img_for_pred = st.session_state["img_clahe"]
                img_input = np.array(img_for_pred) / 255.0
                img_input = np.expand_dims(img_input, axis=0)

                prediksi = model.predict(img_input)
                score = float(prediksi[0][0])

                hasil = "Terinfeksi LSD" if score > 0.5 else "Normal Skin"
                probabilitas = score if score > 0.5 else (1 - score)

            st.subheader("Hasil Analisis")
            if hasil == "Normal Skin":
                st.success(f"**Prediksi: {hasil}**")
            else:
                st.error(f"**Prediksi: {hasil}**")

            st.write(f"Tingkat Keyakinan: {probabilitas * 100:.2f}%")

    # EXPANDER: ARSITEKTUR
    with st.expander("Lihat Arsitektur Model"):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        st.code("\n".join(stringlist), language=None)
