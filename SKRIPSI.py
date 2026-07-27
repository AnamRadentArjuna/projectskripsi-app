import streamlit as st
import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi LSD", layout="wide")

# 2. Setup Path Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "https://github.com/AnamRadentArjuna/projectskripsi-app/releases/download/V1.0/resnet50_baseline_best.h5"

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan di: {MODEL_PATH}")
        return None
    return load_model(MODEL_PATH, compile=False)

model = get_model()

st.title("Klasifikasi Penyakit Lumpy Skin Disease (LSD) pada Sapi")

# 3. Upload File (Dihapus accept_multiple_files=True)
uploaded_file = st.file_uploader("Upload Gambar Sapi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    tab1, tab2 = st.tabs(["Preprocessing", "Prediksi"])

    # TAB 1: PREPROCESSING
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Asli")
            st.image(image, use_container_width=True)
        
        # Resize
        img_resized = image.resize((224, 224))
        with col2:
            st.subheader("Resize (224x224)")
            st.image(img_resized, use_container_width=True)
        
        # CLAHE
        img_array = np.array(img_resized)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        with col3:
            st.subheader("CLAHE")
            st.image(img_clahe, use_container_width=True)

        # BAGIAN AUGMENTASI
        st.subheader("Contoh Augmentasi Data")
        st.write("Variasi gambar untuk mencegah overfitting:")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.write("**Rotation Range**")
            st.image(image.rotate(30), use_container_width=True)
        with col_b:
            st.write("**Brightness Range**")
            enhancer = ImageEnhance.Brightness(image)
            st.image(enhancer.enhance(1.5), use_container_width=True)
        with col_c:
            st.write("**Zoom Range**")
            w, h = image.size
            zoom = 1.2
            st.image(image.crop((w*(1-1/zoom)/2, h*(1-1/zoom)/2, w-w*(1-1/zoom)/2, h-h*(1-1/zoom)/2)).resize((w, h)), use_container_width=True)
        with col_d:
            st.write("**Fill Mode**")
            st.image(image.rotate(30, expand=True, fillcolor="black").resize((w, h)), use_container_width=True)

    # TAB 2: PREDIKSI
    with tab2:
        if st.button("Jalankan Prediksi"):
            with st.spinner("Model sedang menganalisis fitur..."):
                img_input = np.array(img_clahe) / 255.0
                img_input = np.expand_dims(img_input, axis=0)
                
                prediksi = model.predict(img_input)
                score = prediksi[0][0]
                
                hasil = "Terinfeksi LSD" if score > 0.5 else "Normal Skin"
                probabilitas = score if score > 0.5 else (1 - score)
            
            st.subheader("Hasil Analisis")
            if hasil == "Normal Skin":
                st.success(f"**Prediksi: {hasil}**")
            else:
                st.error(f"**Prediksi: {hasil}**")
            
            st.write(f"Tingkat Keyakinan: {probabilitas*100:.2f}%")

    # EXPANDER: ARSITEKTUR
    with st.expander("Lihat Arsitektur Model"):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        st.code("\n".join(stringlist), language=None)

elif uploaded_file is not None and model is None:
    st.error("Mohon pastikan file model (.h5) sudah berada di folder yang sama dengan app.py.")
