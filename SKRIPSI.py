import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Penyakit Kulit Sapi", layout="centered")

st.title("Sistem Klasifikasi Lumpy Skin")
st.write("Unggah gambar kulit sapi untuk mendeteksi apakah Normal atau Lumpy Skin.")

# 2. Fungsi Load Model
@st.cache_resource
def load_model():
    # Sesuaikan path dengan model yang sudah Anda simpan di Drive/Local
    model = tf.keras.models.load_model('model/resnet50_baseline_best.h5')
    return model

model = load_model()

# 3. Fungsi Preprocessing (sesuai diagram Anda)
def preprocess_image(image):
    # Resize
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # CLAHE (Opsional: tambahkan jika ingin hasil yang sama persis dengan training)
    # img_array = apply_clahe(img_array) 
    
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0) # Batch dimension
    return img_array

# 4. Antarmuka Unggah Gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)
    
    if st.button('Klasifikasi'):
        # Proses
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)
        
        # Interpretasi hasil (Sigmoid output)
        score = prediction[0][0]
        if score < 0.5:
            result = f"Normal Skin (Confidence: {1-score:.2%})"
        else:
            result = f"Lumpy Skin (Confidence: {score:.2%})"
            
        st.success(f"Hasil Prediksi: {result}")