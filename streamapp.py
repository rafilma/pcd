import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import tempfile
import os
import gdown

# --- Fungsi untuk download dan load model ---
@st.cache_resource
def download_and_load_model():
    file_id = "1h7Tfhs6CKN6i4RJ-GdTQr1B4eiArMVCY"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    st.info("🔄 Mengunduh model dari Google Drive menggunakan gdown...")
    tmp_path = os.path.join(tempfile.gettempdir(), "Augmentasi_Model.h5")
    gdown.download(download_url, tmp_path, quiet=False)

    st.success("✅ Model berhasil diunduh!")
    model = tf.keras.models.load_model(tmp_path)
    return model

# Load model hanya sekali
model = download_and_load_model()

# --- Judul Aplikasi ---
st.title("🌿 Herbal Leaf Classification")
st.write("""
Aplikasi Pengolahan Citra Digital Oleh Kelompok 1
""")
st.write("""
Upload gambar daun herbal, dan model deep learning akan memprediksi jenis daun tersebut.
""")

# --- Upload Gambar ---
uploaded_file = st.file_uploader("Upload gambar daun herbal", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Buka gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    # --- Preprocessing ---
    target_size = (150, 150)  # Sesuaikan dengan ukuran input model
    img_resized = image.resize(target_size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

    # --- Prediksi ---
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)  # Nilai confidence tertinggi

    # --- Mapping Index ke Label ---
    class_labels = [
        "Belimbing Wuluh", "Jambu Biji", "Jeruk", "Kemangi", "Lidah Buaya",
        "Nangka", "Pandan", "Pepaya", "Seledri", "Sirih"
    ]

    st.subheader("Hasil Prediksi")

    # Jika confidence < 70%, dianggap bukan tanaman herbal
    threshold = 0.70
    if confidence < threshold:
        st.error("❌ Gambar ini **bukan tanaman herbal** atau tidak dikenali oleh model.")
        st.write(f"Confidence terlalu rendah: **{confidence * 100:.2f}%**")
    else:
        predicted_label = class_labels[predicted_class]
        st.write(f"🌱 **Jenis daun terdeteksi:** {predicted_label}")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

elif uploaded_file is None:
    st.info("Silakan upload gambar daun herbal untuk memulai prediksi.")
