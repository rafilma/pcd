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
    file_id = "1rZsWtsWbPidckehjjMeQuwHKxi_E8bgQ"
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

# --- Informasi Anggota Kelompok ---
st.markdown("""
**Anggota Kelompok 1:**
1. Rafil Moehamad Alif - 211351116  
2. Chandra Alnando - 241352003  
3. Nida Dhiya U - 221351147  
4. Syerly Novebriana L - 221351108  
""")

st.write("""
Upload gambar daun herbal, dan model deep learning akan memprediksi jenis daun tersebut.
""")

# --- Mapping Index ke Label ---
class_labels = [
    "Daun Belimbing Wuluh", "Jambu Biji", "Daun Jeruk", "Kemangi", "Lidah Buaya",
    "Daun Nangka", "Pandan", "Daun Pepaya", "Seledri", "Sirih"
]

# --- Manfaat Tanaman Herbal ---
herbal_benefits = {
    "Daun Belimbing Wuluh": "Menurunkan tekanan darah tinggi dan kadar gula darah, mengatasi peradangan, menjaga kesehatan kulit dan mata, serta bersifat antibakteri untuk membantu menyembuhkan infeksi ringan seperti jerawat dan sakit gigi.",
    "Jambu Biji": "Meningkatkan daya tahan tubuh, mempercepat penyembuhan luka, dan baik untuk pencernaan.",
    "Daun Jeruk": "Meningkatkan sistem imun, meredakan stres, mendukung kesehatan pencernaan, memiliki sifat anti-inflamasi, menyegarkan napas, serta menjaga kesehatan kulit dan rambut. ",
    "Kemangi": "Mengurangi bau mulut, melancarkan pencernaan, dan sebagai antioksidan alami.",
    "Lidah Buaya": "Menyembuhkan luka bakar, melembabkan kulit, dan meningkatkan kesehatan rambut.",
    "Daun Nangka": "Mempercepat penyembuhan luka, mengelola kadar gula darah, meredakan peradangan, dan mengatasi masalah kulit seperti jerawat karena kandungan antioksidan, antibakteri, dan antiradangnya.",
    "Pandan": "Membantu mengurangi stres, mengatasi nyeri sendi, dan meningkatkan nafsu makan.",
    "Daun Pepaya": "Mengobati demam berdarah, meningkatkan kekebalan tubuh, membantu melancarkan pencernaan, dan mengurangi nyeri haid.",
    "Seledri": "Menurunkan tekanan darah, menjaga kesehatan ginjal, dan sebagai anti-inflamasi.",
    "Sirih": "Mengatasi bau mulut, mempercepat penyembuhan luka, dan sebagai antiseptik alami."
}

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

    st.subheader("Hasil Prediksi")

    # Jika confidence < 70%, dianggap bukan tanaman herbal
    threshold = 0.87
    if confidence < threshold:
        st.error("❌ Gambar ini **bukan tanaman herbal** atau tidak dikenali oleh model.")
        st.write(f"Confidence terlalu rendah")
    else:
        predicted_label = class_labels[predicted_class]
        manfaat = herbal_benefits.get(predicted_label, "Manfaat belum tersedia.")
        
        st.write(f"🌱 **Jenis daun terdeteksi:** {predicted_label}")
        st.write(f"{predicted_label} adalah tanaman herbal yang bermanfaat untuk: **{manfaat}**")

elif uploaded_file is None:
    st.info("Silakan upload gambar daun herbal untuk memulai prediksi.")








