import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import pandas as pd

# --- 1. SETTING PATH & CONFIG ---
# Paksa sistem cari path fail secara dinamik (elak isu 'file not found')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'model_R10.h5'
MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_NAME)

CLASSES = ['ACP', 'DIPLO', 'FUN', 'MON', 'PORI']

# Warna untuk visualisasi grid (RGBA)
COLORS_RGBA = {
    'ACP': (255, 0, 0, 100),     # Merah
    'DIPLO': (0, 255, 0, 100),   # Hijau
    'FUN': (0, 0, 255, 100),     # Biru
    'MON': (255, 255, 0, 100),   # Kuning
    'PORI': (255, 0, 255, 100)   # Magenta
}

# --- 2. FUNGSI MUAT TURUN MODEL (STABIL) ---
@st.cache_resource
def load_coral_model():
    if not os.path.exists(MODEL_PATH):
        # Debugging: Papar senarai fail jika model hilang
        files_in_dir = os.listdir(CURRENT_DIR)
        st.error(f"Fail '{MODEL_NAME}' tidak dijumpai di server!")
        st.write(f"Fail yang ada dalam folder: {files_in_dir}")
        return None
    
    try:
        # Gunakan compile=False untuk elakkan ralat optimizers/batch_shape
        # Ini adalah cara paling selamat untuk load model .h5 di Streamlit
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Ralat teknikal semasa memuatkan model: {e}")
        st.info("Punca: Versi TensorFlow di Streamlit mungkin berbeza dengan versi masa anda train model.")
        return None

# --- 3. ANTARAMUKA (UI) ---
st.set_page_config(page_title="Coral Analysis AI", layout="centered")
st.title("🪸 Automated Coral Species Classification")
st.markdown("Sistem Prototaip: Klasifikasi Karang Terengganu (MobileNetV2)")

# Load model secara senyap
model = load_coral_model()

if model is not None:
    uploaded_file = st.file_uploader("Pilih imej karang (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imej Input", use_container_width=True)

        if st.button("Jalankan Analisis Grid"):
            with st.spinner('Menganalisis data...'):
                img_array = np.array(image)
                h, w, _ = img_array.shape
                rows, cols = 5, 10
                cell_h, cell_w = h // rows, w // cols

                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)

                # Simpan kiraan spesies
                species_counts = {cls: 0 for cls in CLASSES}
                species_counts['Others'] = 0

                # Loop setiap grid
                for r in range(rows):
                    for c in range(cols):
                        y1, y2 = r * cell_h, (r + 1) * cell_h
                        x1, x2 = c * cell_w, (c + 1) * cell_w

                        cell = img_array[y1:y2, x1:x2]
                        cell_resized = cv2.resize(cell, (128, 128))
                        cell_norm = cell_resized / 255.0
                        cell_batch = np.expand_dims(cell_norm, axis=0)

                        # Predict
                        preds = model.predict(cell_batch, verbose=0)
                        idx = np.argmax(preds)
                        conf = np.max(preds)
                        label = CLASSES[idx]

                        if conf > 0.5:
                            species_counts[label] += 1
                            draw.rectangle([x1, y1, x2, y2], fill=COLORS_RGBA[label])
                            draw.rectangle([x1, y1, x2, y2], outline=(255,255,255,150), width=2)
                        else:
                            species_counts['Others'] += 1
                            draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200, 50), width=1)

                # Gabung gambar asal dengan grid
                result_img = Image.alpha_composite(image.convert('RGBA'), overlay)
                st.subheader("Hasil Grid Segmentation (10x5)")
                st.image(result_img.convert('RGB'), use_container_width=True)

                # Papar statistik
                st.divider()
                st.subheader("📊 Statistik Coverage")
                
                df_stats = []
                for sp, count in species_counts.items():
                    pct = (count / 50) * 100
                    df_stats.append({"Spesies": sp, "Bilangan Grid": count, "Peratusan (%)": f"{pct:.1f}%"})
                
                st.table(pd.DataFrame(df_stats))
                
                # Bar Chart
                chart_data = pd.DataFrame({
                    'Species': list(species_counts.keys()),
                    'Grids': list(species_counts.values())
                })
                st.bar_chart(data=chart_data.set_index('Species'))

st.caption("Developed by Hafiz Hadzrami | Master of Marine Science Research")
