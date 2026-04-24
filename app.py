import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import pandas as pd

# --- 1. SETTING PATH ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'model_R10.h5'
MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_NAME)

CLASSES = ['ACP', 'DIPLO', 'FUN', 'MON', 'PORI']

# Warna RGBA untuk Grid
COLORS_RGBA = {
    'ACP': (255, 0, 0, 100), 'DIPLO': (0, 255, 0, 100), 
    'FUN': (0, 0, 255, 100), 'MON': (255, 255, 0, 100), 
    'PORI': (255, 0, 255, 100)
}

# --- 2. LOAD MODEL (CARA PALING SELAMAT) ---
@st.cache_resource
def load_coral_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Fail {MODEL_NAME} tidak dijumpai!")
        return None
    try:
        # Kita guna 'compile=False' dan biarkan Keras 2 handle
        # TensorFlow 2.13 secara automatik guna Keras 2, jadi isu quantization_config patut hilang
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        # Jika masih error, kita bagi arahan alternatif
        st.error(f"Ralat Versi: {e}")
        st.info("Sila pastikan requirements.txt anda menggunakan tensorflow==2.13.0")
        return None

# --- 3. UI WEB ---
st.set_page_config(page_title="Coral Analysis AI", layout="centered")
st.title("🪸 Automated Coral Species Classification")

model = load_coral_model()

if model is not None:
    uploaded_file = st.file_uploader("Muat naik imej karang (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imej Input", use_container_width=True)

        if st.button("Jalankan Analisis Grid"):
            with st.spinner('Sedang menganalisis 50 patches...'):
                img_array = np.array(image)
                h, w, _ = img_array.shape
                rows, cols = 5, 10
                cell_h, cell_w = h // rows, w // cols

                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                counts = {cls: 0 for cls in CLASSES}
                counts['Others'] = 0

                for r in range(rows):
                    for c in range(cols):
                        y1, y2 = r * cell_h, (r + 1) * cell_h
                        x1, x2 = c * cell_w, (c + 1) * cell_w
                        
                        cell = img_array[y1:y2, x1:x2]
                        cell_resized = cv2.resize(cell, (128, 128))
                        cell_norm = cell_resized / 255.0
                        cell_batch = np.expand_dims(cell_norm, axis=0)

                        # Predict menggunakan model yang di-load
                        preds = model.predict(cell_batch, verbose=0)
                        idx = np.argmax(preds)
                        conf = np.max(preds)
                        
                        if conf > 0.5:
                            label = CLASSES[idx]
                            counts[label] += 1
                            draw.rectangle([x1, y1, x2, y2], fill=COLORS_RGBA[label])
                            draw.rectangle([x1, y1, x2, y2], outline=(255,255,255,150), width=2)
                        else:
                            counts['Others'] += 1
                            draw.rectangle([x1, y1, x2, y2], outline=(200,200,200,50), width=1)

                # Papar hasil gabungan
                final_img = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
                st.subheader("Hasil Analisis Grid (10x5)")
                st.image(final_img, use_container_width=True)
                
                # Papar jadual statistik
                st.divider()
                st.subheader("📊 Statistik Liputan Karang")
                df_stats = []
                for sp, count in counts.items():
                    pct = (count / 50) * 100
                    df_stats.append({"Spesies": sp, "Bilangan Grid": count, "Peratusan (%)": f"{pct:.1f}%"})
                
                st.table(pd.DataFrame(df_stats))
