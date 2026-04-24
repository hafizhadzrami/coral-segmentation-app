import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import pandas as pd

# --- 1. SETTING PATH ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Pastikan nama fail ini sama dengan yang anda upload ke GitHub
MODEL_NAME = 'model_R10.keras'
MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_NAME)

CLASSES = ['ACP', 'DIPLO', 'FUN', 'MON', 'PORI']

# Warna RGBA untuk visualisasi grid
COLORS_RGBA = {
    'ACP': (255, 0, 0, 100), 'DIPLO': (0, 255, 0, 100), 
    'FUN': (0, 0, 255, 100), 'MON': (255, 255, 0, 100), 
    'PORI': (255, 0, 255, 100)
}

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_coral_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Fail '{MODEL_NAME}' tidak dijumpai di GitHub!")
        return None
    try:
        # Format .keras biasanya tidak perlukan compile=False, tapi kita letak untuk keselamatan
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Ralat muat turun model: {e}")
        return None

# --- 3. UI WEB ---
st.set_page_config(page_title="Coral Analysis AI", layout="centered")
st.title("🪸 Automated Coral Species Classification")
st.markdown("Analisis Karang Terengganu menggunakan Grid Segmentation (10x5)")

model = load_coral_model()

if model is not None:
    uploaded_file = st.file_uploader("Muat naik imej karang (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imej Asal", use_container_width=True)

        if st.button("Jalankan Analisis"):
            with st.spinner('Menganalisis grid...'):
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

                        preds = model.predict(cell_batch, verbose=0)
                        idx = np.argmax(preds)
                        conf = np.max(preds)
                        
                        if conf > 0.5:
                            label = CLASSES[idx]
                            counts[label] += 1
                            draw.rectangle([x1, y1, x2, y2], fill=COLORS_RGBA[label])
                        else:
                            counts['Others'] += 1
                            draw.rectangle([x1, y1, x2, y2], outline=(200,200,200,50))

                final_img = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
                st.subheader("Hasil Analisis")
                st.image(final_img, use_container_width=True)
                
                # Statistik
                st.divider()
                st.subheader("📊 Statistik Liputan")
                df_stats = [{"Spesies": k, "Bilangan Grid": v, "Liputan (%)": f"{(v/50)*100:.1f}%"} for k,v in counts.items()]
                st.table(pd.DataFrame(df_stats))

st.caption("Developed by Hafiz Hadzrami | Marine Science AI Research")
