import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import pandas as pd

# --- 1. CONFIGURATION ---
# Pastikan fail model_R10.h5 diupload ke folder yang sama di GitHub
MODEL_PATH = 'model_R10.h5'
CLASSES = ['ACP', 'DIPLO', 'FUN', 'MON', 'PORI']

# Warna untuk visualisasi grid (RGBA)
COLORS_RGBA = {
    'ACP': (255, 0, 0, 100),     # Merah
    'DIPLO': (0, 255, 0, 100),   # Hijau
    'FUN': (0, 0, 255, 100),     # Biru
    'MON': (255, 255, 0, 100),   # Kuning
    'PORI': (255, 0, 255, 100)   # Magenta
}

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_coral_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        # Load model tanpa compile untuk elakkan isu custom objects/optimizers
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Ralat memuatkan model: {e}")
        return None

# --- 3. UI WEB STREAMLIT ---
st.set_page_config(page_title="Coral Grid Segmentation", layout="centered")
st.title("🪸 Automated Coral Species Classification")
st.markdown("""
Sistem ini menggunakan arkitektur **MobileNetV2** untuk mengelaskan spesies karang secara automatik menggunakan kaedah **10x5 Grid Segmentation**.
""")

model = load_coral_model()

if model is None:
    st.error(f"⚠️ Fail '{MODEL_PATH}' tidak dijumpai di repository GitHub!")
    st.info("Sila pastikan anda telah upload fail model_R10.h5 ke GitHub anda.")
else:
    uploaded_file = st.file_uploader("Muat naik imej karang (JPG/PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imej Asal", use_container_width=True)

        if st.button("Analyse & Calculate Coverage"):
            with st.spinner('Sedang menganalisis 50 patches...'):
                img_array = np.array(image)
                h, w, _ = img_array.shape
                rows, cols = 5, 10
                cell_h, cell_w = h // rows, w // cols

                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)

                # Dictionary untuk statistik
                species_counts = {cls: 0 for cls in CLASSES}
                species_counts['Others/Unknown'] = 0

                # Proses setiap grid
                for r in range(rows):
                    for c in range(cols):
                        y1, y2 = r * cell_h, (r + 1) * cell_h
                        x1, x2 = c * cell_w, (c + 1) * cell_w

                        cell = img_array[y1:y2, x1:x2]
                        cell_resized = cv2.resize(cell, (128, 128))
                        cell_norm = cell_resized / 255.0
                        cell_batch = np.expand_dims(cell_norm, axis=0)

                        prediction = model.predict(cell_batch, verbose=0)
                        idx = np.argmax(prediction)
                        conf = np.max(prediction)
                        label = CLASSES[idx]

                        # Logik Confidence Threshold (Reject noise/sand)
                        if conf > 0.5:
                            species_counts[label] += 1
                            draw.rectangle([x1, y1, x2, y2], fill=COLORS_RGBA[label])
                            draw.rectangle([x1, y1, x2, y2], outline=(255,255,255,180), width=2)
                        else:
                            species_counts['Others/Unknown'] += 1
                            draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200, 50), width=1)

                # Papar hasil gabungan
                result_image = Image.alpha_composite(image.convert('RGBA'), overlay)
                st.subheader("Visual Analysis: 10x5 Grid Overlay")
                st.image(result_image.convert('RGB'), use_container_width=True)

                # --- 4. BAHAGIAN STATISTIK ---
                st.divider()
                col1, col2 = st.columns([1, 1])

                total_grid = 50 
                stats_list = []
                for sp, count in species_counts.items():
                    percentage = (count / total_grid) * 100
                    stats_list.append({"Species": sp, "Grids": count, "Percentage": f"{percentage:.1f}%"})

                with col1:
                    st.write("📊 **Coral Coverage Summary**")
                    st.table(pd.DataFrame(stats_list))

                with col2:
                    st.write("📈 **Distribution Chart**")
                    chart_df = pd.DataFrame({
                        'Species': list(species_counts.keys()),
                        'Count': list(species_counts.values())
                    })
                    st.bar_chart(data=chart_df.set_index('Species'))

st.caption("Developed by Hafiz Hadzrami | Marine Science AI Research")
