import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import pandas as pd
import gdown

# --- 1. CONFIGURATION & AUTO-DOWNLOAD ---
MODEL_PATH = 'model_R10.h5'
# File ID yang anda berikan dari Google Drive
GDRIVE_FILE_ID = '1mxVCj7mmzT_wIDH1rpi9fhmXP5lLLtR1'

@st.cache_resource
def download_and_load_model():
    # Jika model belum ada dalam server Streamlit, download automatik
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Initializing AI Model... This may take a minute for the first time.'):
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    try:
        # Load model tanpa compile untuk kestabilan server
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

CLASSES = ['ACP', 'DIPLO', 'FUN', 'MON', 'PORI']

# Warna Semi-Transparent untuk Grid (RGBA)
COLORS_RGBA = {
    'ACP': (255, 0, 0, 100),     # Red
    'DIPLO': (0, 255, 0, 100),   # Green
    'FUN': (0, 0, 255, 100),     # Blue
    'MON': (255, 255, 0, 100),   # Yellow
    'PORI': (255, 0, 255, 100)   # Magenta
}

# --- 2. UI WEB STREAMLIT ---
st.set_page_config(page_title="Coral Grid Segmentation", layout="centered")
st.title("🪸 Automated Coral Species Classification")
st.markdown("""
Sistem ini menggunakan **MobileNetV2** yang dilatih melalui **Active Learning** untuk mengesan dan mengira peratusan liputan spesies karang (coral coverage) secara automatik menggunakan kaedah **10x5 Grid Segmentation**.
""")

model = download_and_load_model()

if model is not None:
    uploaded_file = st.file_uploader("Upload Underwater Coral Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Input Image", use_container_width=True)

        if st.button("Run Segmentation & Coverage Analysis"):
            with st.spinner('Processing 50 image patches...'):
                img_array = np.array(image)
                h, w, _ = img_array.shape
                rows, cols = 5, 10
                cell_h, cell_w = h // rows, w // cols

                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)

                # Dictionary untuk statistik
                species_counts = {cls: 0 for cls in CLASSES}
                species_counts['Others/Unknown'] = 0

                # Proses Analisis Grid
                for r in range(rows):
                    for c in range(cols):
                        y1, y2 = r * cell_h, (r + 1) * cell_h
                        x1, x2 = c * cell_w, (c + 1) * cell_w

                        # Preprocessing patch
                        cell = img_array[y1:y2, x1:x2]
                        cell_resized = cv2.resize(cell, (128, 128))
                        cell_norm = cell_resized / 255.0
                        cell_batch = np.expand_dims(cell_norm, axis=0)

                        # Prediction
                        prediction = model.predict(cell_batch, verbose=0)
                        idx = np.argmax(prediction)
                        conf = np.max(prediction)
                        label = CLASSES[idx]

                        # Threshold 0.5 untuk reject noise/background
                        if conf > 0.5:
                            species_counts[label] += 1
                            draw.rectangle([x1, y1, x2, y2], fill=COLORS_RGBA[label])
                            draw.rectangle([x1, y1, x2, y2], outline=(255,255,255,180), width=2)
                        else:
                            species_counts['Others/Unknown'] += 1
                            draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200, 50), width=1)

                # Display Results
                result_combined = Image.alpha_composite(image.convert('RGBA'), overlay)
                st.subheader("Visual Analysis: 10x5 Grid Overlay")
                st.image(result_combined.convert('RGB'), use_container_width=True)

                # --- STATISTIK ---
                st.divider()
                col1, col2 = st.columns([1, 1])

                total_grid = 50
                stats_list = []
                for sp, count in species_counts.items():
                    percentage = (count / total_grid) * 100
                    stats_list.append({"Species": sp, "Grids": count, "Coverage (%)": f"{percentage:.1f}%"})

                with col1:
                    st.write("📊 **Coverage Summary**")
                    st.table(pd.DataFrame(stats_list))
                
                with col2:
                    st.write("📈 **Distribution Chart**")
                    chart_df = pd.DataFrame({'Species': list(species_counts.keys()), 'Count': list(species_counts.values())})
                    st.bar_chart(data=chart_df.set_index('Species'))

st.info("Note: System performance is optimized for coral species from Terengganu research sites.")