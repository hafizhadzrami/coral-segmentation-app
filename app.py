import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import pandas as pd

# --- 1. CONFIGURATION ---
# Gunakan fail weights yang kita dah download dari Colab & upload ke GitHub
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_NAME = 'model_weights.weights.h5'
WEIGHTS_PATH = os.path.join(CURRENT_DIR, WEIGHTS_NAME)

CLASSES = ['ACP', 'DIPLO', 'FUN', 'MON', 'PORI']

COLORS_RGBA = {
    'ACP': (255, 0, 0, 90),     # Merah
    'DIPLO': (0, 255, 0, 90),   # Hijau
    'FUN': (0, 0, 255, 90),     # Biru
    'MON': (255, 255, 0, 90),   # Kuning
    'PORI': (255, 0, 255, 90)   # Magenta
}

BORDER_COLOR = (255, 255, 255, 150)
TEXT_COLOR = (255, 255, 255, 255)

# --- 2. FUNGSI BINA SEMULA MODEL (STABIL) ---
@st.cache_resource
def load_coral_model():
    if not os.path.exists(WEIGHTS_PATH):
        return None
    try:
        # Bina semula rangka MobileNetV2 sebiji macam dalam PhD project anda
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(128, 128, 3), 
            include_top=False, 
            weights=None 
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        # Masukkan "ilmu" (weights) ke dalam rangka
        model.load_weights(WEIGHTS_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

# --- 3. UI WEB STREAMLIT ---
st.set_page_config(page_title="Coral Grid Segmentation", layout="centered")
st.title("🪸 Coral Grid Segmentation & Coverage Estimation")

model = load_coral_model()

if model is None:
    st.error(f"⚠️ Fail weights '{WEIGHTS_NAME}' tidak dijumpai di repository!")
    st.info("Pastikan anda telah upload fail 'model_weights.weights.h5' ke GitHub.")
else:
    uploaded_file = st.file_uploader("Pilih imej karang (JPG/PNG)...", type=["jpg", "jpeg", "png"])

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

                species_counts = {cls: 0 for cls in CLASSES}
                species_counts['Others'] = 0

                try:
                    font = ImageFont.load_default()
                except:
                    font = None

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
                        
                        if conf > 0.5:
                            label = CLASSES[idx]
                            species_counts[label] += 1
                            color_rgba = COLORS_RGBA[label]
                            draw.rectangle([x1, y1, x2, y2], fill=color_rgba)
                            draw.rectangle([x1, y1, x2, y2], outline=BORDER_COLOR, width=2)
                            if font:
                                draw.text((x1 + 5, y1 + 5), label, fill=TEXT_COLOR, font=font)
                        else:
                            species_counts['Others'] += 1
                            draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200, 50), width=1)

                result_image = Image.alpha_composite(image.convert('RGBA'), overlay)
                st.subheader("Hasil Grid Segmentation")
                st.image(result_image.convert('RGB'), use_container_width=True)

                # --- 4. STATISTIK ---
                st.divider()
                st.subheader("📊 Coral Coverage Estimation")

                total_grid = rows * cols
                stats_list = []
                for sp, count in species_counts.items():
                    percentage = (count / total_grid) * 100
                    stats_list.append({"Species": sp, "Grid Count": count, "Percentage": f"{percentage:.1f}%"})

                df = pd.DataFrame(stats_list)
                st.table(df)

                chart_df = pd.DataFrame({
                    'Species': list(species_counts.keys()),
                    'Count': list(species_counts.values())
                })
                st.bar_chart(data=chart_df.set_index('Species'))

st.caption("Developed by Hafiz Hadzrami | PhD Marine Science Research Pipeline")
