import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import pandas as pd

# --- 1. CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(CURRENT_DIR, 'model_weights.weights.h5')
CLASSES = ['ACP', 'DIPLO', 'FUN', 'MON', 'PORI']

# --- 2. FUNGSI BINA SEMULA MODEL ---
@st.cache_resource
def load_coral_model():
    try:
        # Bina semula rangka MobileNetV2 yang Hafiz guna
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(128, 128, 3), 
            include_top=False, 
            weights=None # Tak perlu download weights asal
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        # Masukkan weights yang Hafiz dah train
        if os.path.exists(WEIGHTS_PATH):
            model.load_weights(WEIGHTS_PATH)
            return model
        else:
            st.error("Fail weights tidak dijumpai di GitHub!")
            return None
    except Exception as e:
        st.error(f"Ralat Bina Model: {e}")
        return None

# --- 3. UI WEB ---
st.set_page_config(page_title="Coral Analysis AI", layout="centered")
st.title("🪸 Automated Coral Classification")

model = load_coral_model()

if model is not None:
    uploaded_file = st.file_uploader("Muat naik imej karang", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imej Input", use_container_width=True)

        if st.button("Analisis Grid"):
            with st.spinner('Memproses...'):
                img_array = np.array(image)
                h, w, _ = img_array.shape
                rows, cols = 5, 10
                cell_h, cell_w = h // rows, w // cols

                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                counts = {cls: 0 for cls in CLASSES}
                
                # Warna untuk grid
                colors = {'ACP': (255,0,0,100), 'DIPLO': (0,255,0,100), 'FUN': (0,0,255,100), 'MON': (255,255,0,100), 'PORI': (255,0,255,100)}

                for r in range(rows):
                    for c in range(cols):
                        y1, y2 = r * cell_h, (r + 1) * cell_h
                        x1, x2 = c * cell_w, (c + 1) * cell_w
                        cell = cv2.resize(img_array[y1:y2, x1:x2], (128, 128)) / 255.0
                        
                        preds = model.predict(np.expand_dims(cell, axis=0), verbose=0)
                        idx = np.argmax(preds)
                        if np.max(preds) > 0.5:
                            label = CLASSES[idx]
                            counts[label] = counts.get(label, 0) + 1
                            draw.rectangle([x1, y1, x2, y2], fill=colors[label])

                st.image(Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB'), use_container_width=True)
                st.table(pd.DataFrame([{"Spesies": k, "Liputan": f"{(v/50)*100:.1f}%"} for k, v in counts.items()]))
