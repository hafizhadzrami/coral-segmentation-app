import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import pandas as pd

# --- 1. CONFIGURATION & MAPPING ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_NAME = 'model_weights.weights.h5'
WEIGHTS_PATH = os.path.join(CURRENT_DIR, WEIGHTS_NAME)

# Professional Mapping
CORAL_MAP = {
    'ACP': {'name': 'Acropora', 'color': (255, 0, 0, 100), 'desc': 'Branching/Table Coral (Red)'},
    'DIPLO': {'name': 'Diploastrea', 'color': (0, 255, 0, 100), 'desc': 'Massive Coral (Green)'},
    'FUN': {'name': 'Fungia', 'color': (0, 0, 255, 100), 'desc': 'Mushroom Coral (Blue)'},
    'MON': {'name': 'Montipora', 'color': (255, 255, 0, 100), 'desc': 'Foliose/Encrusting Coral (Yellow)'},
    'PORI': {'name': 'Porites', 'color': (255, 0, 255, 100), 'desc': 'Massive/Finger Coral (Magenta)'}
}

CLASSES = ['ACP', 'DIPLO', 'FUN', 'MON', 'PORI']

# --- 2. MODEL RECONSTRUCTION ---
@st.cache_resource
def load_coral_model():
    if not os.path.exists(WEIGHTS_PATH):
        return None
    try:
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
        model.load_weights(WEIGHTS_PATH)
        return model
    except Exception as e:
        st.error(f"Model Error: {e}")
        return None

# --- 3. PROFESSIONAL UI ---
st.set_page_config(page_title="CoralVision AI", page_icon="🪸", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("🪸 CoralVision AI: Automated Ecological Analysis")
st.markdown("### Marine Science Research | Quantitative Benthic Cover Estimation")
st.divider()

# Sidebar for Legend and Info
with st.sidebar:
    st.header("Color Legend")
    for code, info in CORAL_MAP.items():
        st.markdown(f"**{info['name']}**")
        st.caption(f"{info['desc']}")
        st.divider()
    st.info("System uses a 10x5 grid (50 patches) to estimate percentage cover.")

model = load_coral_model()

if model is None:
    st.error(f"Critical Error: Weights file '{WEIGHTS_NAME}' missing.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload Survey Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Input", use_container_width=True)

    if uploaded_file is not None:
        with col2:
            if st.button("Generate Analysis Report"):
                with st.spinner('Processing neural network inference...'):
                    img_array = np.array(image)
                    h, w, _ = img_array.shape
                    rows, cols = 5, 10
                    cell_h, cell_w = h // rows, w // cols

                    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                    draw = ImageDraw.Draw(overlay)
                    counts = {cls: 0 for cls in CLASSES}
                    counts['Others/Sand'] = 0

                    # Analysis Loop
                    for r in range(rows):
                        for c in range(cols):
                            y1, y2 = r * cell_h, (r + 1) * cell_h
                            x1, x2 = c * cell_w, (c + 1) * cell_w
                            cell = cv2.resize(img_array[y1:y2, x1:x2], (128, 128)) / 255.0
                            
                            preds = model.predict(np.expand_dims(cell, axis=0), verbose=0)
                            idx = np.argmax(preds)
                            conf = np.max(preds)
                            
                            if conf > 0.5:
                                label_code = CLASSES[idx]
                                full_name = CORAL_MAP[label_code]['name']
                                counts[label_code] += 1
                                draw.rectangle([x1, y1, x2, y2], fill=CORAL_MAP[label_code]['color'], outline=(255,255,255,150), width=2)
                                draw.text((x1 + 5, y1 + 5), full_name, fill=(255,255,255,255))
                            else:
                                counts['Others/Sand'] += 1
                                draw.rectangle([x1, y1, x2, y2], outline=(200,200,200,50), width=1)

                    result_img = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
                    st.image(result_img, caption="Classified Grid Result", use_container_width=True)

        # Bottom Results Section
        if 'result_img' in locals():
            st.divider()
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.subheader("📊 Statistical Breakdown")
                stats_data = []
                for sp, count in counts.items():
                    name = CORAL_MAP[sp]['name'] if sp in CORAL_MAP else sp
                    pct = (count / 50) * 100
                    stats_data.append({"Coral Genus": name, "Grid Count": count, "Cover (%)": f"{pct:.1f}%"})
                st.table(pd.DataFrame(stats_data))

            with res_col2:
                st.subheader("📈 Visual Distribution")
                chart_df = pd.DataFrame({
                    'Genus': [CORAL_MAP[k]['name'] if k in CORAL_MAP else k for k in counts.keys()],
                    'Count': list(counts.values())
                })
                st.bar_chart(data=chart_df.set_index('Genus'))

st.markdown("---")
st.caption("Marine Science Research | Automated Monitoring Pipeline | Hafiz Hadzrami")
