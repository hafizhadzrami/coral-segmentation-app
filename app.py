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

# Custom CSS for Professional Theme
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1E3A8A; color: white; font-weight: bold; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    h1 { color: #1E3A8A; }
    </style>
    """, unsafe_allow_html=True)

st.title("🪸 CoralVision AI: Automated Ecological Analysis")
st.markdown("#### Marine Science Research | Quantitative Benthic Cover Estimation")
st.divider()

# Sidebar for Settings and Legend
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    # Fitur Utama: Confidence Score Filter
    conf_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="Higher threshold means stricter classification. Only high-confidence predictions will be labeled."
    )
    
    st.divider()
    st.header("🔍 Genus Legend")
    for code, info in CORAL_MAP.items():
        st.markdown(f"**{info['name']}**")
        st.caption(info['desc'])
    
    st.divider()
    st.info("System identifies coral genus across a 10x5 grid (50 samples per image).")

model = load_coral_model()

if model is None:
    st.error(f"Critical Error: Weights file '{WEIGHTS_NAME}' missing in repository.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload Survey Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Survey Image", use_container_width=True)

    if uploaded_file is not None:
        with col2:
            if st.button("Run Quantitative Analysis"):
                with st.spinner('AI is analyzing coral patches...'):
                    img_array = np.array(image)
                    h, w, _ = img_array.shape
                    rows, cols = 5, 10
                    cell_h, cell_w = h // rows, w // cols

                    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                    draw = ImageDraw.Draw(overlay)
                    
                    counts = {cls: 0 for cls in CLASSES}
                    counts['Uncertain/Others'] = 0

                    # Analysis Loop
                    for r in range(rows):
                        for c in range(cols):
                            y1, y2 = r * cell_h, (r + 1) * cell_h
                            x1, x2 = c * cell_w, (c + 1) * cell_w
                            cell = cv2.resize(img_array[y1:y2, x1:x2], (128, 128)) / 255.0
                            
                            preds = model.predict(np.expand_dims(cell, axis=0), verbose=0)
                            idx = np.argmax(preds)
                            conf = np.max(preds)
                            
                            # Apply Confidence Filter
                            if conf >= conf_threshold:
                                label_code = CLASSES[idx]
                                genus_name = CORAL_MAP[label_code]['name']
                                counts[label_code] += 1
                                
                                # Visualization
                                draw.rectangle([x1, y1, x2, y2], fill=CORAL_MAP[label_code]['color'], outline=(255,255,255,180), width=2)
                                draw.text((x1 + 5, y1 + 5), f"{genus_name}", fill=(255,255,255,255))
                            else:
                                counts['Uncertain/Others'] += 1
                                draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200, 80), width=1)

                    result_img = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
                    st.image(result_img, caption=f"Analysis Result (Threshold: {conf_threshold})", use_container_width=True)

        # Statistical Summary
        if 'result_img' in locals():
            st.divider()
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.subheader("📊 Quantitative Results")
                stats_data = []
                for sp, count in counts.items():
                    name = CORAL_MAP[sp]['name'] if sp in CORAL_MAP else sp
                    pct = (count / 50) * 100
                    stats_data.append({"Coral Genus": name, "Grid Count": count, "Benthic Cover (%)": f"{pct:.1f}%"})
                st.table(pd.DataFrame(stats_data))

            with res_col2:
                st.subheader("📈 Genus Distribution")
                chart_df = pd.DataFrame({
                    'Genus': [CORAL_MAP[k]['name'] if k in CORAL_MAP else k for k in counts.keys()],
                    'Count': list(counts.values())
                })
                st.bar_chart(data=chart_df.set_index('Genus'))

st.markdown("---")
st.caption("Marine Science Research | Automated Monitoring Pipeline | Hafiz Hadzrami")
