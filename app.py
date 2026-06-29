import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import pandas as pd
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['ACP', 'DIPLO', 'FUN', 'MON', 'PORI']
CORAL_MAP = {
    'ACP': {'name': 'Acropora', 'color': (255, 0, 0, 100), 'desc': 'Branching/Table Coral (Red)'},
    'DIPLO': {'name': 'Diploastrea', 'color': (0, 255, 0, 100), 'desc': 'Massive Coral (Green)'},
    'FUN': {'name': 'Fungia', 'color': (0, 0, 255, 100), 'desc': 'Mushroom Coral (Blue)'},
    'MON': {'name': 'Montipora', 'color': (255, 255, 0, 100), 'desc': 'Foliose/Encrusting Coral (Yellow)'},
    'PORI': {'name': 'Porites', 'color': (255, 0, 255, 100), 'desc': 'Massive/Finger Coral (Magenta)'}
}

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    try:
        # Ganti dengan nama fail model sebenar anda
        m_eff = tf.keras.models.load_model('efficientnet_model.h5', compile=False)
        m_res = tf.keras.models.load_model('resnet_model.h5', compile=False)
        m_mob = tf.keras.models.load_model('mobilenet_model.h5', compile=False)
        return m_eff, m_res, m_mob
    except Exception as e:
        st.error(f"Gagal memuatkan model: {e}")
        return None, None, None

# --- UI SETUP ---
st.set_page_config(page_title="CoralVision AI", layout="wide")
st.title("🪸 CoralVision AI: Multi-Model Comparison")

model_eff, model_res, model_mob = load_models()

uploaded_file = st.file_uploader("Upload Survey Image", type=["jpg", "png"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

if uploaded_file and model_eff:
    image = Image.open(uploaded_file).convert("RGB")
    
    if st.button("Run Quantitative Analysis"):
        with st.spinner('Memproses 3 model...'):
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            rows, cols = 5, 10
            cell_h, cell_w = h // rows, w // cols
            
            overlays = {name: Image.new('RGBA', image.size, (0, 0, 0, 0)) for name in ["EfficientNet", "ResNet", "MobileNet"]}
            draws = {name: ImageDraw.Draw(overlays[name]) for name in overlays}
            
            for r in range(rows):
                for c in range(cols):
                    # Prep patch
                    y1, y2 = r * cell_h, (r + 1) * cell_h
                    x1, x2 = c * cell_w, (c + 1) * cell_w
                    cell = cv2.resize(img_array[y1:y2, x1:x2], (128, 128)).astype(np.float32)
                    cell = np.expand_dims(preprocess_input(cell), axis=0)
                    
                    # Inference
                    preds_list = [model_eff.predict(cell, verbose=0), 
                                  model_res.predict(cell, verbose=0), 
                                  model_mob.predict(cell, verbose=0)]
                    
                    for name, preds in zip(overlays.keys(), preds_list):
                        idx = np.argmax(preds)
                        if np.max(preds) >= conf_threshold:
                            label = CLASSES[idx]
                            draws[name].rectangle([x1, y1, x2, y2], fill=CORAL_MAP[label]['color'], outline="white")

            # Display Results
            cols_ui = st.columns(3)
            results = []
            for i, name in enumerate(overlays.keys()):
                res_img = Image.alpha_composite(image.convert('RGBA'), overlays[name]).convert('RGB')
                cols_ui[i].image(res_img, caption=f"{name} Result", use_container_width=True)
