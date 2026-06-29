import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from data_utils import parse_uploaded_file, preprocess_lightcurve, generate_synthetic_data, sliding_window_preprocess
from model import ExoplanetPipeline, train_pipeline

# Configure page layout
st.set_page_config(layout="wide", page_title="AI Exoplanet Detector")

# Custom CSS to mimic the sleek dashboard look in the mockup
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #ffffff;
    }
    .metric-card {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        color: #00ff88;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #aaaaaa;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI-enabled Detection of Exoplanets")
st.markdown("Upload a light curve (CSV, TXT, or FITS) to analyze it for exoplanet transit signals.")

# --- Session State & Model Loading ---
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource(show_spinner="Loading AI model (real data model if available)...")
def get_trained_model():
    model = ExoplanetPipeline()
    if os.path.exists("real_model.pth"):
        model.load_state_dict(torch.load("real_model.pth", weights_only=True))
        model.eval()
        return model
    else:
        # Fallback to quick synthetic
        X, Y = generate_synthetic_data(num_samples=500, seq_len=256)
        model = train_pipeline(X, Y, epochs=15, batch_size=32)
        model.eval()
        return model

# Load the model implicitly so it's ready when user uploads data
model = get_trained_model()
st.session_state.model = model

# --- Sidebar for Upload ---
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Light Curve (CSV/FITS)", type=['csv', 'txt', 'fits'])

if st.sidebar.button("Load Sample Data"):
    demo_X, demo_Y = generate_synthetic_data(num_samples=1, seq_len=256)
    st.session_state.demo_flux = demo_X[0, 0, :].numpy()
    st.session_state.demo_time = np.linspace(0, 40, 256)
    st.session_state.using_demo = True
else:
    if 'using_demo' not in st.session_state:
        st.session_state.using_demo = False

# --- Main Dashboard ---
if uploaded_file is not None or st.session_state.using_demo:
    try:
        if uploaded_file is not None:
            time_raw, flux_raw = parse_uploaded_file(uploaded_file)
            st.session_state.using_demo = False
        else:
            time_raw = st.session_state.demo_time
            flux_raw = st.session_state.demo_flux

        # Preprocess using sliding windows to preserve high-resolution real transits
        seq_len = 256
        X_tensor, time_windows = sliding_window_preprocess(time_raw, flux_raw, window_size=seq_len, step_size=128)
        
        # Inference
        with torch.no_grad():
            decoded, prob = st.session_state.model(X_tensor)
            
        # Find the window with the highest transit probability
        max_prob_idx = torch.argmax(prob).item()
        transit_prob = prob[max_prob_idx].item()
        
        best_decoded = decoded[max_prob_idx].squeeze().numpy()
        best_flux_raw = X_tensor[max_prob_idx].squeeze().numpy()
        best_time_window = time_windows[max_prob_idx]
        
        # Probability over time
        window_centers = [np.mean(tw) for tw in time_windows]
        prob_series = prob.squeeze().numpy()
        if prob_series.ndim == 0:
            prob_series = [prob_series.item()]
            
        # Determine Detection
        threshold = 0.5
        is_detected = transit_prob > threshold
        
        # --- Visualization ---
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("1. Noisy Light Curve")
            fig1, ax1 = plt.subplots(figsize=(10, 3), facecolor='#0e1117')
            ax1.set_facecolor('#0e1117')
            ax1.plot(time_raw, flux_raw, color='white', linewidth=1)
            ax1.set_xlabel("Time", color='gray')
            ax1.set_ylabel("Relative Flux", color='gray')
            ax1.tick_params(colors='gray')
            ax1.grid(color='#333333', linestyle='--', linewidth=0.5)
            st.pyplot(fig1)

            st.subheader("3. Model Denoised Light Curve (Highest Prob Window)")
            fig2, ax2 = plt.subplots(figsize=(10, 3), facecolor='#0e1117')
            ax2.set_facecolor('#0e1117')
            ax2.plot(best_time_window, best_flux_raw, color='#555555', label="Original (Norm)", alpha=0.5)
            ax2.plot(best_time_window, best_decoded, color='#00ffff', linewidth=2, label="Denoised")
            ax2.set_xlabel("Time", color='gray')
            ax2.set_ylabel("Relative Flux", color='gray')
            ax2.tick_params(colors='gray')
            ax2.legend(loc='upper right', facecolor='#0e1117', edgecolor='none', labelcolor='white')
            ax2.grid(color='#333333', linestyle='--', linewidth=0.5)
            st.pyplot(fig2)
            
            st.subheader("4. Transit Probability Over Time")
            fig3, ax3 = plt.subplots(figsize=(10, 3), facecolor='#0e1117')
            ax3.set_facecolor('#0e1117')
            ax3.plot(window_centers, prob_series, color='#ff00ff', linewidth=2, marker='o', markersize=4)
            ax3.axhline(threshold, color='red', linestyle='--', label="Detection Threshold")
            ax3.set_xlabel("Time", color='gray')
            ax3.set_ylabel("Transit Probability", color='gray')
            ax3.tick_params(colors='gray')
            ax3.legend(loc='upper right', facecolor='#0e1117', edgecolor='none', labelcolor='white')
            ax3.grid(color='#333333', linestyle='--', linewidth=0.5)
            ax3.set_ylim(-0.1, 1.1)
            st.pyplot(fig3)

        with col2:
            st.subheader("5. Detection Result")
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Output Detection Probability</div>', unsafe_allow_html=True)
            
            # Change color based on detection
            color = "#00ff88" if is_detected else "#ff4444"
            st.markdown(f'<div class="metric-value" style="color: {color};">{transit_prob:.2f}</div>', unsafe_allow_html=True)
            
            if is_detected:
                st.markdown(f'<div style="color: {color}; font-size: 1.5rem; font-weight: bold; margin-top: 10px;">Transit Detected</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color: {color}; font-size: 1.5rem; font-weight: bold; margin-top: 10px;">No Transit</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### AI Pipeline Architecture")
            st.markdown("""
            **1D Convolutional Neural Network**
            * **Input**: Noisy Flux Sequence (1x256)
            * **Encoder**: Conv1D layers extract features & compress.
            * **Decoder**: ConvTranspose1D layers reconstruct clean signal.
            * **Classifier**: Fully Connected layers predict probability.
            """)
            
            
    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Upload a data file or load sample data from the sidebar to begin.")
