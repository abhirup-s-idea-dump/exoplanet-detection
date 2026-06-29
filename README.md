# ExoDetect: AI Exoplanet Transit Detection

Built this pipeline for a hackathon to automatically detect exoplanet transits from noisy astronomical light curves. Instead of relying on traditional curve-fitting, it uses a 1D Convolutional Denoising Autoencoder in PyTorch to clean the signal and extract features, followed by a classifier to flag transits.

The cool part is that it actually works on real NASA data from the MAST archive (Kepler/TESS).

## Features
- **Real Data Fetching:** Includes a script to pull raw light curves directly from NASA's MAST archive via `lightkurve`.
- **Sliding Window Preprocessing:** Doesn't just squash the timeline. It chops the light curve into high-res overlapping windows so the microscopic transits don't get lost in the noise.
- **PyTorch Pipeline:** 1D CNN Autoencoder + Fully Connected Classifier.
- **Streamlit Dashboard:** A sleek, dark-mode UI to visualize the raw flux, the denoised model output, and the temporal probability.

## Setup

Make sure you have Python installed, then just grab the requirements:

```bash
pip install -r requirements.txt
```

## How to run the pipeline

1. **(Optional) Fetch some real data & train the model:**
   If you want to train it from scratch on some real Kepler data (like Kepler-10), run the fetch script and then the training script:
   ```bash
   python fetch_training_data.py
   python train_real_data.py
   ```
   *Note: This saves the weights to `real_model.pth`.*

2. **Launch the Dashboard:**
   To open the UI, just run:
   ```bash
   streamlit run app.py
   ```

## Tech Stack
- Python, PyTorch, Streamlit, Lightkurve, Pandas, Matplotlib

## Screenshots
*(Add a screenshot of your dashboard here!)*
