import numpy as np
import pandas as pd
import lightkurve as lk
import torch
import tempfile
import os

def generate_synthetic_data(num_samples=1000, seq_len=256):
    X = np.zeros((num_samples, 1, seq_len))
    Y = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Base flux
        base = np.ones(seq_len) * 0.9 + np.random.rand() * 0.1
        
        # Noise
        noise_level = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_level, seq_len)
        
        flux = base + noise
        
        # Add transit (50% chance)
        is_transit = np.random.rand() > 0.5
        if is_transit:
            # Random transit parameters
            transit_depth = np.random.uniform(0.05, 0.2)
            transit_duration = int(np.random.uniform(10, 40))
            transit_start = int(np.random.uniform(10, seq_len - transit_duration - 10))
            
            # Create U-shape dip
            dip = np.zeros(seq_len)
            dip[transit_start:transit_start+transit_duration] = -transit_depth
            
            # Smooth the dip slightly for realism (limb darkening)
            kernel = np.ones(5) / 5
            dip = np.convolve(dip, kernel, mode='same')
            
            flux += dip
            Y[i] = 1
            
        # Normalize to [0, 1]
        flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
        X[i, 0, :] = flux
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

def parse_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()
    
    # For Kepler/TESS FITS files
    if filename.endswith('.fits'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            lc = lk.read(tmp_path)
            lc = lc.remove_nans()
            time = lc.time.value
            flux = lc.flux.value
            os.remove(tmp_path)
            return time, flux
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise ValueError(f"Could not parse FITS file: {e}")
            
    # For CSV/TXT files
    else:
        try:
            # Attempt to read as CSV
            # Handle potential header comments
            df = pd.read_csv(uploaded_file, comment='#')
            
            # Guess columns
            time_col = None
            flux_col = None
            for col in df.columns:
                lower_col = col.lower().strip()
                if 'time' in lower_col or lower_col == 't' or 'bjd' in lower_col or 'jd' in lower_col:
                    if time_col is None:
                        time_col = col
                if 'flux' in lower_col or lower_col == 'f' or 'sap_flux' in lower_col:
                    if flux_col is None:
                        flux_col = col
            
            if time_col and flux_col:
                return df[time_col].values, df[flux_col].values
            elif len(df.columns) >= 2:
                # Fallback to first two columns
                return df.iloc[:, 0].values, df.iloc[:, 1].values
            else:
                raise ValueError("CSV must have at least two columns for Time and Flux.")
        except Exception as e:
            raise ValueError(f"Could not parse CSV/TXT file: {e}")

def preprocess_lightcurve(time, flux, seq_len=256):
    # Sort by time
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]
    
    # Remove NaNs
    valid = ~np.isnan(flux) & ~np.isnan(time)
    time = time[valid]
    flux = flux[valid]
    
    if len(time) == 0:
        raise ValueError("No valid data points found after removing NaNs.")
    
    # Interpolate to fixed sequence length uniformly
    time_uniform = np.linspace(time.min(), time.max(), seq_len)
    flux_uniform = np.interp(time_uniform, time, flux)
    
    # Normalize flux to [0, 1]
    flux_min = np.min(flux_uniform)
    flux_max = np.max(flux_uniform)
    
    if flux_max - flux_min > 0:
        flux_normalized = (flux_uniform - flux_min) / (flux_max - flux_min)
    else:
        flux_normalized = flux_uniform - flux_min
        
    return time_uniform, flux_normalized

def sliding_window_preprocess(time, flux, window_size=256, step_size=128):
    """
    Chops a long light curve into multiple sliding windows.
    This preserves the high resolution needed to detect real, tiny transits.
    Returns: X_tensor shape (num_windows, 1, window_size), time_windows list
    """
    # Sort and clean
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]
    valid = ~np.isnan(flux) & ~np.isnan(time)
    time = time[valid]
    flux = flux[valid]
    
    if len(time) < window_size:
        # If too short, just pad it or return one window
        time_uniform, flux_norm = preprocess_lightcurve(time, flux, window_size)
        return torch.tensor(flux_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0), [time_uniform]
        
    windows_flux = []
    windows_time = []
    
    # We step through the indices
    for i in range(0, len(time) - window_size + 1, step_size):
        t_window = time[i:i+window_size]
        f_window = flux[i:i+window_size]
        
        # Normalize this specific window
        f_min = np.min(f_window)
        f_max = np.max(f_window)
        if f_max - f_min > 0:
            f_norm = (f_window - f_min) / (f_max - f_min)
        else:
            f_norm = f_window - f_min
            
        windows_flux.append(f_norm)
        windows_time.append(t_window)
        
    X_tensor = torch.tensor(np.array(windows_flux), dtype=torch.float32).unsqueeze(1)
    return X_tensor, windows_time

