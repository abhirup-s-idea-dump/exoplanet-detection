import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from data_utils import sliding_window_preprocess
from model import ExoplanetPipeline

def load_and_preprocess_folder(folder_path, label, window_size=256, step_size=128):
    all_X = []
    all_Y = []
    
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    for file in csv_files:
        df = pd.read_csv(file)
        if 'Time' not in df.columns or 'Flux' not in df.columns:
            continue
            
        time = df['Time'].values
        flux = df['Flux'].values
        
        # Chop into windows
        X_tensor, _ = sliding_window_preprocess(time, flux, window_size, step_size)
        
        # In a real rigorous setup, you'd only label windows containing the actual transit as 1.
        # But for this prototype, we'll softly assume positive targets have some transits.
        # A more advanced model would use attention or weak supervision, but this is a start.
        Y_tensor = torch.ones(X_tensor.size(0), 1) * label
        
        all_X.append(X_tensor)
        all_Y.append(Y_tensor)
        
    if len(all_X) > 0:
        return torch.cat(all_X, dim=0), torch.cat(all_Y, dim=0)
    else:
        return torch.empty(0), torch.empty(0)

if __name__ == "__main__":
    print("Preparing training data...")
    base_dir = "training_data"
    pos_dir = os.path.join(base_dir, "positive")
    neg_dir = os.path.join(base_dir, "negative")
    
    X_pos, Y_pos = load_and_preprocess_folder(pos_dir, label=1.0)
    X_neg, Y_neg = load_and_preprocess_folder(neg_dir, label=0.0)
    
    if len(X_pos) == 0 and len(X_neg) == 0:
        print("No training data found. Please run fetch_training_data.py first.")
        exit()
        
    X_train = torch.cat([X_pos, X_neg], dim=0)
    Y_train = torch.cat([Y_pos, Y_neg], dim=0)
    
    print(f"Total training windows: {len(X_train)} ({len(X_pos)} positive, {len(X_neg)} negative)")
    
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = ExoplanetPipeline()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ae_criterion = nn.MSELoss()
    clf_criterion = nn.BCELoss()
    
    epochs = 10
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            decoded, prob = model(batch_x)
            
            smoothed_x = torch.nn.functional.avg_pool1d(batch_x, kernel_size=5, stride=1, padding=2)
            loss_ae = ae_criterion(decoded, smoothed_x)
            loss_clf = clf_criterion(prob, batch_y)
            loss = loss_ae + 2.0 * loss_clf
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
    # Save the trained model
    torch.save(model.state_dict(), "real_model.pth")
    print("Training complete! Saved weights to 'real_model.pth'.")
