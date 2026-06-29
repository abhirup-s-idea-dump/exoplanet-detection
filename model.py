import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class DenoisingAutoencoder(nn.Module):
    def __init__(self, seq_len=256):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class TransitClassifier(nn.Module):
    def __init__(self, latent_dim=64 * 32): # For seq_len=256
        super(TransitClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten
        prob = self.classifier(x)
        return prob

class ExoplanetPipeline(nn.Module):
    def __init__(self):
        super(ExoplanetPipeline, self).__init__()
        self.autoencoder = DenoisingAutoencoder()
        self.classifier = TransitClassifier()
        
    def forward(self, x):
        decoded, encoded = self.autoencoder(x)
        prob = self.classifier(encoded)
        return decoded, prob

def train_pipeline(X, Y, epochs=10, batch_size=32):
    """
    Trains the Autoencoder and Classifier on provided synthetic/real data.
    X: tensor of shape (N, 1, seq_len)
    Y: tensor of shape (N, 1) labels (1=transit, 0=no transit)
    """
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ExoplanetPipeline()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss functions
    ae_criterion = nn.MSELoss()
    clf_criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            decoded, prob = model(batch_x)
            
            # Loss = Autoencoder reconstruction loss + Classifier classification loss
            # For reconstruction, we want the target to be a "clean" signal, but here
            # for simplicity we use self-reconstruction for denoising and rely on CNN bottleneck.
            # (In a real scenario, we'd train on clean signals without noise as targets)
            # Let's smooth the input batch_x to create a pseudo-clean target
            smoothed_x = torch.nn.functional.avg_pool1d(batch_x, kernel_size=5, stride=1, padding=2)
            
            loss_ae = ae_criterion(decoded, smoothed_x)
            loss_clf = clf_criterion(prob, batch_y)
            loss = loss_ae + 2.0 * loss_clf # Weight classifier loss higher
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    return model
