"""
Dense autoencoder for FSM anomaly detection.

Architecture:
  Encoder: input_dim -> 128 -> 64 -> bottleneck_dim  (ReLU)
  Decoder: bottleneck_dim -> 64 -> 128 -> input_dim   (ReLU + Sigmoid output)

Loss: Binary Cross-Entropy (BCE)
Optimizer: Adam
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class FSMAutoencoder(nn.Module):
    """Dense autoencoder for one-hot encoded FSM state windows."""

    def __init__(self, input_dim, bottleneck_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(model, train_data, val_data,
                      epochs=50, batch_size=64, lr=1e-3, device="cuda",
                      print_every=10):
    """Train the autoencoder on normal windows (input == target).

    Args:
        model: FSMAutoencoder instance
        train_data: np.ndarray of shape (n_train, input_dim)
        val_data: np.ndarray of shape (n_val, input_dim)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate for Adam
        device: 'cpu' or 'cuda'

    Returns:
        (train_losses, val_losses) — per-epoch mean BCE values
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_tensor = torch.from_numpy(train_data).to(device)
    val_tensor = torch.from_numpy(val_data).to(device)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        train_loss = epoch_loss / len(train_data)
        train_losses.append(train_loss)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor)
            val_loss = criterion(val_recon, val_tensor).item()
        val_losses.append(val_loss)

        if print_every and ((epoch + 1) % print_every == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:>3}/{epochs}"
                  f"  train_loss={train_loss:.6f}"
                  f"  val_loss={val_loss:.6f}")

    return train_losses, val_losses


def compute_reconstruction_errors(model, data, device="cuda"):
    """Compute per-window mean BCE reconstruction error.

    Args:
        model: trained FSMAutoencoder
        data: np.ndarray of shape (n_windows, input_dim)

    Returns:
        np.ndarray of shape (n_windows,) — scalar error per window
    """
    model.eval()
    errors_list = []
    tensor = torch.from_numpy(data).to(device)

    # Process in chunks to avoid memory issues on large datasets
    chunk_size = 4096
    with torch.no_grad():
        for i in range(0, len(tensor), chunk_size):
            chunk = tensor[i:i + chunk_size]
            recon = model(chunk)
            # Per-window mean BCE
            bce = nn.functional.binary_cross_entropy(
                recon, chunk, reduction="none"
            ).mean(dim=1)
            errors_list.append(bce.cpu().numpy())

    return np.concatenate(errors_list)


def select_threshold(val_errors, percentile=95):
    """Set anomaly threshold as a percentile of validation error distribution."""
    return float(np.percentile(val_errors, percentile))
