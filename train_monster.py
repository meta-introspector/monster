#!/usr/bin/env python3
"""Train the Monster Autoencoder"""

import numpy as np
from monster_autoencoder import train_monster_autoencoder, self_optimize
import torch

# Load data
X_train = np.load('monster_features.npy')

print(f"Training data: {X_train.shape}")
print()

# Train
print("Phase 1: Standard training")
model = train_monster_autoencoder(X_train, epochs=100, batch_size=32)

# Self-optimize
print("\nPhase 2: Self-optimization with Monster symmetry")
model = self_optimize(model, X_train, iterations=10)

# Save
torch.save(model.state_dict(), 'monster_autoencoder.pth')
print("\n✅ Training complete!")
print("Model saved: monster_autoencoder.pth")

# Test reconstruction
print("\nTesting reconstruction...")
X_tensor = torch.FloatTensor(X_train[:10])
with torch.no_grad():
    reconstructed, latent = model(X_tensor)
    mse = ((X_tensor - reconstructed) ** 2).mean()
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Latent space shape: {latent.shape}")

# Test Hecke operators
print("\nTesting Hecke operators...")
for i in [2, 3, 5, 7, 11, 71]:
    with torch.no_grad():
        reconstructed_hecke, _, z_transformed = model.forward_with_hecke(X_tensor, i % 71)
        mse_hecke = ((X_tensor - reconstructed_hecke) ** 2).mean()
        print(f"T_{i} reconstruction MSE: {mse_hecke:.6f}")
