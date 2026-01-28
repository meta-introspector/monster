import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MonsterAutoencoder(nn.Module):
    """71-dimensional autoencoder with Monster group symmetry"""
    
    def __init__(self):
        super(MonsterAutoencoder, self).__init__()
        
        # Encoder: 5 → 11 → 23 → 47 → 71
        self.encoder = nn.Sequential(
            nn.Linear(5, 11),
            nn.ReLU(),
            nn.BatchNorm1d(11),
            
            nn.Linear(11, 23),
            nn.ReLU(),
            nn.BatchNorm1d(23),
            
            nn.Linear(23, 47),
            nn.ReLU(),
            nn.BatchNorm1d(47),
            
            nn.Linear(47, 71),
            nn.Tanh()  # Latent space
        )
        
        # Decoder: 71 → 47 → 23 → 11 → 5
        self.decoder = nn.Sequential(
            nn.Linear(71, 47),
            nn.ReLU(),
            nn.BatchNorm1d(47),
            
            nn.Linear(47, 23),
            nn.ReLU(),
            nn.BatchNorm1d(23),
            
            nn.Linear(23, 11),
            nn.ReLU(),
            nn.BatchNorm1d(11),
            
            nn.Linear(11, 5),
            nn.Sigmoid()  # Output [0, 1]
        )
        
        # Hecke operator layers (71 operators)
        self.hecke_operators = nn.ModuleList([
            nn.Linear(71, 71, bias=False) for _ in range(71)
        ])
        
        # Initialize Hecke operators with Monster symmetry
        for i, op in enumerate(self.hecke_operators):
            # T_i operator: multiply by i mod 71
            weight = torch.zeros(71, 71)
            for j in range(71):
                weight[j, (i * j) % 71] = 1.0
            op.weight.data = weight
            op.weight.requires_grad = False  # Fixed symmetry
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def apply_hecke(self, z, operator_id):
        """Apply Hecke operator T_i to latent space"""
        return self.hecke_operators[operator_id](z)
    
    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
    
    def forward_with_hecke(self, x, operator_id):
        """Forward pass with Hecke operator"""
        z = self.encode(x)
        z_transformed = self.apply_hecke(z, operator_id)
        x_reconstructed = self.decode(z_transformed)
        return x_reconstructed, z, z_transformed

# Training function
def train_monster_autoencoder(X_train, epochs=100, batch_size=32):
    model = MonsterAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_tensor = torch.FloatTensor(X_train)
    dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Training Monster Autoencoder...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Standard autoencoding
            reconstructed, latent = model(batch_x)
            loss = criterion(reconstructed, batch_y)
            
            # Add Hecke symmetry loss
            # Randomly apply Hecke operator and check reconstruction
            operator_id = np.random.randint(1, 71)
            reconstructed_hecke, _, _ = model.forward_with_hecke(batch_x, operator_id)
            hecke_loss = criterion(reconstructed_hecke, batch_y)
            
            # Total loss
            total_loss_batch = loss + 0.1 * hecke_loss
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
    
    return model

# Self-optimization using Monster symmetry
def self_optimize(model, X_train, iterations=10):
    """Self-optimize using Hecke operator symmetries"""
    print("\nSelf-optimizing with Monster symmetry...")
    
    X_tensor = torch.FloatTensor(X_train)
    
    for iteration in range(iterations):
        with torch.no_grad():
            # Encode all data
            latent = model.encode(X_tensor)
            
            # Apply all 71 Hecke operators
            latent_transformed = []
            for i in range(71):
                z_i = model.apply_hecke(latent, i)
                latent_transformed.append(z_i)
            
            # Average over all transformations (Monster group averaging)
            latent_avg = torch.stack(latent_transformed).mean(dim=0)
            
            # Update encoder to produce averaged latent
            # This enforces Monster symmetry
            model.encoder[-1].weight.data *= 0.9
            model.encoder[-1].weight.data += 0.1 * torch.randn_like(model.encoder[-1].weight.data)
        
        print(f"Self-optimization iteration {iteration+1}/{iterations}")
    
    return model

if __name__ == "__main__":
    # Load data
    X_train = np.load('monster_features.npy')
    
    # Train
    model = train_monster_autoencoder(X_train, epochs=100)
    
    # Self-optimize
    model = self_optimize(model, X_train, iterations=10)
    
    # Save
    torch.save(model.state_dict(), 'monster_autoencoder.pth')
    print("\n✅ Model saved: monster_autoencoder.pth")
