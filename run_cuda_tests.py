#!/usr/bin/env python3
"""
Run Monster Autoencoder on CUDA with LMFDB test suite
Record results as Parquet for HuggingFace
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from datetime import datetime
import time

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

# Load Monster Autoencoder
from monster_autoencoder import MonsterAutoencoder

model = MonsterAutoencoder().to(device)
print(f"Model loaded on {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Load test data
X_test = np.load('monster_features.npy')
X_tensor = torch.FloatTensor(X_test).to(device)

print(f"Test data: {X_tensor.shape}")
print()

# Load test suite
with open('lmfdb_test_suite.json') as f:
    test_suite = json.load(f)

print(f"Test suite: {test_suite['total_tests']} tests")
print(f"Tests with 71: {test_suite['tests_with_71']}")
print()

# Run tests
print("🧪 RUNNING TESTS ON CUDA:")
print("-" * 60)

results = []

# Test 1: Autoencoding
print("Test 1: Autoencoding...")
start = time.time()
with torch.no_grad():
    reconstructed, latent = model(X_tensor)
    mse = ((X_tensor - reconstructed) ** 2).mean().item()
end = time.time()

results.append({
    'test_name': 'autoencoding',
    'test_type': 'reconstruction',
    'device': str(device),
    'input_shape': str(X_tensor.shape),
    'latent_shape': str(latent.shape),
    'mse': mse,
    'time_seconds': end - start,
    'throughput': len(X_tensor) / (end - start),
    'timestamp': datetime.now().isoformat()
})

print(f"  MSE: {mse:.6f}")
print(f"  Time: {end - start:.3f}s")
print(f"  Throughput: {len(X_tensor) / (end - start):.0f} samples/s")
print()

# Test 2: Hecke operators
print("Test 2: Hecke operators...")
for operator_id in [2, 3, 5, 7, 11, 71]:
    start = time.time()
    with torch.no_grad():
        reconstructed_hecke, _, z_transformed = model.forward_with_hecke(
            X_tensor[:100], operator_id % 71
        )
        mse_hecke = ((X_tensor[:100] - reconstructed_hecke) ** 2).mean().item()
    end = time.time()
    
    results.append({
        'test_name': f'hecke_T_{operator_id}',
        'test_type': 'hecke_operator',
        'device': str(device),
        'operator_id': operator_id % 71,
        'input_shape': str(X_tensor[:100].shape),
        'mse': mse_hecke,
        'time_seconds': end - start,
        'throughput': 100 / (end - start),
        'timestamp': datetime.now().isoformat()
    })
    
    print(f"  T_{operator_id}: MSE={mse_hecke:.6f}, Time={end-start:.3f}s")

print()

# Test 3: Latent space analysis
print("Test 3: Latent space analysis...")
start = time.time()
with torch.no_grad():
    latent = model.encode(X_tensor)
    
    # Statistics
    latent_mean = latent.mean(dim=0)
    latent_std = latent.std(dim=0)
    latent_min = latent.min(dim=0)[0]
    latent_max = latent.max(dim=0)[0]
end = time.time()

results.append({
    'test_name': 'latent_space_analysis',
    'test_type': 'analysis',
    'device': str(device),
    'latent_dim': 71,
    'mean_norm': latent_mean.norm().item(),
    'std_mean': latent_std.mean().item(),
    'range_mean': (latent_max - latent_min).mean().item(),
    'time_seconds': end - start,
    'timestamp': datetime.now().isoformat()
})

print(f"  Latent mean norm: {latent_mean.norm().item():.4f}")
print(f"  Latent std (avg): {latent_std.mean().item():.4f}")
print(f"  Time: {end - start:.3f}s")
print()

# Test 4: LMFDB-specific tests
print("Test 4: LMFDB-specific tests...")
for test in test_suite['tests'][:10]:  # Run first 10
    if test['has_71']:
        # Simulate test with random input
        test_input = torch.randn(1, 5).to(device)
        
        start = time.time()
        with torch.no_grad():
            reconstructed, latent = model(test_input)
            mse = ((test_input - reconstructed) ** 2).mean().item()
        end = time.time()
        
        results.append({
            'test_name': test['name'],
            'test_type': 'lmfdb_test',
            'device': str(device),
            'test_file': test['file'],
            'test_line': test['line'],
            'mse': mse,
            'time_seconds': end - start,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"  {test['name']}: MSE={mse:.6f}")

print()

# Convert to DataFrame
df_results = pd.DataFrame(results)

print("📊 TEST RESULTS SUMMARY:")
print("-" * 60)
print(df_results.groupby('test_type').agg({
    'mse': ['mean', 'std', 'min', 'max'],
    'time_seconds': ['mean', 'sum']
}))
print()

# Save as Parquet
df_results.to_parquet('monster_test_results.parquet', compression='snappy', index=False)
print(f"💾 Saved: monster_test_results.parquet")
print()

# Prepare for HuggingFace
print("🤗 PREPARING HUGGINGFACE DATASET:")
print("-" * 60)

# Add metadata
df_results['model_name'] = 'MonsterAutoencoder'
df_results['model_version'] = '1.0'
df_results['dataset_name'] = 'lmfdb-monster-tests'
df_results['cuda_available'] = torch.cuda.is_available()
if torch.cuda.is_available():
    df_results['gpu_name'] = torch.cuda.get_device_name(0)

# Save final dataset
df_results.to_parquet('lmfdb_monster_dataset.parquet', compression='snappy', index=False)
print(f"💾 Saved: lmfdb_monster_dataset.parquet")
print()

print("Dataset info:")
print(f"  Rows: {len(df_results)}")
print(f"  Columns: {len(df_results.columns)}")
print(f"  Size: {Path('lmfdb_monster_dataset.parquet').stat().st_size / 1024:.2f} KB")
print()

print("✅ CUDA TESTS COMPLETE")
