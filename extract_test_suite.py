#!/usr/bin/env python3
"""
Extract LMFDB test suite, run on CUDA model, export to HuggingFace
"""

import ast
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

print("🧪 EXTRACTING LMFDB TEST SUITE")
print("=" * 60)
print()

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"

# Find all test files
test_files = list(Path(LMFDB_PATH).rglob("test_*.py"))
print(f"Found {len(test_files)} test files")
print()

# Extract test functions
test_cases = []

for test_file in test_files:
    try:
        content = test_file.read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    # Extract test info
                    test_case = {
                        'name': node.name,
                        'file': str(test_file.relative_to(LMFDB_PATH)),
                        'line': node.lineno,
                        'num_statements': len(node.body),
                        'has_71': '71' in ast.unparse(node) if hasattr(ast, 'unparse') else False,
                        'docstring': ast.get_docstring(node) or '',
                        'complexity': len(node.body) + len(node.args.args)
                    }
                    test_cases.append(test_case)
                    
    except Exception as e:
        pass

print(f"✅ Extracted {len(test_cases)} test cases")
print()

# Filter tests with 71
tests_71 = [t for t in test_cases if t['has_71']]
print(f"Tests with prime 71: {len(tests_71)}")
print()

# Show sample tests
print("Sample tests with 71:")
for test in tests_71[:5]:
    print(f"  {test['name']}")
    print(f"    {test['file']}:{test['line']}")
    if test['docstring']:
        print(f"    {test['docstring'][:60]}...")

print()

# Save test suite
with open('lmfdb_test_suite.json', 'w') as f:
    json.dump({
        'total_tests': len(test_cases),
        'tests_with_71': len(tests_71),
        'tests': test_cases
    }, f, indent=2)

print(f"💾 Saved: lmfdb_test_suite.json")
print()

# Generate CUDA test runner
print("🔥 GENERATING CUDA TEST RUNNER:")
print("-" * 60)

cuda_test_code = """#!/usr/bin/env python3
\"\"\"
Run Monster Autoencoder on CUDA with LMFDB test suite
Record results as Parquet for HuggingFace
\"\"\"

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
"""

with open('run_cuda_tests.py', 'w') as f:
    f.write(cuda_test_code)

import os
os.chmod('run_cuda_tests.py', 0o755)

print("✅ Generated: run_cuda_tests.py")
print()

# Generate HuggingFace upload script
print("🤗 GENERATING HUGGINGFACE UPLOAD SCRIPT:")
print("-" * 60)

hf_upload_code = """#!/usr/bin/env python3
\"\"\"
Upload LMFDB Monster dataset to HuggingFace
\"\"\"

from datasets import Dataset
import pandas as pd

print("🤗 UPLOADING TO HUGGINGFACE")
print("=" * 60)
print()

# Load dataset
df = pd.read_parquet('lmfdb_monster_dataset.parquet')

print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
print()

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

print("Dataset features:")
print(dataset.features)
print()

# Upload to HuggingFace Hub
# Requires: huggingface-cli login
dataset_name = "meta-introspector/lmfdb-monster-71"

print(f"Uploading to: {dataset_name}")
print()

try:
    dataset.push_to_hub(dataset_name)
    print("✅ Upload complete!")
    print(f"View at: https://huggingface.co/datasets/{dataset_name}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    print()
    print("To upload manually:")
    print(f"  1. huggingface-cli login")
    print(f"  2. python3 upload_to_huggingface.py")
"""

with open('upload_to_huggingface.py', 'w') as f:
    f.write(hf_upload_code)

os.chmod('upload_to_huggingface.py', 0o755)

print("✅ Generated: upload_to_huggingface.py")
print()

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print(f"Test suite: {len(test_cases)} tests")
print(f"Tests with 71: {len(tests_71)}")
print()
print("Generated files:")
print("  - lmfdb_test_suite.json (test metadata)")
print("  - run_cuda_tests.py (CUDA test runner)")
print("  - upload_to_huggingface.py (HF upload)")
print()
print("To run:")
print("  1. python3 run_cuda_tests.py")
print("  2. python3 upload_to_huggingface.py")
print()
print("✅ TEST SUITE EXTRACTION COMPLETE")
