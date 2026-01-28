#!/usr/bin/env python3
"""
Upload LMFDB Monster dataset to HuggingFace
"""

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
