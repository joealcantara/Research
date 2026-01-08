#!/usr/bin/env python3
"""Download Numerai v5.2 dataset"""

from numerapi import NumerAPI
import os

# Create directory
os.makedirs("data/numerai", exist_ok=True)
os.chdir("data/numerai")

print("Downloading Numerai v5.2 dataset...")
print("This may take a few minutes...")

api = NumerAPI()

# Download training data (main dataset we need)
print("\n1. Downloading training data...")
api.download_dataset("v5.2/train.parquet", "train.parquet")
print("✓ Training data downloaded")

# Download features metadata
print("\n2. Downloading features metadata...")
api.download_dataset("v5.2/features.json", "features.json")
print("✓ Features metadata downloaded")

print("\n✓ Download complete!")
print(f"Files saved to: {os.getcwd()}")
