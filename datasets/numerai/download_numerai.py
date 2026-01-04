"""Download Numerai dataset v5.2"""
from numerapi import NumerAPI
import os

VERSION = "v5.2"

# Change to numerai directory
os.chdir('/Users/joe/Documents/Research/datasets/numerai')

print(f"Downloading Numerai {VERSION} dataset...")
napi = NumerAPI(verbosity="info")

# Download training data
print("Downloading train.parquet...")
napi.download_dataset(f"{VERSION}/train.parquet")

# Download validation data
print("Downloading validation.parquet...")
napi.download_dataset(f"{VERSION}/validation.parquet")

# Download features metadata
print("Downloading features.json...")
napi.download_dataset(f"{VERSION}/features.json")

print("\nDownload complete!")
print(f"Files saved to: /Users/joe/Documents/Research/datasets/numerai/")
