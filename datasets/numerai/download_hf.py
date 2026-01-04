"""Download Numerai dataset from HuggingFace mirror"""
from huggingface_hub import hf_hub_download
import os

os.chdir('/Users/joe/Documents/Research/datasets/numerai')

print("Downloading Numerai v5.2 from HuggingFace...")
print("This should be faster than the official API...")

# Download train parquet from HuggingFace
# The Numerati/numerai-datasets repo structure
try:
    file_path = hf_hub_download(
        repo_id="Numerati/numerai-datasets",
        filename="v5.2/train.parquet",
        repo_type="dataset",
        local_dir=".",
        local_dir_use_symlinks=False
    )
    print(f"\nDownload complete!")
    print(f"File saved to: {file_path}")
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative download method...")
    # Try downloading latest if v5.2 doesn't exist
    file_path = hf_hub_download(
        repo_id="Numerati/numerai-datasets",
        filename="train.parquet",
        repo_type="dataset",
        local_dir=".",
        local_dir_use_symlinks=False
    )
    print(f"Downloaded latest version to: {file_path}")
