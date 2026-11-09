"""
Download MeshGraphNets datasets using Python
"""

import urllib.request
import os
from pathlib import Path

def download_dataset(dataset_name, output_dir):
    """
    Download MeshGraphNets dataset

    Args:
        dataset_name: e.g., 'cylinder_flow'
        output_dir: Directory to save files
    """

    base_url = f"https://storage.googleapis.com/dm-meshgraphnets/{dataset_name}/"
    files = ['meta.json', 'train.tfrecord', 'valid.tfrecord', 'test.tfrecord']

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name} dataset...")
    print(f"Output directory: {output_path}")
    print()

    for filename in files:
        url = base_url + filename
        output_file = output_path / filename

        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, output_file)
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            print(f"  [OK] {filename} ({file_size:.2f} MB)")
        except Exception as e:
            print(f"  [ERROR] Failed to download {filename}: {e}")

    print()
    print(f"Download complete: {output_path}")

if __name__ == "__main__":
    # Download cylinder_flow dataset
    download_dataset("cylinder_flow", "datasets/meshgraphnets/cylinder_flow")
