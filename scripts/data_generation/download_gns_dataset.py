"""
Download DeepMind Learning-to-Simulate (GNS) datasets
"""

import urllib.request
import os
from pathlib import Path

def download_gns_dataset(dataset_name, output_dir):
    """
    Download GNS dataset from DeepMind

    Args:
        dataset_name: e.g., 'Water-3D', 'WaterRamps', 'FluidShake'
        output_dir: Base directory to save files
    """

    base_url = f"https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{dataset_name}/"
    files = ['metadata.json', 'train.tfrecord', 'valid.tfrecord', 'test.tfrecord']

    # Create output directory
    dataset_path = Path(output_dir) / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    print(f"=" * 70)
    print(f"DOWNLOADING GNS DATASET: {dataset_name}")
    print(f"=" * 70)
    print(f"Output directory: {dataset_path}")
    print()

    for filename in files:
        url = base_url + filename
        output_file = dataset_path / filename

        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, output_file)
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            print(f"  [OK] {filename} ({file_size:.2f} MB)")
        except Exception as e:
            print(f"  [ERROR] Failed to download {filename}: {e}")

    print()
    print(f"=" * 70)
    print(f"DOWNLOAD COMPLETE: {dataset_path}")
    print(f"=" * 70)

if __name__ == "__main__":
    import sys

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "WaterRamps"
    download_gns_dataset(dataset_name, "datasets/gns_datasets")
