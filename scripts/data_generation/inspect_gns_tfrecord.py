"""
Inspect GNS TFRecord structure to check for rigid bodies
"""

import tensorflow as tf
import numpy as np
import json
from pathlib import Path

def inspect_tfrecord(tfrecord_path, metadata_path):
    """
    Inspect the first trajectory to understand data structure

    Args:
        tfrecord_path: Path to .tfrecord file
        metadata_path: Path to metadata.json
    """

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print("=" * 70)
    print("GNS TFRECORD INSPECTION")
    print("=" * 70)
    print(f"TFRecord: {tfrecord_path}")
    print(f"Metadata: {metadata_path}")
    print()
    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print()

    # Parse TFRecord
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))

    print("Inspecting first trajectory...")
    print()

    for idx, raw_record in enumerate(dataset.take(1)):
        # Parse to get keys
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        print(f"Available features in TFRecord:")
        for key in example.features.feature.keys():
            print(f"  - {key}")
        print()

        # Try to extract particle types
        if 'particle_type' in example.features.feature:
            particle_type_feature = example.features.feature['particle_type']
            print("Particle type feature found!")

            # Parse particle types
            pt_list = particle_type_feature.bytes_list.value
            if len(pt_list) > 0:
                # Decode first frame
                pt_tensor = tf.io.parse_tensor(pt_list[0], out_type=tf.int32)
                particle_types = pt_tensor.numpy().flatten()

                unique_types, counts = np.unique(particle_types, return_counts=True)

                print(f"\\nParticle Types in Frame 0:")
                for ptype, count in zip(unique_types, counts):
                    print(f"  Type {ptype}: {count} particles")

                # Check GNS particle type meanings
                print("\\nGNS Particle Type Codes:")
                print("  0 = Fluid")
                print("  3 = Wall/Boundary (static)")
                print("  6 = Dynamic Rigid Body (if exists)")
                print()

                if 6 in unique_types:
                    print("[OK] DYNAMIC RIGID BODIES FOUND!")
                elif 3 in unique_types:
                    print("[INFO] Static boundaries found, but NO dynamic rigid bodies")
                else:
                    print("[INFO] Only fluid particles detected")

        # Check positions
        if 'position' in example.features.feature:
            position_feature = example.features.feature['position']
            pos_list = position_feature.bytes_list.value

            if len(pos_list) > 0:
                # Decode first frame
                pos_tensor = tf.io.parse_tensor(pos_list[0], out_type=tf.float32)
                positions = pos_tensor.numpy()

                print(f"\\nPosition data:")
                print(f"  Shape: {positions.shape}")
                print(f"  Dimensions: {positions.shape[1]}D")
                print(f"  Particles: {positions.shape[0]}")
                print(f"  X range: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
                print(f"  Y range: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
                if positions.shape[1] == 3:
                    print(f"  Z range: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    dataset_dir = Path("datasets/gns_datasets/Water-3D")
    tfrecord_path = dataset_dir / "train.tfrecord"
    metadata_path = dataset_dir / "metadata.json"

    if tfrecord_path.exists() and metadata_path.exists():
        inspect_tfrecord(tfrecord_path, metadata_path)
    else:
        print("ERROR: Dataset files not found")
        print(f"  TFRecord: {tfrecord_path} - exists: {tfrecord_path.exists()}")
        print(f"  Metadata: {metadata_path} - exists: {metadata_path.exists()}")
