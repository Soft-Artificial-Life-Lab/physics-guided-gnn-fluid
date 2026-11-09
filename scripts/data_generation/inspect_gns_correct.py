"""
Inspect GNS TFRecord using correct SequenceExample parsing
Based on DeepMind's reading_utils.py
"""

import tensorflow as tf
import numpy as np
import json
from pathlib import Path

def convert_to_tensor(string, dtype):
    """Decode serialized numpy array."""
    return np.frombuffer(string, dtype=dtype)

def parse_sequence_example(example_proto, metadata):
    """Parse SequenceExample from GNS TFRecord."""

    # Context features (static across sequence)
    context_features = {
        'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'particle_type': tf.io.VarLenFeature(tf.string)
    }

    # Sequence features (vary over time)
    sequence_features = {
        'position': tf.io.VarLenFeature(tf.string),
    }

    # Parse
    context, sequences = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )

    return context, sequences

def inspect_tfrecord(tfrecord_path, metadata_path):
    """Inspect GNS TFRecord structure."""

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print("=" * 70)
    print("GNS TFRECORD INSPECTION (Correct Parsing)")
    print("=" * 70)
    print(f"TFRecord: {tfrecord_path}")
    print()
    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print()

    dim = metadata.get('dim', 3)
    dt = metadata.get('dt', 0.005)

    # Parse first trajectory
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))

    print("Inspecting first trajectory...")
    print()

    for idx, raw_record in enumerate(dataset.take(1)):
        context, sequences = parse_sequence_example(raw_record, metadata)

        # Parse particle types (context - static)
        particle_type_sparse = context['particle_type']
        particle_type_dense = tf.sparse.to_dense(particle_type_sparse)

        if len(particle_type_dense) > 0:
            # Decode particle types
            particle_type_bytes = particle_type_dense[0].numpy()
            particle_types = np.frombuffer(particle_type_bytes, dtype=np.int64)

            unique_types, counts = np.unique(particle_types, return_counts=True)

            print(f"Particle Types (static across trajectory):")
            for ptype, count in zip(unique_types, counts):
                print(f"  Type {ptype}: {count} particles ({count/len(particle_types)*100:.1f}%)")

            print()
            print("GNS Particle Type Codes:")
            print("  0 = Fluid")
            print("  3 = Wall/Boundary (static)")
            print("  6 = Dynamic Rigid Body")
            print()

            if 6 in unique_types:
                print("[OK] DYNAMIC RIGID BODIES FOUND!")
                print(f"     {counts[unique_types==6][0] if 6 in unique_types else 0} dynamic rigid body particles")
            elif 3 in unique_types:
                print("[WARNING] Only static boundaries found (type 3)")
                print("          NO dynamic rigid bodies (type 6)")
            else:
                print("[INFO] Only fluid particles (type 0)")

        # Parse positions (sequence - varies over time)
        position_sparse = sequences['position']
        position_dense = tf.sparse.to_dense(position_sparse)

        num_frames = len(position_dense)
        print()
        print(f"Sequence Information:")
        print(f"  Number of frames: {num_frames}")
        print(f"  Time step (dt): {dt}")
        print(f"  Total simulation time: {num_frames * dt:.3f}s")
        print()

        if len(position_dense) > 0:
            # Decode first and last frame positions
            for frame_idx in [0, num_frames-1]:
                pos_bytes = position_dense[frame_idx].numpy()
                positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, dim)

                print(f"Frame {frame_idx}:")
                print(f"  Particles: {positions.shape[0]}")
                print(f"  Dimensions: {positions.shape[1]}D")
                print(f"  X range: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
                print(f"  Y range: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
                if dim == 3:
                    print(f"  Z range: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

                # Check if particle types exist and show positions by type
                if len(particle_type_dense) > 0:
                    print(f"  Positions by particle type:")
                    for ptype in unique_types:
                        mask = particle_types == ptype
                        if mask.sum() > 0:
                            type_pos = positions[mask]
                            print(f"    Type {ptype}: {mask.sum()} particles, "
                                  f"Z-height: [{type_pos[:, 2].min():.3f}, {type_pos[:, 2].max():.3f}]"
                                  if dim == 3 else "")
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
