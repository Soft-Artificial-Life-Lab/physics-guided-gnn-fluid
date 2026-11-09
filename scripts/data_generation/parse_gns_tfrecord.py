"""
Parse DeepMind GNS TFRecord files and convert to .npz format
"""

import tensorflow as tf
import numpy as np
import json
from pathlib import Path

def parse_tfrecord(tfrecord_path, metadata_path, output_dir, max_trajectories=5):
    """
    Parse GNS TFRecord file and convert to .npz format

    Args:
        tfrecord_path: Path to .tfrecord file
        metadata_path: Path to metadata.json
        output_dir: Directory to save .npz files
        max_trajectories: Maximum number of trajectories to process
    """

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"=" * 70)
    print(f"PARSING GNS TFRECORD")
    print(f"=" * 70)
    print(f"TFRecord: {tfrecord_path}")
    print(f"Metadata: {metadata_path}")
    print()
    print("Metadata:")
    print(json.dumps(metadata, indent=2))
    print()

    # Check dimensions
    dim = metadata.get('dim', 2)  # 2D or 3D
    print(f"Dimensions: {dim}D")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define feature description for parsing
    feature_description = {
        'position': tf.io.VarLenFeature(tf.string),
        'particle_type': tf.io.VarLenFeature(tf.string)
    }

    # Parse TFRecord
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))

    trajectory_count = 0

    for raw_record in dataset:
        if trajectory_count >= max_trajectories:
            break

        # Parse record
        example = tf.io.parse_single_example(raw_record, feature_description)

        # Extract position data
        position_data = example['position']
        position_values = tf.sparse.to_dense(position_data)

        # Decode position (stored as serialized tensors)
        positions = []
        for pos_bytes in position_values.numpy():
            pos_tensor = tf.io.parse_tensor(pos_bytes, out_type=tf.float32)
            positions.append(pos_tensor.numpy())

        # Extract particle types
        particle_type_data = example['particle_type']
        particle_type_values = tf.sparse.to_dense(particle_type_data)

        particle_types = []
        for pt_bytes in particle_type_values.numpy():
            pt_tensor = tf.io.parse_tensor(pt_bytes, out_type=tf.int32)
            particle_types.append(pt_tensor.numpy())

        # Save each frame
        trajectory_dir = output_path / f"trajectory_{trajectory_count:03d}"
        trajectory_dir.mkdir(exist_ok=True)

        num_frames = len(positions)
        print(f"Trajectory {trajectory_count}: {num_frames} frames, {len(positions[0])} particles, {dim}D")

        for frame_idx in range(num_frames):
            pos_array = np.array(positions[frame_idx])
            pt_array = np.array(particle_types[frame_idx])

            # Create velocities (compute from position differences)
            if frame_idx > 0:
                vel_array = (pos_array - np.array(positions[frame_idx-1])) / metadata.get('dt', 0.01)
            else:
                vel_array = np.zeros_like(pos_array)

            # Save to .npz
            frame_file = trajectory_dir / f"frame_{frame_idx:04d}.npz"
            np.savez(
                frame_file,
                positions=pos_array,
                velocities=vel_array,
                particle_types=pt_array,
                n_particles=len(pos_array),
                time=frame_idx * metadata.get('dt', 0.01)
            )

        trajectory_count += 1

    print()
    print(f"=" * 70)
    print(f"PARSING COMPLETE")
    print(f"  Trajectories processed: {trajectory_count}")
    print(f"  Output: {output_path}")
    print(f"=" * 70)

if __name__ == "__main__":
    # Parse Water-3D dataset
    dataset_dir = Path("datasets/gns_datasets/Water-3D")
    tfrecord_path = dataset_dir / "train.tfrecord"
    metadata_path = dataset_dir / "metadata.json"
    output_dir = Path("datasets/gns_water3d_parsed")

    if tfrecord_path.exists() and metadata_path.exists():
        parse_tfrecord(tfrecord_path, metadata_path, output_dir, max_trajectories=5)
    else:
        print(f"ERROR: Dataset files not found")
        print(f"  TFRecord: {tfrecord_path.exists()}")
        print(f"  Metadata: {metadata_path.exists()}")
