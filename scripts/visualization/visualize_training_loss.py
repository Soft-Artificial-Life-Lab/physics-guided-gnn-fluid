"""
Visualize Training Loss from Model Checkpoint or Log Files
Creates comprehensive loss plots and GIFs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import torch
import json
import re


def load_loss_from_checkpoint(checkpoint_path):
    """Load training and validation loss from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    train_losses = []
    val_losses = []

    if 'train_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
    if 'val_losses' in checkpoint:
        val_losses = checkpoint['val_losses']

    # If not in checkpoint, try to extract from history
    if 'history' in checkpoint:
        history = checkpoint['history']
        if 'train_loss' in history:
            train_losses = history['train_loss']
        if 'val_loss' in history:
            val_losses = history['val_loss']

    return train_losses, val_losses


def load_loss_from_log(log_path):
    """Load loss from log file"""
    train_losses = []
    val_losses = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Match patterns like: "Epoch 1/50, Train Loss: 0.1234"
            train_match = re.search(r'Train Loss:\s*([\d.]+)', line)
            if train_match:
                train_losses.append(float(train_match.group(1)))

            # Match validation loss
            val_match = re.search(r'Val Loss:\s*([\d.]+)', line)
            if val_match:
                val_losses.append(float(val_match.group(1)))

    return train_losses, val_losses


def create_loss_plot(train_losses, val_losses, output_path, title="Training Loss"):
    """Create static loss plot"""
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if len(val_losses) > 0:
        val_epochs = np.linspace(1, len(train_losses), len(val_losses))
        ax.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Log scale if loss varies greatly
    if len(train_losses) > 0:
        max_loss = max(train_losses)
        min_loss = min(train_losses)
        if max_loss / min_loss > 10:
            ax.set_yscale('log')
            ax.set_ylabel('Loss (log scale)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Loss plot: {output_path}")
    plt.close()


def create_loss_gif(train_losses, val_losses, output_path, title="Training Progress"):
    """Create animated GIF showing loss progression"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    def update(frame):
        ax1.clear()
        ax2.clear()

        # Left plot: Loss curves up to current epoch
        current_epoch = frame + 1
        epochs = range(1, current_epoch + 1)

        ax1.plot(epochs, train_losses[:current_epoch], 'b-',
                label='Training Loss', linewidth=2)

        if len(val_losses) > 0:
            val_epoch_end = min(len(val_losses), current_epoch)
            val_epochs = np.linspace(1, current_epoch, val_epoch_end)
            ax1.plot(val_epochs, val_losses[:val_epoch_end], 'r-',
                    label='Validation Loss', linewidth=2)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{title} - Epoch {current_epoch}/{len(train_losses)}',
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, len(train_losses))

        # Set y limit based on full data
        y_max = max(train_losses[:10]) if len(train_losses) >= 10 else max(train_losses)
        y_min = min(train_losses)
        ax1.set_ylim(y_min * 0.9, y_max * 1.1)

        # Right plot: Loss statistics
        ax2.text(0.1, 0.9, f'Current Epoch: {current_epoch}', fontsize=14,
                transform=ax2.transAxes, fontweight='bold')

        current_train_loss = train_losses[current_epoch - 1]
        ax2.text(0.1, 0.8, f'Training Loss: {current_train_loss:.6f}',
                fontsize=12, transform=ax2.transAxes)

        if len(val_losses) > 0 and current_epoch <= len(val_losses):
            current_val_loss = val_losses[current_epoch - 1]
            ax2.text(0.1, 0.7, f'Validation Loss: {current_val_loss:.6f}',
                    fontsize=12, transform=ax2.transAxes, color='red')

        # Best loss so far
        best_train_loss = min(train_losses[:current_epoch])
        best_train_epoch = train_losses[:current_epoch].index(best_train_loss) + 1
        ax2.text(0.1, 0.6, f'Best Train Loss: {best_train_loss:.6f} (Epoch {best_train_epoch})',
                fontsize=11, transform=ax2.transAxes, color='blue')

        if len(val_losses) > 0 and current_epoch <= len(val_losses):
            best_val_loss = min(val_losses[:current_epoch])
            best_val_epoch = val_losses[:current_epoch].index(best_val_loss) + 1
            ax2.text(0.1, 0.5, f'Best Val Loss: {best_val_loss:.6f} (Epoch {best_val_epoch})',
                    fontsize=11, transform=ax2.transAxes, color='red')

        # Improvement
        if current_epoch > 1:
            prev_train_loss = train_losses[current_epoch - 2]
            improvement = (prev_train_loss - current_train_loss) / prev_train_loss * 100
            color = 'green' if improvement > 0 else 'orange'
            ax2.text(0.1, 0.4, f'Improvement: {improvement:+.2f}%',
                    fontsize=11, transform=ax2.transAxes, color=color)

        # Loss reduction
        initial_loss = train_losses[0]
        reduction = (initial_loss - current_train_loss) / initial_loss * 100
        ax2.text(0.1, 0.3, f'Total Reduction: {reduction:.1f}%',
                fontsize=11, transform=ax2.transAxes, color='darkgreen')

        ax2.axis('off')

        plt.tight_layout()

    # Create animation - sample every N epochs if too many
    max_frames = 100
    if len(train_losses) > max_frames:
        frame_indices = np.linspace(0, len(train_losses) - 1, max_frames, dtype=int)
        frames = len(frame_indices)
        update_func = lambda f: update(frame_indices[f])
    else:
        frames = len(train_losses)
        update_func = update

    anim = FuncAnimation(fig, update_func, frames=frames, interval=100)

    # Save as GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer)

    print(f"[SAVED] Loss animation GIF: {output_path}")
    print(f"  Frames: {frames}")
    plt.close()


def create_multi_model_comparison(model_paths, labels, output_path):
    """Compare training losses from multiple models"""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, (model_path, label) in enumerate(zip(model_paths, labels)):
        train_losses, val_losses = load_loss_from_checkpoint(model_path)

        if len(train_losses) > 0:
            epochs = range(1, len(train_losses) + 1)
            color = colors[i % len(colors)]
            ax.plot(epochs, train_losses, color=color, linestyle='-',
                   label=f'{label} (Train)', linewidth=2, alpha=0.8)

            if len(val_losses) > 0:
                val_epochs = np.linspace(1, len(train_losses), len(val_losses))
                ax.plot(val_epochs, val_losses, color=color, linestyle='--',
                       label=f'{label} (Val)', linewidth=2, alpha=0.6)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Model Comparison: Training Loss', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Model comparison plot: {output_path}")
    plt.close()


def visualize_loss(source_path, output_dir="visualizations", create_gif=True):
    """
    Main function to visualize loss from checkpoint or log

    Args:
        source_path: Path to checkpoint (.pt) or log file (.log)
        output_dir: Directory to save outputs
        create_gif: Whether to create animated GIF
    """
    print("="*70)
    print("LOSS VISUALIZATION")
    print("="*70)

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load loss data
    print(f"\nLoading loss data from {source_path}...")

    if source_path.suffix == '.pt':
        train_losses, val_losses = load_loss_from_checkpoint(source_path)
        model_name = source_path.stem
    elif source_path.suffix == '.log':
        train_losses, val_losses = load_loss_from_log(source_path)
        model_name = source_path.stem
    else:
        raise ValueError(f"Unsupported file type: {source_path.suffix}")

    print(f"  Training epochs: {len(train_losses)}")
    print(f"  Validation epochs: {len(val_losses)}")

    if len(train_losses) == 0:
        print("  No training loss data found!")
        return

    # Print statistics
    print(f"\nTraining Loss Statistics:")
    print(f"  Initial: {train_losses[0]:.6f}")
    print(f"  Final: {train_losses[-1]:.6f}")
    print(f"  Best: {min(train_losses):.6f} (Epoch {train_losses.index(min(train_losses)) + 1})")
    print(f"  Reduction: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%")

    if len(val_losses) > 0:
        print(f"\nValidation Loss Statistics:")
        print(f"  Best: {min(val_losses):.6f} (Epoch {val_losses.index(min(val_losses)) + 1})")

    # Create static plot
    plot_path = output_dir / f"{model_name}_loss_plot.png"
    create_loss_plot(train_losses, val_losses, plot_path,
                    title=f"Training Loss - {model_name}")

    # Create animated GIF
    if create_gif:
        gif_path = output_dir / f"{model_name}_loss_animation.gif"
        create_loss_gif(train_losses, val_losses, gif_path,
                       title=f"Training Progress - {model_name}")

    print("\n" + "="*70)
    print("LOSS VISUALIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize training loss')
    parser.add_argument('source', type=str, help='Path to checkpoint (.pt) or log file (.log)')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory')
    parser.add_argument('--no-gif', action='store_true', help='Skip GIF creation')
    parser.add_argument('--compare', nargs='+', help='Additional models to compare')
    parser.add_argument('--labels', nargs='+', help='Labels for comparison models')

    args = parser.parse_args()

    if args.compare:
        if not args.labels or len(args.labels) != len(args.compare) + 1:
            print("Error: --labels must match number of models (including first model)")
            exit(1)

        output_path = Path(args.output_dir) / "model_comparison.png"
        create_multi_model_comparison(
            [args.source] + args.compare,
            args.labels,
            output_path
        )
    else:
        visualize_loss(args.source, args.output_dir, create_gif=not args.no_gif)
