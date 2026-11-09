"""
Compare Old Model (88 samples) vs New Model (14,558 samples)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from datetime import datetime
import logging
import json

from src.models.gnn import MeshGraphNet
from scipy.spatial import cKDTree

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*60)
print("Model Comparison: Old vs New")
print("="*60)


def load_model(model_path):
    """Load trained model"""
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)

    # Get model config - handle both old and new format
    if 'config' in checkpoint:
        config = checkpoint['config']
        if 'model' in config:
            # New format
            model = MeshGraphNet(
                input_node_dim=config['model']['input_node_dim'],
                input_edge_dim=config['model']['input_edge_dim'],
                hidden_dim=config['model']['hidden_dim'],
                output_dim=config['model']['output_dim'],
                num_layers=config['model']['num_layers']
            )
        else:
            # Old format - config has the values directly
            model = MeshGraphNet(
                input_node_dim=config.get('input_node_dim', 6),
                input_edge_dim=config.get('input_edge_dim', 3),
                hidden_dim=config.get('hidden_dim', 64),
                output_dim=config.get('output_dim', 3),
                num_layers=config.get('num_layers', 5)
            )
    else:
        # Default config
        model = MeshGraphNet(
            input_node_dim=6,
            input_edge_dim=3,
            hidden_dim=64,
            output_dim=3,
            num_layers=5
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    val_loss = checkpoint.get('val_loss', 'N/A')
    logger.info(f"Model loaded - Val Loss: {val_loss}")

    return model, checkpoint


def evaluate_model(model, val_dataset, device='cuda', max_samples=50):
    """Evaluate model on validation set"""
    model.to(device)
    model.eval()

    mse_list = []
    mae_list = []
    rel_error_list = []

    energy_errors = []
    momentum_errors = []

    with torch.no_grad():
        for i in range(min(len(val_dataset), max_samples)):
            data = val_dataset[i]

            node_features = data['node_features'].to(device)
            edge_index = data['edge_index'].to(device)
            edge_attr = data['edge_attr'].to(device)
            target = data['target'].to(device)
            pos_t = data['positions_t'].to(device)
            pos_t1 = data['positions_t1'].to(device)

            # Predict
            pred_vel = model(node_features, edge_index, edge_attr)

            # Metrics
            mse = torch.mean((pred_vel - target) ** 2).item()
            mae = torch.mean(torch.abs(pred_vel - target)).item()
            rel_error = torch.mean(torch.abs(pred_vel - target) / (torch.abs(target) + 1e-6)).item()

            mse_list.append(mse)
            mae_list.append(mae)
            rel_error_list.append(rel_error)

            # Energy conservation
            pred_energy = torch.sum(pred_vel ** 2).item()
            true_energy = torch.sum(target ** 2).item()
            energy_error = abs(pred_energy - true_energy) / (true_energy + 1e-6)
            energy_errors.append(energy_error)

            # Momentum conservation
            pred_momentum = torch.sum(pred_vel, dim=0)
            true_momentum = torch.sum(target, dim=0)
            momentum_error = torch.mean((pred_momentum - true_momentum) ** 2).item()
            momentum_errors.append(momentum_error)

    return {
        'mse_mean': np.mean(mse_list),
        'mse_std': np.std(mse_list),
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'rel_error_mean': np.mean(rel_error_list),
        'rel_error_std': np.std(rel_error_list),
        'energy_error_mean': np.mean(energy_errors),
        'energy_error_std': np.std(energy_errors),
        'momentum_error_mean': np.mean(momentum_errors),
        'momentum_error_std': np.std(momentum_errors)
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load both models
    print("\n[1/4] Loading Old Model (88 samples)...")
    old_model, old_checkpoint = load_model("models/gnn_sph_best.pt")
    old_train_loss = old_checkpoint.get('train_loss', 'N/A')
    old_val_loss = old_checkpoint.get('val_loss', 'N/A')

    print("\n[2/4] Loading New Model (14,558 samples, 1 epoch)...")
    new_model, new_checkpoint = load_model("models/gnn_improved_sph_best.pt")
    new_train_loss = new_checkpoint.get('train_loss', 'N/A')
    new_val_loss = new_checkpoint.get('val_loss', 'N/A')

    # Load test dataset (using a different scenario for fair comparison)
    print("\n[3/4] Loading test dataset...")
    test_dataset = FLIPDataset("datasets/flip_simulations/sph_splash", sequence_length=2, radius=0.3)
    logger.info(f"Test dataset size: {len(test_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Evaluate both models
    print("\n[4/4] Evaluating both models...")
    print("\nEvaluating Old Model...")
    old_metrics = evaluate_model(old_model, test_dataset, device=device, max_samples=39)

    print("Evaluating New Model...")
    new_metrics = evaluate_model(new_model, test_dataset, device=device, max_samples=39)

    # Print comparison report
    print("\n" + "="*60)
    print("MODEL COMPARISON REPORT")
    print("="*60)

    print("\n--- TRAINING CONFIGURATION ---")
    print(f"Old Model:")
    print(f"  Training samples: 88")
    print(f"  Train Loss: {old_train_loss}")
    print(f"  Val Loss: {old_val_loss}")
    if isinstance(old_train_loss, (int, float)) and isinstance(old_val_loss, (int, float)):
        overfitting_ratio = old_val_loss / old_train_loss if old_train_loss > 0 else float('inf')
        print(f"  Train/Val Ratio: {overfitting_ratio:.1f}x")

    print(f"\nNew Model:")
    print(f"  Training samples: 14,558")
    print(f"  Epochs: 1")
    print(f"  Train Loss: {new_train_loss:.6f}")
    print(f"  Val Loss: {new_val_loss:.6f}")
    if isinstance(new_train_loss, (int, float)) and isinstance(new_val_loss, (int, float)):
        new_overfitting_ratio = new_val_loss / new_train_loss if new_train_loss > 0 else float('inf')
        print(f"  Train/Val Ratio: {new_overfitting_ratio:.1f}x")

    print("\n--- TEST SET PERFORMANCE ---")
    print(f"\nMean Squared Error (MSE):")
    print(f"  Old: {old_metrics['mse_mean']:.6f} +/- {old_metrics['mse_std']:.6f}")
    print(f"  New: {new_metrics['mse_mean']:.6f} +/- {new_metrics['mse_std']:.6f}")
    improvement = ((old_metrics['mse_mean'] - new_metrics['mse_mean']) / old_metrics['mse_mean']) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    print(f"\nMean Absolute Error (MAE):")
    print(f"  Old: {old_metrics['mae_mean']:.6f} +/- {old_metrics['mae_std']:.6f}")
    print(f"  New: {new_metrics['mae_mean']:.6f} +/- {new_metrics['mae_std']:.6f}")
    improvement = ((old_metrics['mae_mean'] - new_metrics['mae_mean']) / old_metrics['mae_mean']) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    print(f"\nRelative Error:")
    print(f"  Old: {old_metrics['rel_error_mean']:.4f} +/- {old_metrics['rel_error_std']:.4f}")
    print(f"  New: {new_metrics['rel_error_mean']:.4f} +/- {new_metrics['rel_error_std']:.4f}")
    improvement = ((old_metrics['rel_error_mean'] - new_metrics['rel_error_mean']) / old_metrics['rel_error_mean']) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    print("\n--- PHYSICS CONSERVATION ---")
    print(f"\nEnergy Conservation Error:")
    print(f"  Old: {old_metrics['energy_error_mean']:.6f} +/- {old_metrics['energy_error_std']:.6f}")
    print(f"  New: {new_metrics['energy_error_mean']:.6f} +/- {new_metrics['energy_error_std']:.6f}")
    improvement = ((old_metrics['energy_error_mean'] - new_metrics['energy_error_mean']) / old_metrics['energy_error_mean']) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    print(f"\nMomentum Conservation Error:")
    print(f"  Old: {old_metrics['momentum_error_mean']:.6f} +/- {old_metrics['momentum_error_std']:.6f}")
    print(f"  New: {new_metrics['momentum_error_mean']:.6f} +/- {new_metrics['momentum_error_std']:.6f}")
    improvement = ((old_metrics['momentum_error_mean'] - new_metrics['momentum_error_mean']) / old_metrics['momentum_error_mean']) * 100
    print(f"  Improvement: {improvement:+.1f}%")

    print("\n--- SUMMARY ---")
    print(f"Data scaling: 88 -> 14,558 samples (165x increase)")
    print(f"Overfitting reduction: {overfitting_ratio:.1f}x -> {new_overfitting_ratio:.1f}x")
    print(f"Average improvement across all metrics: {np.mean([
        ((old_metrics['mse_mean'] - new_metrics['mse_mean']) / old_metrics['mse_mean']) * 100,
        ((old_metrics['mae_mean'] - new_metrics['mae_mean']) / old_metrics['mae_mean']) * 100,
        ((old_metrics['rel_error_mean'] - new_metrics['rel_error_mean']) / old_metrics['rel_error_mean']) * 100
    ]):.1f}%")

    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)

    # Save results
    results = {
        'old_model': {
            'train_samples': 88,
            'train_loss': float(old_train_loss) if isinstance(old_train_loss, (int, float)) else str(old_train_loss),
            'val_loss': float(old_val_loss) if isinstance(old_val_loss, (int, float)) else str(old_val_loss),
            'metrics': {k: float(v) for k, v in old_metrics.items()}
        },
        'new_model': {
            'train_samples': 14558,
            'epochs': 1,
            'train_loss': float(new_train_loss),
            'val_loss': float(new_val_loss),
            'metrics': {k: float(v) for k, v in new_metrics.items()}
        }
    }

    results_path = Path("logs") / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    print(f"\nResults saved to: {results_path}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
