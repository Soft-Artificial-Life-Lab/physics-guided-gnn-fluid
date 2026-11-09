"""
Comprehensive Model Evaluation

Evaluates the SPH-trained model on multiple metrics:
1. Quantitative metrics (MSE, MAE, etc.)
2. Physics conservation (energy, momentum)
3. Visual comparison
4. Error analysis
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from src.models.gnn import MeshGraphNet
from scipy.spatial import cKDTree

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
print("Model Evaluation - SPH Trained GNN")
print("="*60)
logger.info("="*60)
logger.info("Model Evaluation Started")
logger.info(f"Log file: {log_file}")
logger.info("="*60)


def load_model(model_path):
    """Load trained model"""
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)

    model = MeshGraphNet(
        input_node_dim=6,
        input_edge_dim=3,
        hidden_dim=64,
        output_dim=3,
        num_layers=5
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded - Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    return model, checkpoint


def evaluate_metrics(model, dataset, device='cpu', max_samples=None):
    """
    Compute quantitative metrics

    Returns:
        dict with MSE, MAE, relative error, etc.
    """
    logger.info("Computing quantitative metrics...")

    model.to(device)
    model.eval()

    mse_list = []
    mae_list = []
    rel_error_list = []

    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]

            node_features = sample['node_features'].unsqueeze(0).to(device)
            edge_index = sample['edge_index'].unsqueeze(0).to(device)
            edge_attr = sample['edge_attr'].unsqueeze(0).to(device)
            target = sample['target'].unsqueeze(0).to(device)

            # Prediction
            pred = model(node_features.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0))

            # Metrics
            mse = torch.mean((pred - target.squeeze(0)) ** 2).item()
            mae = torch.mean(torch.abs(pred - target.squeeze(0))).item()

            # Relative error
            target_norm = torch.norm(target.squeeze(0), dim=1).mean()
            error_norm = torch.norm(pred - target.squeeze(0), dim=1).mean()
            rel_error = (error_norm / (target_norm + 1e-6)).item()

            mse_list.append(mse)
            mae_list.append(mae)
            rel_error_list.append(rel_error)

            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {i+1}/{num_samples} samples")

    metrics = {
        'mse_mean': np.mean(mse_list),
        'mse_std': np.std(mse_list),
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'rel_error_mean': np.mean(rel_error_list),
        'rel_error_std': np.std(rel_error_list),
        'mse_list': mse_list,
        'mae_list': mae_list,
        'rel_error_list': rel_error_list
    }

    logger.info("Quantitative Metrics:")
    logger.info(f"  MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
    logger.info(f"  MAE: {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
    logger.info(f"  Relative Error: {metrics['rel_error_mean']:.4f} ± {metrics['rel_error_std']:.4f}")

    return metrics


def evaluate_physics_conservation(model, dataset, device='cpu', max_samples=10):
    """
    Check if physics laws are conserved

    Checks:
    1. Energy conservation (kinetic energy)
    2. Momentum conservation
    3. Position consistency
    """
    logger.info("Checking physics conservation...")

    model.to(device)
    model.eval()

    energy_errors = []
    momentum_errors = []
    position_errors = []

    dt = 0.01  # Time step

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            sample = dataset[i]

            node_features = sample['node_features'].to(device)
            edge_index = sample['edge_index'].to(device)
            edge_attr = sample['edge_attr'].to(device)

            pos_t = sample['positions_t'].to(device)
            pos_t1 = sample['positions_t1'].to(device)
            vel_t = node_features[:, 3:6]  # Extract velocity
            vel_t1_true = sample['target'].to(device)

            # Predict velocity at t+1
            vel_t1_pred = model(node_features, edge_index, edge_attr)

            # 1. Energy conservation
            # Kinetic energy = 0.5 * m * v^2
            ke_true = torch.sum(vel_t1_true ** 2)
            ke_pred = torch.sum(vel_t1_pred ** 2)
            energy_error = torch.abs(ke_pred - ke_true) / (ke_true + 1e-6)
            energy_errors.append(energy_error.item())

            # 2. Momentum conservation
            # Total momentum = sum(m * v)
            momentum_true = torch.sum(vel_t1_true, dim=0)
            momentum_pred = torch.sum(vel_t1_pred, dim=0)
            momentum_error = torch.norm(momentum_pred - momentum_true) / (torch.norm(momentum_true) + 1e-6)
            momentum_errors.append(momentum_error.item())

            # 3. Position consistency
            # pos_t+1 should be approximately pos_t + vel * dt
            pos_pred = pos_t + vel_t1_pred * dt
            pos_error = torch.mean(torch.norm(pos_pred - pos_t1, dim=1))
            position_errors.append(pos_error.item())

    physics_metrics = {
        'energy_error_mean': np.mean(energy_errors),
        'energy_error_std': np.std(energy_errors),
        'momentum_error_mean': np.mean(momentum_errors),
        'momentum_error_std': np.std(momentum_errors),
        'position_error_mean': np.mean(position_errors),
        'position_error_std': np.std(position_errors)
    }

    logger.info("Physics Conservation Metrics:")
    logger.info(f"  Energy Error: {physics_metrics['energy_error_mean']:.6f} ± {physics_metrics['energy_error_std']:.6f}")
    logger.info(f"  Momentum Error: {physics_metrics['momentum_error_mean']:.6f} ± {physics_metrics['momentum_error_std']:.6f}")
    logger.info(f"  Position Error: {physics_metrics['position_error_mean']:.6f} ± {physics_metrics['position_error_std']:.6f}")

    return physics_metrics


def rollout_comparison(model, dataset, device='cpu', num_steps=20, sample_idx=0):
    """
    Compare multi-step rollout vs ground truth

    Simulates multiple steps and compares with ground truth trajectory
    """
    logger.info(f"Running rollout comparison ({num_steps} steps)...")

    model.to(device)
    model.eval()

    # Get initial state
    sample = dataset[sample_idx]
    positions = sample['positions_t'].numpy()
    velocities = sample['node_features'][:, 3:6].numpy()

    # Ground truth trajectory
    gt_positions = [positions]
    for step in range(1, num_steps):
        if sample_idx + step < len(dataset):
            next_sample = dataset[sample_idx + step]
            gt_positions.append(next_sample['positions_t'].numpy())
        else:
            break

    # Predicted trajectory
    pred_positions = [positions]
    current_pos = positions.copy()
    current_vel = velocities.copy()

    dt = 0.01

    with torch.no_grad():
        for step in range(1, len(gt_positions)):
            # Build graph
            tree = cKDTree(current_pos)
            edges = []
            edge_features = []
            for i, pos in enumerate(current_pos):
                neighbors = tree.query_ball_point(pos, 0.3)
                for j in neighbors:
                    if i < j:
                        edges.extend([[i, j], [j, i]])
                        rel = current_pos[j] - current_pos[i]
                        edge_features.extend([rel, -rel])

            if not edges:
                logger.warning(f"  Step {step}: No edges, stopping rollout")
                break

            edge_index = torch.from_numpy(np.array(edges).T).long().to(device)
            edge_attr = torch.from_numpy(np.array(edge_features).astype(np.float32)).to(device)
            node_features = torch.from_numpy(np.concatenate([current_pos, current_vel], axis=1)).float().to(device)

            # Predict
            vel_update = model(node_features, edge_index, edge_attr)
            current_vel = vel_update.cpu().numpy()
            current_pos = current_pos + current_vel * dt

            # Boundary conditions
            current_pos[:, 2] = np.maximum(current_pos[:, 2], 0)

            pred_positions.append(current_pos.copy())

            if (step + 1) % 5 == 0:
                logger.info(f"  Rollout step {step+1}/{len(gt_positions)}")

    # Compute rollout errors
    rollout_errors = []
    for step in range(min(len(gt_positions), len(pred_positions))):
        error = np.mean(np.linalg.norm(pred_positions[step] - gt_positions[step], axis=1))
        rollout_errors.append(error)

    logger.info(f"Rollout Errors:")
    logger.info(f"  Step 1: {rollout_errors[0]:.6f}")
    logger.info(f"  Step {len(rollout_errors)//2}: {rollout_errors[len(rollout_errors)//2]:.6f}")
    logger.info(f"  Step {len(rollout_errors)-1}: {rollout_errors[-1]:.6f}")

    return {
        'gt_positions': gt_positions,
        'pred_positions': pred_positions,
        'rollout_errors': rollout_errors
    }


def create_evaluation_plots(metrics, physics_metrics, rollout_data):
    """Create evaluation visualization plots"""
    logger.info("Creating evaluation plots...")

    fig = plt.figure(figsize=(16, 10))

    # 1. MSE distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(metrics['mse_list'], bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(metrics['mse_mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["mse_mean"]:.4f}')
    ax1.set_xlabel('MSE')
    ax1.set_ylabel('Frequency')
    ax1.set_title('MSE Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. MAE distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(metrics['mae_list'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(metrics['mae_mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["mae_mean"]:.4f}')
    ax2.set_xlabel('MAE')
    ax2.set_ylabel('Frequency')
    ax2.set_title('MAE Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Relative error distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(metrics['rel_error_list'], bins=20, edgecolor='black', alpha=0.7, color='green')
    ax3.axvline(metrics['rel_error_mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["rel_error_mean"]:.4f}')
    ax3.set_xlabel('Relative Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Relative Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Physics conservation bar chart
    ax4 = plt.subplot(2, 3, 4)
    conservation_names = ['Energy', 'Momentum', 'Position']
    conservation_values = [
        physics_metrics['energy_error_mean'],
        physics_metrics['momentum_error_mean'],
        physics_metrics['position_error_mean']
    ]
    conservation_stds = [
        physics_metrics['energy_error_std'],
        physics_metrics['momentum_error_std'],
        physics_metrics['position_error_std']
    ]
    bars = ax4.bar(conservation_names, conservation_values, yerr=conservation_stds,
                   capsize=5, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Error')
    ax4.set_title('Physics Conservation Errors')
    ax4.grid(True, alpha=0.3, axis='y')

    # Color bars based on error magnitude
    colors = ['green' if v < 0.1 else 'orange' if v < 0.5 else 'red' for v in conservation_values]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 5. Rollout error over time
    ax5 = plt.subplot(2, 3, 5)
    steps = range(len(rollout_data['rollout_errors']))
    ax5.plot(steps, rollout_data['rollout_errors'], marker='o', linewidth=2)
    ax5.set_xlabel('Rollout Step')
    ax5.set_ylabel('Position Error (m)')
    ax5.set_title('Rollout Error Accumulation')
    ax5.grid(True, alpha=0.3)

    # 6. Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
    Model Evaluation Summary
    ========================

    Quantitative Metrics:
    • MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}
    • MAE: {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}
    • Rel Error: {metrics['rel_error_mean']:.4f} ± {metrics['rel_error_std']:.4f}

    Physics Conservation:
    • Energy Error: {physics_metrics['energy_error_mean']:.6f}
    • Momentum Error: {physics_metrics['momentum_error_mean']:.6f}
    • Position Error: {physics_metrics['position_error_mean']:.6f}

    Rollout Performance:
    • Initial Error: {rollout_data['rollout_errors'][0]:.6f}
    • Final Error: {rollout_data['rollout_errors'][-1]:.6f}
    • Error Growth: {rollout_data['rollout_errors'][-1]/rollout_data['rollout_errors'][0]:.2f}x

    Evaluation: {'PASS' if metrics['rel_error_mean'] < 0.5 else 'NEEDS IMPROVEMENT'}
    """

    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.tight_layout()

    plot_path = Path("visualizations") / "model_evaluation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Evaluation plot saved: {plot_path}")

    return plot_path


def main():
    # Load model
    print("\n[1/5] Loading model...")
    model, checkpoint = load_model("models/gnn_sph_best.pt")

    # Load validation dataset
    print("\n[2/5] Loading validation dataset...")
    val_dataset = FLIPDataset("datasets/flip_simulations/sph_splash", sequence_length=2, radius=0.3)
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Quantitative evaluation
    print("\n[3/5] Computing quantitative metrics...")
    metrics = evaluate_metrics(model, val_dataset, device='cuda', max_samples=39)

    # Physics conservation
    print("\n[4/5] Checking physics conservation...")
    physics_metrics = evaluate_physics_conservation(model, val_dataset, device='cuda', max_samples=10)

    # Rollout comparison
    print("\n[5/5] Running rollout comparison...")
    rollout_data = rollout_comparison(model, val_dataset, device='cuda', num_steps=15, sample_idx=0)

    # Create plots
    print("\nCreating evaluation plots...")
    plot_path = create_evaluation_plots(metrics, physics_metrics, rollout_data)

    # Final summary
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"\nResults:")
    print(f"  MSE: {metrics['mse_mean']:.6f}")
    print(f"  MAE: {metrics['mae_mean']:.6f}")
    print(f"  Relative Error: {metrics['rel_error_mean']:.4f}")
    print(f"\nPhysics Conservation:")
    print(f"  Energy Error: {physics_metrics['energy_error_mean']:.6f}")
    print(f"  Momentum Error: {physics_metrics['momentum_error_mean']:.6f}")
    print(f"  Position Error: {physics_metrics['position_error_mean']:.6f}")
    print(f"\nPlot: {plot_path}")
    print(f"Log: {log_file}")
    print("="*60)

    logger.info("="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
