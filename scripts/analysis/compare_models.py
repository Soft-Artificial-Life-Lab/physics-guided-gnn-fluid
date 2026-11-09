"""
Model Comparison Script

Compares performance of GNN, GAT, and Hybrid models on rigid body scenarios.
Generates comprehensive comparison metrics and visualizations.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
import sys
import time
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.gnn import MeshGraphNet
from src.models.gat import GraphAttentionNet
from src.models.hybrid_gnn import HybridPhysicsGNN


class ModelComparator:
    """Compare multiple models on fluid-rigid body simulation"""

    def __init__(self, model_paths, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.configs = {}

        for name, path in model_paths.items():
            model, config = self.load_model(path)
            self.models[name] = model
            self.configs[name] = config

        print(f"\n[OK] Loaded {len(self.models)} models")

    def load_model(self, model_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']['model']

        # Determine model type from config
        if 'num_heads' in config:
            # GAT model
            model = GraphAttentionNet(
                input_node_dim=config['input_node_dim'],
                input_edge_dim=config['input_edge_dim'],
                hidden_dim=config['hidden_dim'],
                output_dim=config['output_dim'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                dropout=config.get('dropout', 0.1)
            )
            model_type = 'GAT'
        elif 'use_learned_attention' in config or 'boundary_importance' in config:
            # Hybrid model
            model = HybridPhysicsGNN(
                input_node_dim=config['input_node_dim'],
                input_edge_dim=config['input_edge_dim'],
                hidden_dim=config['hidden_dim'],
                output_dim=config['output_dim'],
                num_layers=config['num_layers'],
                use_learned_attention=config.get('use_learned_attention', True),
                use_type_weighting=config.get('use_type_weighting', True),
                boundary_importance=config.get('boundary_importance', 2.0)
            )
            model_type = 'Hybrid'
        else:
            # Standard GNN
            model = MeshGraphNet(
                input_node_dim=config['input_node_dim'],
                input_edge_dim=config['input_edge_dim'],
                hidden_dim=config['hidden_dim'],
                output_dim=config['output_dim'],
                num_layers=config['num_layers']
            )
            model_type = 'GNN'

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"  Loaded {model_type}: {sum(p.numel() for p in model.parameters()):,} parameters")

        return model, config

    def evaluate_scenario(self, scenario_path, max_frames=50, radius=0.3):
        """
        Evaluate all models on a scenario

        Returns:
            Dictionary with metrics for each model
        """
        scenario_path = Path(scenario_path)
        frame_files = sorted(scenario_path.glob("frame_*.npz"))[:max_frames]

        print(f"\nEvaluating {scenario_path.name}...")
        print(f"  Frames: {len(frame_files)}")

        results = {name: defaultdict(list) for name in self.models.keys()}

        for i, frame_file in enumerate(frame_files[:-1]):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(frame_files)-1}")

            # Load data
            data_t = np.load(frame_file)
            data_t1 = np.load(frame_files[i + 1])

            positions = data_t['positions']
            velocities = data_t['velocities']
            particle_types = data_t['particle_types']
            true_vel_next = data_t1['velocities']

            # Build graph
            edge_index, edge_attr = self.build_graph(positions, radius)

            # Create node features
            node_features = np.concatenate([
                positions,
                velocities,
                particle_types[:, np.newaxis]
            ], axis=1)

            # Convert to tensors
            x = torch.FloatTensor(node_features).to(self.device)
            edge_index_t = torch.LongTensor(edge_index).to(self.device)
            edge_attr_t = torch.FloatTensor(edge_attr).to(self.device)
            true_vel = torch.FloatTensor(true_vel_next).to(self.device)

            # Evaluate each model
            for model_name, model in self.models.items():
                with torch.no_grad():
                    start_time = time.time()
                    pred_vel = model(x, edge_index_t, edge_attr_t)
                    inference_time = time.time() - start_time

                # Compute metrics
                is_fluid = particle_types < 0.5

                # Overall MSE
                mse = torch.mean((pred_vel - true_vel) ** 2).item()
                results[model_name]['mse_overall'].append(mse)

                # Fluid-only MSE
                pred_vel_fluid = pred_vel[is_fluid]
                true_vel_fluid = true_vel[is_fluid]
                mse_fluid = torch.mean((pred_vel_fluid - true_vel_fluid) ** 2).item()
                results[model_name]['mse_fluid'].append(mse_fluid)

                # Inference time
                results[model_name]['inference_time'].append(inference_time)

        return results

    def build_graph(self, positions, radius=0.3):
        """Build k-NN graph"""
        tree = cKDTree(positions)
        edges = []
        edge_features = []

        for i, pos in enumerate(positions):
            neighbors = tree.query_ball_point(pos, radius)
            for j in neighbors:
                if i != j:
                    edges.append([i, j])
                    edge_features.append(positions[j] - positions[i])

        if len(edges) == 0:
            return np.array([[0], [0]]), np.array([[0, 0, 0]])

        return np.array(edges).T, np.array(edge_features)

    def compare_all(self, scenarios, output_dir):
        """Compare all models on all scenarios"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}

        for scenario in scenarios:
            results = self.evaluate_scenario(scenario)
            all_results[scenario.name] = results

        # Generate comparison report
        self.generate_report(all_results, output_dir)

        # Create visualizations
        self.create_visualizations(all_results, output_dir)

        return all_results

    def generate_report(self, all_results, output_dir):
        """Generate text report"""
        report_file = output_dir / 'comparison_report.txt'

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")

            # Model info
            f.write("MODELS:\n")
            f.write("-" * 80 + "\n")
            for name, model in self.models.items():
                param_count = sum(p.numel() for p in model.parameters())
                f.write(f"  {name}: {param_count:,} parameters\n")

            # Per-scenario results
            for scenario_name, results in all_results.items():
                f.write(f"\n\nSCENARIO: {scenario_name}\n")
                f.write("-" * 80 + "\n")

                for model_name in results.keys():
                    mse_fluid = np.mean(results[model_name]['mse_fluid'])
                    inference_time = np.mean(results[model_name]['inference_time']) * 1000  # ms

                    f.write(f"\n{model_name}:\n")
                    f.write(f"  MSE (fluid): {mse_fluid:.8f}\n")
                    f.write(f"  Inference time: {inference_time:.2f} ms\n")

            # Overall comparison
            f.write("\n\n" + "="*80 + "\n")
            f.write("OVERALL COMPARISON\n")
            f.write("="*80 + "\n")

            overall_mse = {name: [] for name in self.models.keys()}
            overall_time = {name: [] for name in self.models.keys()}

            for scenario_name, results in all_results.items():
                for model_name in results.keys():
                    overall_mse[model_name].extend(results[model_name]['mse_fluid'])
                    overall_time[model_name].extend(results[model_name]['inference_time'])

            # Rank by MSE
            mse_ranking = sorted(overall_mse.items(), key=lambda x: np.mean(x[1]))

            f.write("\nRanking by MSE (lower is better):\n")
            for rank, (model_name, mse_values) in enumerate(mse_ranking, 1):
                mean_mse = np.mean(mse_values)
                std_mse = np.std(mse_values)
                f.write(f"  {rank}. {model_name}: {mean_mse:.8f} ± {std_mse:.8f}\n")

            # Speed comparison
            f.write("\nInference Speed:\n")
            for model_name, times in overall_time.items():
                mean_time = np.mean(times) * 1000
                std_time = np.std(times) * 1000
                f.write(f"  {model_name}: {mean_time:.2f} ± {std_time:.2f} ms\n")

        print(f"\n[OK] Generated report: {report_file}")

    def create_visualizations(self, all_results, output_dir):
        """Create comparison visualizations"""

        # 1. MSE comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Box plot for MSE
        data_for_box = []
        labels_for_box = []

        for model_name in self.models.keys():
            all_mse = []
            for scenario_name, results in all_results.items():
                all_mse.extend(results[model_name]['mse_fluid'])
            data_for_box.append(all_mse)
            labels_for_box.append(model_name)

        axes[0].boxplot(data_for_box, labels=labels_for_box)
        axes[0].set_ylabel('MSE (Fluid Particles)')
        axes[0].set_title('Model Comparison: Prediction Error')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        # Inference time comparison
        time_means = []
        time_stds = []
        for model_name in self.models.keys():
            all_times = []
            for scenario_name, results in all_results.items():
                all_times.extend(results[model_name]['inference_time'])
            time_means.append(np.mean(all_times) * 1000)
            time_stds.append(np.std(all_times) * 1000)

        x_pos = np.arange(len(labels_for_box))
        axes[1].bar(x_pos, time_means, yerr=time_stds, capsize=5)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels_for_box)
        axes[1].set_ylabel('Inference Time (ms)')
        axes[1].set_title('Model Comparison: Speed')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir / 'model_comparison.png'}")
        plt.close()

        # 2. Per-scenario comparison
        scenario_names = list(all_results.keys())
        num_scenarios = len(scenario_names)

        fig, axes = plt.subplots(1, num_scenarios, figsize=(5*num_scenarios, 5))
        if num_scenarios == 1:
            axes = [axes]

        for i, scenario_name in enumerate(scenario_names):
            results = all_results[scenario_name]

            for model_name in self.models.keys():
                mse_values = results[model_name]['mse_fluid']
                axes[i].plot(mse_values, label=model_name, alpha=0.7)

            axes[i].set_xlabel('Frame')
            axes[i].set_ylabel('MSE')
            axes[i].set_title(scenario_name.replace('sph_', '').replace('_000', ''))
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yscale('log')

        plt.tight_layout()
        plt.savefig(output_dir / 'per_scenario_comparison.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir / 'per_scenario_comparison.png'}")
        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare GNN, GAT, and Hybrid models')
    parser.add_argument('--gnn', type=str, required=True, help='Path to GNN model')
    parser.add_argument('--gat', type=str, required=True, help='Path to GAT model')
    parser.add_argument('--hybrid', type=str, required=True, help='Path to Hybrid model')
    parser.add_argument('--data-root', type=str, default='datasets/flip_simulations',
                       help='Root directory for scenario data')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       default=['sph_falling_ball_000', 'sph_paddle_wave_000', 'sph_rotating_stirrer_000'],
                       help='Scenarios to evaluate')
    parser.add_argument('--output-dir', type=str, default='analysis/model_comparison',
                       help='Output directory')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Max frames per scenario')

    args = parser.parse_args()

    print("="*80)
    print("MODEL COMPARISON: GNN vs GAT vs HYBRID")
    print("="*80)

    # Load models
    model_paths = {
        'GNN': args.gnn,
        'GAT': args.gat,
        'Hybrid': args.hybrid
    }

    comparator = ModelComparator(model_paths)

    # Prepare scenarios
    data_root = Path(args.data_root)
    scenarios = [data_root / s for s in args.scenarios if (data_root / s).exists()]

    print(f"\nScenarios: {len(scenarios)}")
    for s in scenarios:
        print(f"  - {s.name}")

    # Run comparison
    results = comparator.compare_all(scenarios, args.output_dir)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
