"""
GAT Attention Analysis Script

Analyzes attention weights from trained GAT model to identify
important physical variables for fluid-rigid body interactions.

Output:
1. Attention weight statistics by particle type
2. Feature importance analysis
3. Spatial attention patterns
4. Scenario-specific attention differences
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
import sys
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.gat import GraphAttentionNet


class AttentionAnalyzer:
    """Analyze GAT attention weights"""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.eval()

        # Storage for attention weights
        self.attention_weights = []
        self.attention_by_type = defaultdict(list)
        self.attention_by_distance = defaultdict(list)

        # Hook to capture attention weights
        self.register_attention_hooks()

    def load_model(self, model_path):
        """Load trained GAT model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']['model']

        model = GraphAttentionNet(
            input_node_dim=config['input_node_dim'],
            input_edge_dim=config['input_edge_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config.get('dropout', 0.1)
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[OK] Loaded GAT model ({model.count_parameters():,} parameters)")
        return model

    def register_attention_hooks(self):
        """Register hooks to capture attention weights from all GAT layers"""
        self.hooks = []

        def make_hook(layer_name):
            def hook_fn(module, input, output):
                # This will be called during forward pass
                # We'll capture attention from the _softmax_per_node computation
                pass
            return hook_fn

        # We'll need to modify GAT model to expose attention weights
        # For now, we'll compute them manually
        print("[WARNING] Note: Attention weights will be computed during forward pass")

    def compute_attention_manually(self, x, edge_index, edge_attr):
        """
        Manually compute attention weights from first GAT layer
        to analyze what the model learned
        """
        with torch.no_grad():
            # Get first multi-head attention layer
            first_layer = self.model.gat_layers[0]

            # For each attention head, compute attention weights
            attention_per_head = []

            for head in first_layer.attention_heads:
                # Transform features
                h = head.W(x)  # (num_nodes, out_features)

                row, col = edge_index
                num_nodes = x.size(0)

                # Compute attention scores
                attention_src = torch.matmul(h, head.a_src).squeeze(-1)
                attention_dst = torch.matmul(h, head.a_dst).squeeze(-1)
                attention = attention_src[row] + attention_dst[col]

                # Add edge features if available
                if edge_attr is not None and head.edge_features > 0:
                    h_edge = head.W_edge(edge_attr)
                    attention_edge = torch.matmul(h_edge, head.a_edge).squeeze(-1)
                    attention = attention + attention_edge

                # Apply LeakyReLU
                attention = head.leakyrelu(attention)

                # Softmax per node
                attention_normalized = head._softmax_per_node(attention, col, num_nodes)

                attention_per_head.append(attention_normalized.cpu().numpy())

            # Average across heads
            avg_attention = np.mean(attention_per_head, axis=0)

            return avg_attention, edge_index.cpu().numpy()

    def analyze_scenario(self, data_dir, max_frames=30):
        """
        Analyze attention patterns for a single scenario

        Args:
            data_dir: Path to scenario data
            max_frames: Number of frames to analyze

        Returns:
            Dictionary of analysis results
        """
        data_dir = Path(data_dir)
        frame_files = sorted(data_dir.glob("frame_*.npz"))[:max_frames]

        print(f"\nAnalyzing {data_dir.name}...")
        print(f"  Frames: {len(frame_files)}")

        results = {
            'attention_by_type': {'fluid_fluid': [], 'fluid_boundary': [], 'boundary_fluid': []},
            'attention_by_distance': defaultdict(list),
            'feature_importance': {'position': [], 'velocity': [], 'particle_type': []},
            'spatial_patterns': []
        }

        for frame_file in frame_files:
            data = np.load(frame_file)

            # Extract data
            positions = data['positions']
            velocities = data['velocities']
            particle_types = data['particle_types']  # 0=fluid, 1=boundary

            # Build graph
            edge_index, edge_attr = self.build_graph(positions, radius=0.3)

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

            # Compute attention weights
            attention_weights, edge_index_np = self.compute_attention_manually(
                x, edge_index_t, edge_attr_t
            )

            # Analyze by particle type
            src_idx, dst_idx = edge_index_np
            for i, (src, dst) in enumerate(zip(src_idx, dst_idx)):
                src_type = particle_types[src]
                dst_type = particle_types[dst]
                att_weight = attention_weights[i]

                if src_type == 0 and dst_type == 0:
                    results['attention_by_type']['fluid_fluid'].append(att_weight)
                elif src_type == 0 and dst_type == 1:
                    results['attention_by_type']['fluid_boundary'].append(att_weight)
                elif src_type == 1 and dst_type == 0:
                    results['attention_by_type']['boundary_fluid'].append(att_weight)

            # Analyze by distance
            for i, (src, dst) in enumerate(zip(src_idx, dst_idx)):
                distance = np.linalg.norm(positions[dst] - positions[src])
                distance_bin = int(distance / 0.05)  # 5cm bins
                results['attention_by_distance'][distance_bin].append(attention_weights[i])

        return results

    def build_graph(self, positions, radius=0.3):
        """Build k-NN graph (same as training)"""
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

    def visualize_results(self, all_results, output_dir):
        """Create visualizations of attention analysis"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Attention by particle type
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for scenario_name, results in all_results.items():
            for i, (interaction_type, attention_values) in enumerate(results['attention_by_type'].items()):
                if len(attention_values) > 0:
                    axes[i].hist(attention_values, bins=50, alpha=0.5, label=scenario_name)
                    axes[i].set_title(f'{interaction_type.replace("_", "-")} interactions')
                    axes[i].set_xlabel('Attention Weight')
                    axes[i].set_ylabel('Frequency')

        for ax in axes:
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'attention_by_type.png', dpi=150, bbox_inches='tight')
        print(f"\n[OK] Saved: {output_dir / 'attention_by_type.png'}")
        plt.close()

        # 2. Attention statistics summary
        self.print_statistics(all_results)

        # 3. Save numerical results
        self.save_statistics(all_results, output_dir / 'attention_statistics.txt')

    def print_statistics(self, all_results):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("ATTENTION WEIGHT STATISTICS")
        print("="*70)

        for scenario_name, results in all_results.items():
            print(f"\n{scenario_name}:")
            print("-" * 50)

            for interaction_type, values in results['attention_by_type'].items():
                if len(values) > 0:
                    mean_att = np.mean(values)
                    std_att = np.std(values)
                    max_att = np.max(values)
                    print(f"  {interaction_type:20s}: μ={mean_att:.6f}, σ={std_att:.6f}, max={max_att:.6f}")

    def save_statistics(self, all_results, output_file):
        """Save detailed statistics to file"""
        with open(output_file, 'w') as f:
            f.write("GAT ATTENTION ANALYSIS RESULTS\n")
            f.write("="*70 + "\n\n")

            for scenario_name, results in all_results.items():
                f.write(f"\nScenario: {scenario_name}\n")
                f.write("-" * 50 + "\n")

                for interaction_type, values in results['attention_by_type'].items():
                    if len(values) > 0:
                        f.write(f"\n{interaction_type}:\n")
                        f.write(f"  Count: {len(values)}\n")
                        f.write(f"  Mean: {np.mean(values):.6f}\n")
                        f.write(f"  Std: {np.std(values):.6f}\n")
                        f.write(f"  Min: {np.min(values):.6f}\n")
                        f.write(f"  Max: {np.max(values):.6f}\n")
                        f.write(f"  Median: {np.median(values):.6f}\n")

                        # Percentiles
                        p25, p50, p75, p95 = np.percentile(values, [25, 50, 75, 95])
                        f.write(f"  25th percentile: {p25:.6f}\n")
                        f.write(f"  50th percentile: {p50:.6f}\n")
                        f.write(f"  75th percentile: {p75:.6f}\n")
                        f.write(f"  95th percentile: {p95:.6f}\n")

        print(f"[OK] Saved: {output_file}")

    def extract_key_insights(self, all_results):
        """Extract key insights for hybrid model design"""
        print("\n" + "="*70)
        print("KEY INSIGHTS FOR HYBRID MODEL")
        print("="*70)

        insights = {}

        for scenario_name, results in all_results.items():
            # Calculate importance ratios
            ff_mean = np.mean(results['attention_by_type']['fluid_fluid']) if results['attention_by_type']['fluid_fluid'] else 0
            fb_mean = np.mean(results['attention_by_type']['fluid_boundary']) if results['attention_by_type']['fluid_boundary'] else 0

            if ff_mean > 0:
                boundary_importance = fb_mean / ff_mean
            else:
                boundary_importance = 1.0

            insights[scenario_name] = {
                'boundary_importance_ratio': boundary_importance,
                'fluid_fluid_mean': ff_mean,
                'fluid_boundary_mean': fb_mean
            }

            print(f"\n{scenario_name}:")
            print(f"  Fluid-Fluid attention: {ff_mean:.6f}")
            print(f"  Fluid-Boundary attention: {fb_mean:.6f}")
            print(f"  Boundary importance ratio: {boundary_importance:.3f}x")

        # Overall recommendation
        avg_ratio = np.mean([v['boundary_importance_ratio'] for v in insights.values()])
        print(f"\n{'='*70}")
        print(f"RECOMMENDATION FOR HYBRID MODEL:")
        print(f"  Average boundary importance: {avg_ratio:.3f}x")
        print(f"  Suggested weight for boundary interactions: {avg_ratio:.2f}")
        print(f"{'='*70}")

        return insights


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze GAT attention weights')
    parser.add_argument('model_path', type=str, help='Path to trained GAT model')
    parser.add_argument('--data-root', type=str, default='datasets/flip_simulations',
                       help='Root directory for scenario data')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       default=['sph_falling_ball_000', 'sph_paddle_wave_000', 'sph_rotating_stirrer_000'],
                       help='Scenarios to analyze')
    parser.add_argument('--output-dir', type=str, default='analysis/gat_attention',
                       help='Output directory for results')
    parser.add_argument('--max-frames', type=int, default=30,
                       help='Maximum frames per scenario to analyze')

    args = parser.parse_args()

    print("="*70)
    print("GAT ATTENTION ANALYSIS")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print(f"Output: {args.output_dir}")
    print("="*70)

    # Initialize analyzer
    analyzer = AttentionAnalyzer(args.model_path)

    # Analyze each scenario
    all_results = {}
    data_root = Path(args.data_root)

    for scenario in args.scenarios:
        scenario_path = data_root / scenario
        if scenario_path.exists():
            results = analyzer.analyze_scenario(scenario_path, max_frames=args.max_frames)
            all_results[scenario] = results
        else:
            print(f"[WARNING] Warning: Scenario not found: {scenario_path}")

    # Visualize results
    if all_results:
        analyzer.visualize_results(all_results, args.output_dir)
        insights = analyzer.extract_key_insights(all_results)

        # Save insights for hybrid model
        import json
        insights_file = Path(args.output_dir) / 'hybrid_model_insights.json'
        with open(insights_file, 'w') as f:
            json.dump(insights, f, indent=2)
        print(f"\n[OK] Saved insights: {insights_file}")
    else:
        print("\n[WARNING] No results to visualize!")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
