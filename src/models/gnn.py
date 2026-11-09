"""
Graph Neural Network (GNN) for Particle-based Simulations (FLIP, MPM)

Based on DeepMind's MeshGraphNets:
"Learning Mesh-Based Simulation with Graph Networks"
https://arxiv.org/abs/2010.03409

Ideal for FLIP (particle-based fluid) simulations where data is naturally
represented as graphs.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class EdgeModel(nn.Module):
    """Edge update network"""

    def __init__(self, edge_dim: int, node_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = 2 * node_dim + edge_dim  # sender + receiver + edge features

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )

    def forward(
        self,
        src: torch.Tensor,
        dest: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            src: Source node features (num_edges, node_dim)
            dest: Destination node features (num_edges, node_dim)
            edge_attr: Edge features (num_edges, edge_dim)

        Returns:
            Updated edge features (num_edges, edge_dim)
        """
        out = torch.cat([src, dest, edge_attr], dim=-1)
        return self.net(out)


class NodeModel(nn.Module):
    """Node update network"""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = node_dim + edge_dim  # node + aggregated edge features

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, node_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim)

        Returns:
            Updated node features (num_nodes, node_dim)
        """
        row, col = edge_index

        # Aggregate edge features to destination nodes
        agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        agg.index_add_(0, col, edge_attr)

        # Update nodes
        out = torch.cat([x, agg], dim=-1)
        return self.net(out)


class GraphNetworkBlock(nn.Module):
    """Single message-passing block"""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.edge_model = EdgeModel(edge_dim, node_dim, hidden_dim)
        self.node_model = NodeModel(node_dim, edge_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Message passing step

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features

        Returns:
            Updated (node_features, edge_features)
        """
        row, col = edge_index

        # Update edges
        edge_attr_updated = self.edge_model(x[row], x[col], edge_attr)
        edge_attr = edge_attr + edge_attr_updated  # Residual connection

        # Update nodes
        x_updated = self.node_model(x, edge_index, edge_attr)
        x = x + x_updated  # Residual connection

        return x, edge_attr


class MeshGraphNet(nn.Module):
    """
    MeshGraphNet for particle-based fluid simulation

    Suitable for FLIP simulations where particles form the graph.
    """

    def __init__(
        self,
        input_node_dim: int = 10,  # e.g., position, velocity, pressure, etc.
        input_edge_dim: int = 3,   # e.g., relative position
        hidden_dim: int = 128,
        output_dim: int = 3,       # e.g., velocity update
        num_layers: int = 15
    ):
        """
        Args:
            input_node_dim: Dimension of input node features
            input_edge_dim: Dimension of input edge features
            hidden_dim: Hidden dimension for message passing
            output_dim: Output dimension (typically velocity or position update)
            num_layers: Number of message-passing layers
        """
        super().__init__()

        # Encoder: map input features to hidden dimension
        self.node_encoder = nn.Sequential(
            nn.Linear(input_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(input_edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Processor: message-passing layers
        self.processor = nn.ModuleList([
            GraphNetworkBlock(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Decoder: map to output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: (num_nodes, input_node_dim)
            edge_index: (2, num_edges)
            edge_features: (num_edges, input_edge_dim)

        Returns:
            Output predictions (num_nodes, output_dim)
        """
        # Encode
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)

        # Process (message passing)
        for block in self.processor:
            x, edge_attr = block(x, edge_index, edge_attr)

        # Decode
        out = self.decoder(x)

        return out


class GNNPredictor:
    """High-level interface for GNN-based particle simulation"""

    def __init__(self, model: Optional[MeshGraphNet] = None, device: str = "cuda"):
        """
        Args:
            model: Pre-trained MeshGraphNet model
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model if model is not None else MeshGraphNet()
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cuda") -> "GNNPredictor":
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = MeshGraphNet(**checkpoint.get('config', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
        return cls(model, device)

    def predict_next_state(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        other_features: Optional[np.ndarray] = None,
        radius: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Predict next simulation state

        Args:
            positions: Particle positions (N, 3)
            velocities: Particle velocities (N, 3)
            other_features: Additional features (N, D)
            radius: Radius for building graph connectivity

        Returns:
            Dictionary with 'positions', 'velocities', etc.
        """
        # Build graph
        edge_index, edge_features = self._build_graph(positions, radius)

        # Prepare node features
        node_features = np.concatenate([positions, velocities], axis=-1)
        if other_features is not None:
            node_features = np.concatenate([node_features, other_features], axis=-1)

        # Convert to tensors
        node_features = torch.from_numpy(node_features).float().to(self.device)
        edge_index = torch.from_numpy(edge_index).long().to(self.device)
        edge_features = torch.from_numpy(edge_features).float().to(self.device)

        # Inference
        with torch.no_grad():
            velocity_update = self.model(node_features, edge_index, edge_features)

        # Update state
        new_velocities = velocities + velocity_update.cpu().numpy()
        new_positions = positions + new_velocities * 0.01  # Assuming dt=0.01

        return {
            'positions': new_positions,
            'velocities': new_velocities
        }

    def _build_graph(
        self,
        positions: np.ndarray,
        radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph connectivity based on spatial proximity

        Args:
            positions: Particle positions (N, 3)
            radius: Connection radius

        Returns:
            edge_index (2, E), edge_features (E, 3)
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(positions)
        edges = []
        edge_features = []

        for i, pos in enumerate(positions):
            # Find neighbors within radius
            neighbors = tree.query_ball_point(pos, radius)

            for j in neighbors:
                if i != j:
                    edges.append([i, j])
                    # Edge feature: relative position
                    edge_features.append(positions[j] - positions[i])

        edge_index = np.array(edges).T if edges else np.zeros((2, 0), dtype=int)
        edge_features = np.array(edge_features) if edge_features else np.zeros((0, 3))

        return edge_index, edge_features

    def export_onnx(self, output_path: str):
        """Export model to ONNX (more complex due to graph structure)"""
        # TODO: Implement ONNX export for graph models
        raise NotImplementedError("ONNX export for GNN requires special handling")


if __name__ == "__main__":
    # Example usage
    print("MeshGraphNet Model Test")

    # Create model
    model = MeshGraphNet(
        input_node_dim=10,
        input_edge_dim=3,
        hidden_dim=128,
        output_dim=3,
        num_layers=10
    )

    # Create sample graph
    num_nodes = 100
    num_edges = 500

    node_features = torch.randn(num_nodes, 10)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_features = torch.randn(num_edges, 3)

    # Forward pass
    output = model(node_features, edge_index, edge_features)

    print(f"Input nodes: {node_features.shape}")
    print(f"Edge index: {edge_index.shape}")
    print(f"Output: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
