"""
Hybrid Physics-Guided GNN

Combines MeshGraphNet architecture with attention-based importance weighting
learned from GAT analysis. Uses physical insights about boundary particle
importance to guide message passing.

Key Features:
1. Explicit particle type handling (fluid vs boundary)
2. Learned attention gates for edge importance
3. Distance-aware message weighting
4. Optional physics-based heuristics from GAT analysis
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class AttentionGate(nn.Module):
    """
    Learned attention gate for edge importance

    Computes importance weights for edges based on:
    - Source and destination node features
    - Edge features (relative position)
    - Particle types (fluid/boundary)
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64):
        super().__init__()

        input_dim = 2 * node_dim + edge_dim  # src + dst + edge

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output: 0 to 1 importance weight
        )

    def forward(self, src_features, dst_features, edge_features):
        """
        Args:
            src_features: Source node features (num_edges, node_dim)
            dst_features: Destination node features (num_edges, node_dim)
            edge_features: Edge features (num_edges, edge_dim)

        Returns:
            Importance weights (num_edges, 1) in range [0, 1]
        """
        combined = torch.cat([src_features, dst_features, edge_features], dim=-1)
        return self.gate_net(combined)


class PhysicsGuidedEdgeModel(nn.Module):
    """
    Edge update model with physics-guided attention

    Incorporates:
    1. Learned attention gates
    2. Particle type importance (from GAT analysis)
    3. Distance-based weighting
    """

    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        hidden_dim: int = 128,
        use_learned_attention: bool = True,
        use_type_weighting: bool = True,
        boundary_importance: float = 2.0
    ):
        super().__init__()

        self.use_learned_attention = use_learned_attention
        self.use_type_weighting = use_type_weighting
        self.boundary_importance = boundary_importance

        input_dim = 2 * node_dim + edge_dim

        # Standard edge network
        self.edge_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )

        # Optional learned attention gate
        if use_learned_attention:
            self.attention_gate = AttentionGate(node_dim, edge_dim, hidden_dim)

    def forward(
        self,
        src: torch.Tensor,
        dest: torch.Tensor,
        edge_attr: torch.Tensor,
        particle_types: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source node features (num_edges, node_dim)
            dest: Destination node features (num_edges, node_dim)
            edge_attr: Edge features (num_edges, edge_dim)
            particle_types: Particle types (num_nodes,) - 0=fluid, 1=boundary
            edge_index: Edge connectivity (2, num_edges)

        Returns:
            Updated edge features (num_edges, edge_dim)
        """
        # Standard edge update
        combined = torch.cat([src, dest, edge_attr], dim=-1)
        edge_update = self.edge_net(combined)

        # Apply attention weighting
        if self.use_learned_attention:
            attention_weights = self.attention_gate(src, dest, edge_attr)
            edge_update = attention_weights * edge_update

        # Apply physics-based type weighting (from GAT analysis)
        if self.use_type_weighting and particle_types is not None and edge_index is not None:
            row, col = edge_index

            # Check if interaction involves boundary particle
            is_boundary_interaction = (particle_types[row] > 0.5) | (particle_types[col] > 0.5)

            # Increase importance of boundary interactions
            type_weights = torch.where(
                is_boundary_interaction,
                torch.tensor(self.boundary_importance, device=edge_update.device),
                torch.tensor(1.0, device=edge_update.device)
            ).unsqueeze(-1)

            edge_update = type_weights * edge_update

        return edge_update


class PhysicsGuidedNodeModel(nn.Module):
    """Node update model (unchanged from MeshGraphNet)"""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = node_dim + edge_dim

        self.node_net = nn.Sequential(
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
        combined = torch.cat([x, agg], dim=-1)
        return self.node_net(combined)


class HybridGraphBlock(nn.Module):
    """Single message-passing block with physics-guided attention"""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        use_learned_attention: bool = True,
        use_type_weighting: bool = True,
        boundary_importance: float = 2.0
    ):
        super().__init__()

        self.edge_model = PhysicsGuidedEdgeModel(
            edge_dim, node_dim, hidden_dim,
            use_learned_attention, use_type_weighting, boundary_importance
        )
        self.node_model = PhysicsGuidedNodeModel(node_dim, edge_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        particle_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Message passing step with physics-guided attention

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            particle_types: Particle types (0=fluid, 1=boundary)

        Returns:
            Updated (node_features, edge_features)
        """
        row, col = edge_index

        # Update edges with attention
        edge_attr_updated = self.edge_model(
            x[row], x[col], edge_attr,
            particle_types=particle_types,
            edge_index=edge_index
        )
        edge_attr = edge_attr + edge_attr_updated  # Residual connection

        # Update nodes
        x_updated = self.node_model(x, edge_index, edge_attr)
        x = x + x_updated  # Residual connection

        return x, edge_attr


class HybridPhysicsGNN(nn.Module):
    """
    Hybrid GNN combining MeshGraphNet architecture with GAT-inspired attention

    Architecture:
    - Encoder: Transform input features to hidden space
    - Processor: Stack of physics-guided message passing blocks
    - Decoder: Map to output predictions

    Key innovations:
    1. Learned attention gates for edge importance
    2. Physics-based boundary particle weighting (from GAT analysis)
    3. Maintains efficiency of MeshGraphNet
    4. Interpretable attention weights
    """

    def __init__(
        self,
        input_node_dim: int = 7,       # pos(3) + vel(3) + type(1)
        input_edge_dim: int = 3,        # relative position
        hidden_dim: int = 64,
        output_dim: int = 3,            # velocity prediction
        num_layers: int = 5,
        use_learned_attention: bool = True,
        use_type_weighting: bool = True,
        boundary_importance: float = 2.0  # From GAT analysis
    ):
        """
        Args:
            input_node_dim: Input node feature dimension
            input_edge_dim: Input edge feature dimension
            hidden_dim: Hidden dimension for processing
            output_dim: Output dimension (velocity prediction)
            num_layers: Number of message passing layers
            use_learned_attention: Use learned attention gates
            use_type_weighting: Use physics-based type weighting
            boundary_importance: Importance multiplier for boundary interactions
        """
        super().__init__()

        self.input_node_dim = input_node_dim
        self.input_edge_dim = input_edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_learned_attention = use_learned_attention
        self.use_type_weighting = use_type_weighting
        self.boundary_importance = boundary_importance

        # Encoder
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

        # Processor: Stack of hybrid blocks
        self.processor_blocks = nn.ModuleList([
            HybridGraphBlock(
                hidden_dim, hidden_dim, hidden_dim,
                use_learned_attention, use_type_weighting, boundary_importance
            )
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            node_features: Input node features (num_nodes, input_node_dim)
                           Should include particle type as last feature
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge features (num_edges, input_edge_dim)

        Returns:
            Predicted velocities (num_nodes, output_dim)
        """
        # Extract particle types from node features (last dimension)
        particle_types = node_features[:, -1]

        # Encode
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_attr)

        # Process with hybrid blocks
        for block in self.processor_blocks:
            x, edge_attr = block(x, edge_index, edge_attr, particle_types)

        # Decode
        output = self.decoder(x)

        return output

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_weights(self, node_features, edge_index, edge_attr):
        """
        Extract attention weights for analysis

        Useful for validating that the model learned correct importance patterns
        """
        if not self.use_learned_attention:
            return None

        particle_types = node_features[:, -1]
        x = self.node_encoder(node_features)
        edge_attr_enc = self.edge_encoder(edge_attr)

        row, col = edge_index

        # Get attention weights from first layer
        first_block = self.processor_blocks[0]
        attention_weights = first_block.edge_model.attention_gate(
            x[row], x[col], edge_attr_enc
        )

        return attention_weights.detach().cpu().numpy()


def create_hybrid_from_insights(insights_file: str, **kwargs):
    """
    Create hybrid model with parameters learned from GAT analysis

    Args:
        insights_file: Path to JSON file with GAT insights
        **kwargs: Additional model parameters

    Returns:
        Configured HybridPhysicsGNN model
    """
    import json

    with open(insights_file, 'r') as f:
        insights = json.load(f)

    # Compute average boundary importance across scenarios
    boundary_ratios = [v['boundary_importance_ratio'] for v in insights.values()]
    avg_boundary_importance = np.mean(boundary_ratios)

    print(f"Creating hybrid model with boundary_importance={avg_boundary_importance:.3f}")

    return HybridPhysicsGNN(
        boundary_importance=avg_boundary_importance,
        **kwargs
    )
