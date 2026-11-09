"""
Hybrid Physics-Guided GNN — Tuned variant (paper-figure compliant).

Tier 1 + Tier 2 modifications, applied to Hybrid only:
  Tier 1
    - AttentionGate bias initialised to +2.0  (sigmoid -> ~0.88 at start)
    - LayerNorm inside φ_e (PhysicsGuidedEdgeModelTuned.edge_net)
    - LayerNorm inside φ_v (PhysicsGuidedNodeModelTuned.node_net)
    - Dropout p=0.1 in AttentionGateTuned
  Tier 2
    - LayerNorm in node_encoder / edge_encoder / decoder
    - Edge & Node residual scale 1.0 -> 0.5
    - AttentionGateTuned internal hidden h//2 -> h (32 -> 64 by default)

The architecture diagram in the paper (5-layer Encoder-Processor-Decoder,
ε_v/ε_e -> φ_e + α_ij + w_ij -> ρ_e→v -> φ_v -> δ, residual on edge & node)
is preserved exactly.  hidden_dim, num_layers, β=2.0 (boundary_importance),
input/output dims, K-NN radius, and the Δh_ij = w_ij·α_ij·~h_ij formula all
remain identical to the original.

The original `src/models/hybrid_gnn.py` is NOT modified; this is a separate
file used by the new `--arch hybrid_tuned` option in the training driver.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class AttentionGateTuned(nn.Module):
    """Attention gate with bias-2 init, larger internal dim, dropout."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        input_dim = 2 * node_dim + edge_dim   # src + dst + edge

        # Tier 2: internal hidden = hidden_dim (vs h//2 originally)
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),                  # Tier 1: dropout
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Tier 1: bias init = +2 so sigmoid(2)≈0.88 — start near identity
        # so attention does not cripple the residual at training start.
        with torch.no_grad():
            self.gate_net[-2].bias.fill_(2.0)

    def forward(self, src_features, dst_features, edge_features):
        combined = torch.cat([src_features, dst_features, edge_features], dim=-1)
        return self.gate_net(combined)


class PhysicsGuidedEdgeModelTuned(nn.Module):
    """Edge model φ_e with LayerNorm + Dropout inside the 3-layer MLP."""

    def __init__(self, edge_dim: int, node_dim: int, hidden_dim: int = 64,
                 use_learned_attention: bool = True,
                 use_type_weighting: bool = True,
                 boundary_importance: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.use_learned_attention = use_learned_attention
        self.use_type_weighting = use_type_weighting
        self.boundary_importance = boundary_importance

        input_dim = 2 * node_dim + edge_dim

        # Tier 1: LayerNorm + Dropout inside the 3-layer MLP.
        # Paper figure shows the MLP as "FC→ReLU→FC→ReLU→FC" — internal
        # LayerNorm/Dropout is not visible in the figure, so adding them is
        # diagram-compatible.
        self.edge_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim),
        )

        if use_learned_attention:
            self.attention_gate = AttentionGateTuned(node_dim, edge_dim, hidden_dim, dropout=dropout)

    def forward(self, src, dest, edge_attr,
                particle_types: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        combined = torch.cat([src, dest, edge_attr], dim=-1)
        edge_update = self.edge_net(combined)

        if self.use_learned_attention:
            attention_weights = self.attention_gate(src, dest, edge_attr)
            edge_update = attention_weights * edge_update

        if self.use_type_weighting and particle_types is not None and edge_index is not None:
            row, col = edge_index
            is_boundary_interaction = (particle_types[row] > 0.5) | (particle_types[col] > 0.5)
            type_weights = torch.where(
                is_boundary_interaction,
                torch.tensor(self.boundary_importance, device=edge_update.device),
                torch.tensor(1.0, device=edge_update.device),
            ).unsqueeze(-1)
            edge_update = type_weights * edge_update

        return edge_update


class PhysicsGuidedNodeModelTuned(nn.Module):
    """Node model φ_v with LayerNorm + Dropout inside the 3-layer MLP."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        input_dim = node_dim + edge_dim
        self.node_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        agg.index_add_(0, col, edge_attr)
        combined = torch.cat([x, agg], dim=-1)
        return self.node_net(combined)


class HybridGraphBlockTuned(nn.Module):
    """Single processor block with Tier 1+2 mods and residual scale 0.5."""

    RESIDUAL_SCALE = 0.5     # Tier 2: was 1.0 (full add)

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64,
                 use_learned_attention: bool = True,
                 use_type_weighting: bool = True,
                 boundary_importance: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.edge_model = PhysicsGuidedEdgeModelTuned(
            edge_dim, node_dim, hidden_dim,
            use_learned_attention, use_type_weighting, boundary_importance,
            dropout=dropout,
        )
        self.node_model = PhysicsGuidedNodeModelTuned(node_dim, edge_dim, hidden_dim, dropout=dropout)

    def forward(self, x, edge_index, edge_attr,
                particle_types: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        edge_attr_updated = self.edge_model(
            x[row], x[col], edge_attr,
            particle_types=particle_types, edge_index=edge_index,
        )
        edge_attr = edge_attr + self.RESIDUAL_SCALE * edge_attr_updated   # Tier 2

        x_updated = self.node_model(x, edge_index, edge_attr)
        x = x + self.RESIDUAL_SCALE * x_updated                            # Tier 2

        return x, edge_attr


class HybridPhysicsGNNTuned(nn.Module):
    """Hybrid Physics-Guided GNN with Tier 1+2 modifications.

    Externally identical signature to HybridPhysicsGNN — same input/output
    dims, same hidden_dim, num_layers, boundary_importance defaults.
    """

    def __init__(
        self,
        input_node_dim: int = 7,
        input_edge_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 3,
        num_layers: int = 5,
        use_learned_attention: bool = True,
        use_type_weighting: bool = True,
        boundary_importance: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_node_dim = input_node_dim
        self.input_edge_dim = input_edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_learned_attention = use_learned_attention
        self.use_type_weighting = use_type_weighting
        self.boundary_importance = boundary_importance
        self.dropout = dropout

        # Tier 2: LayerNorm in encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.processor_blocks = nn.ModuleList([
            HybridGraphBlockTuned(
                hidden_dim, hidden_dim, hidden_dim,
                use_learned_attention, use_type_weighting, boundary_importance,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Tier 2: LayerNorm in decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, node_features, edge_index, edge_attr) -> torch.Tensor:
        particle_types = node_features[:, -1]
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_attr)
        for block in self.processor_blocks:
            x, edge_attr = block(x, edge_index, edge_attr, particle_types)
        output = self.decoder(x)
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
