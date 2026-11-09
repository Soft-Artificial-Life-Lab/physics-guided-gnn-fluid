"""
Graph Attention Network (GAT) for Particle-based Simulations

Based on "Graph Attention Networks" (Veličković et al., ICLR 2018)
https://arxiv.org/abs/1710.10903

GAT uses attention mechanisms to assign different weights to different neighbors,
allowing the model to focus on the most relevant particles for each prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer

    Computes attention weights between nodes and aggregates neighbor features
    using those weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_features: int = 0,
        dropout: float = 0.0,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Args:
            in_features: Input node feature dimension
            out_features: Output node feature dimension
            edge_features: Edge feature dimension (optional)
            dropout: Dropout rate for attention weights
            alpha: LeakyReLU negative slope
            concat: If True, apply activation (for intermediate layers)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Linear transformation for node features
        self.W = nn.Linear(in_features, out_features, bias=False)

        # Edge feature transformation (if edges have features)
        if edge_features > 0:
            self.W_edge = nn.Linear(edge_features, out_features, bias=False)

        # Attention mechanism parameters
        # a = [a_src || a_dst] for computing attention scores
        self.a_src = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a_dst = nn.Parameter(torch.zeros(size=(out_features, 1)))

        if edge_features > 0:
            self.a_edge = nn.Parameter(torch.zeros(size=(out_features, 1)))

        # Initialization
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        if edge_features > 0:
            nn.init.xavier_uniform_(self.W_edge.weight, gain=1.414)
            nn.init.xavier_uniform_(self.a_edge, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_features)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge features (num_edges, edge_features)

        Returns:
            Updated node features (num_nodes, out_features)
        """
        # Linear transformation
        h = self.W(x)  # (num_nodes, out_features)

        num_nodes = x.size(0)
        row, col = edge_index  # source and destination nodes

        # Compute attention scores
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j || We_ij])
        attention_src = torch.matmul(h, self.a_src).squeeze(-1)  # (num_nodes,)
        attention_dst = torch.matmul(h, self.a_dst).squeeze(-1)  # (num_nodes,)

        # Attention for each edge: a_src[i] + a_dst[j]
        attention = attention_src[row] + attention_dst[col]  # (num_edges,)

        # Add edge features to attention if available
        if edge_attr is not None and self.edge_features > 0:
            h_edge = self.W_edge(edge_attr)  # (num_edges, out_features)
            attention_edge = torch.matmul(h_edge, self.a_edge).squeeze(-1)  # (num_edges,)
            attention = attention + attention_edge

        attention = self.leakyrelu(attention)

        # Normalize attention coefficients using softmax per node
        attention = self._softmax_per_node(attention, col, num_nodes)

        # Apply dropout to attention weights
        if self.training and self.dropout > 0:
            attention = F.dropout(attention, p=self.dropout, training=True)

        # Aggregate neighbor features weighted by attention
        # h'_i = sum_j (alpha_ij * h_j)
        h_agg = torch.zeros(num_nodes, self.out_features, device=x.device)

        # Multiply each neighbor's feature by its attention weight
        h_weighted = h[row] * attention.unsqueeze(-1)  # (num_edges, out_features)

        # Sum over all incoming edges for each node
        h_agg.index_add_(0, col, h_weighted)

        # Apply activation if this is not the last layer
        if self.concat:
            return F.elu(h_agg)
        else:
            return h_agg

    def _softmax_per_node(
        self,
        attention: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Apply softmax to attention scores per destination node

        Args:
            attention: Attention scores (num_edges,)
            index: Destination node indices (num_edges,)
            num_nodes: Total number of nodes

        Returns:
            Normalized attention weights (num_edges,)
        """
        # Subtract max for numerical stability
        attention_max = torch.zeros(num_nodes, device=attention.device)
        attention_max.index_reduce_(0, index, attention, 'amax', include_self=False)
        attention = attention - attention_max[index]

        # Compute exp
        attention_exp = torch.exp(attention)

        # Sum of exp per node
        attention_sum = torch.zeros(num_nodes, device=attention.device)
        attention_sum.index_add_(0, index, attention_exp)

        # Normalize
        return attention_exp / (attention_sum[index] + 1e-16)


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer

    Applies multiple attention heads in parallel and concatenates or averages results.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_features: int = 0,
        num_heads: int = 8,
        dropout: float = 0.0,
        alpha: float = 0.2,
        concat_heads: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            edge_features: Edge feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            alpha: LeakyReLU negative slope
            concat_heads: If True, concatenate heads; if False, average them
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads

        # Create multiple attention heads
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(
                in_features,
                out_features,
                edge_features,
                dropout,
                alpha,
                concat=True
            )
            for _ in range(num_heads)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features

        Returns:
            Updated node features
        """
        # Apply all attention heads
        head_outputs = [
            head(x, edge_index, edge_attr)
            for head in self.attention_heads
        ]

        if self.concat_heads:
            # Concatenate all heads
            return torch.cat(head_outputs, dim=-1)
        else:
            # Average all heads
            return torch.mean(torch.stack(head_outputs), dim=0)


class GATEncoder(nn.Module):
    """
    GAT Encoder: Transforms input features to hidden representation
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_heads: int = 4
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

    def forward(self, node_features, edge_features):
        return self.node_encoder(node_features), self.edge_encoder(edge_features)


class GATProcessor(nn.Module):
    """
    GAT Processor: Stack of multi-head attention layers
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Intermediate layers (concatenate heads)
        self.layers = nn.ModuleList([
            MultiHeadGATLayer(
                in_features=hidden_dim * num_heads if i > 0 else hidden_dim,
                out_features=hidden_dim,
                edge_features=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                concat_heads=True
            )
            for i in range(num_layers - 1)
        ])

        # Final layer (average heads)
        self.final_layer = MultiHeadGATLayer(
            in_features=hidden_dim * num_heads,
            out_features=hidden_dim,
            edge_features=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat_heads=False
        )

    def forward(self, x, edge_index, edge_attr):
        # Process through intermediate layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Final layer
        x = self.final_layer(x, edge_index, edge_attr)
        return x


class GATDecoder(nn.Module):
    """
    GAT Decoder: Maps hidden features to output predictions
    """

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class GraphAttentionNet(nn.Module):
    """
    Complete Graph Attention Network for particle-based fluid simulation

    Architecture:
    - Encoder: Transform node/edge features to hidden space
    - Processor: Stack of multi-head attention layers
    - Decoder: Map to output predictions
    """

    def __init__(
        self,
        input_node_dim: int = 7,      # pos(3) + vel(3) + type(1)
        input_edge_dim: int = 3,       # relative position
        hidden_dim: int = 64,
        output_dim: int = 3,           # velocity prediction
        num_layers: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            input_node_dim: Dimension of input node features
            input_edge_dim: Dimension of input edge features
            hidden_dim: Hidden dimension for attention layers
            output_dim: Output dimension (velocity prediction)
            num_layers: Number of attention layers
            num_heads: Number of attention heads per layer
            dropout: Dropout rate for attention weights
        """
        super().__init__()

        self.input_node_dim = input_node_dim
        self.input_edge_dim = input_edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

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

        # Processor: Multi-head GAT layers
        self.gat_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer
                in_dim = hidden_dim
            else:
                # Subsequent layers (heads are concatenated)
                in_dim = hidden_dim * num_heads

            # Last layer averages heads, others concatenate
            concat = (i < num_layers - 1)

            self.gat_layers.append(
                MultiHeadGATLayer(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    edge_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat_heads=concat
                )
            )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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

        # Process through GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)

        # Decode
        out = self.decoder(x)

        return out

    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test GAT model
    print("="*70)
    print("GRAPH ATTENTION NETWORK (GAT) TEST")
    print("="*70)

    # Create model
    model = GraphAttentionNet(
        input_node_dim=7,
        input_edge_dim=3,
        hidden_dim=64,
        output_dim=3,
        num_layers=5,
        num_heads=4,
        dropout=0.1
    )

    print(f"\nModel Architecture:")
    print(f"  Input node dim: 7 (pos + vel + type)")
    print(f"  Input edge dim: 3 (relative position)")
    print(f"  Hidden dim: 64")
    print(f"  Output dim: 3 (velocity)")
    print(f"  Num layers: 5")
    print(f"  Num attention heads: 4")
    print(f"  Total parameters: {model.count_parameters():,}")

    # Test forward pass
    num_nodes = 100
    num_edges = 500

    node_features = torch.randn(num_nodes, 7)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_features = torch.randn(num_edges, 3)

    print(f"\nTest Forward Pass:")
    print(f"  Input nodes: {node_features.shape}")
    print(f"  Edge index: {edge_index.shape}")
    print(f"  Edge features: {edge_features.shape}")

    # Forward pass
    output = model(node_features, edge_index, edge_features)

    print(f"  Output: {output.shape}")
    print(f"\n{'='*70}")
    print("TEST PASSED!")
    print("="*70)
