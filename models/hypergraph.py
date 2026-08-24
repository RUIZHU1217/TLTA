from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Hypergraph:
    """Dense batched hypergraph.

    Attributes:
        vertex_features: ``[B, N, C]`` vertex feature tensor X.
        incidence: ``[B, N, M]`` incidence matrix H (binary or continuous).
        edge_weights: ``[B, M]`` diagonal values of W.
        vertex_mask: optional ``[B, N]`` mask for valid padded vertices.
    """

    vertex_features: torch.Tensor
    incidence: torch.Tensor
    edge_weights: torch.Tensor | None = None
    vertex_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.vertex_features.ndim != 3:
            raise ValueError("vertex_features must be [B, N, C]")
        if self.incidence.ndim != 3:
            raise ValueError("incidence must be [B, N, M]")
        if self.vertex_features.shape[:2] != self.incidence.shape[:2]:
            raise ValueError("vertex and incidence batch/vertex dimensions differ")
        if self.edge_weights is None:
            self.edge_weights = self.incidence.new_ones(self.incidence.shape[0], self.incidence.shape[2])
        if self.vertex_mask is None:
            self.vertex_mask = torch.ones(
                self.vertex_features.shape[:2], dtype=torch.bool, device=self.vertex_features.device
            )

    def replace_features(self, features: torch.Tensor) -> "Hypergraph":
        return Hypergraph(features, self.incidence, self.edge_weights, self.vertex_mask)


def normalized_hypergraph_propagation(
    incidence: torch.Tensor,
    edge_weights: torch.Tensor,
    symmetric_vertex_norm: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the propagation matrix in Equations (9) or (16).

    Args:
        incidence: ``[B, N, M]`` H.
        edge_weights: ``[B, M]`` diagonal entries of W.
        symmetric_vertex_norm: use ``Dv^-1/2`` on both sides for spectral
            convolution; otherwise use ``Dv^-1`` on the left for spatial
            convolution.
    Returns:
        Dense propagation matrix ``[B, N, N]``.
    """

    edge_degree = incidence.sum(dim=1).clamp_min(eps)  # [B, M]
    vertex_degree = (incidence * edge_weights[:, None, :]).sum(dim=2).clamp_min(eps)  # [B, N]
    weighted_incidence = incidence * (edge_weights / edge_degree)[:, None, :]
    core = torch.bmm(weighted_incidence, incidence.transpose(1, 2))
    if symmetric_vertex_norm:
        norm = vertex_degree.rsqrt()
        return norm[:, :, None] * core * norm[:, None, :]
    return vertex_degree.reciprocal()[:, :, None] * core


def propagate_vertex_features(
    features: torch.Tensor,
    incidence: torch.Tensor,
    edge_weights: torch.Tensor,
    symmetric_vertex_norm: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply normalized propagation without materializing an ``N x N`` matrix."""
    edge_degree = incidence.sum(dim=1).clamp_min(eps)
    vertex_degree = (incidence * edge_weights[:, None, :]).sum(dim=2).clamp_min(eps)
    if symmetric_vertex_norm:
        norm = vertex_degree.rsqrt()
        features = features * norm.unsqueeze(-1)
    edge_features = torch.bmm(incidence.transpose(1, 2), features)
    edge_features = edge_features * (edge_weights / edge_degree).unsqueeze(-1)
    output = torch.bmm(incidence, edge_features)
    if symmetric_vertex_norm:
        return output * vertex_degree.rsqrt().unsqueeze(-1)
    return output * vertex_degree.reciprocal().unsqueeze(-1)


class SpectralHypergraphConvolution(nn.Module):
    """First-order Chebyshev spectral hypergraph convolution, Equations (5)-(9).

    The paper fixes ``K=1`` and ``lambda_max=2`` and simplifies the convolution
    to ``Dv^-1/2 H W De^-1 H^T Dv^-1/2 X Theta``.
    """

    chebyshev_order = 1
    lambda_max = 2.0

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.theta = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, graph: Hypergraph) -> torch.Tensor:
        """Map ``[B,N,Cin]`` vertices to ``[B,N,Cout]`` vertices."""
        output = propagate_vertex_features(
            self.theta(graph.vertex_features), graph.incidence, graph.edge_weights, symmetric_vertex_norm=True
        ) + self.bias
        return F.relu(output) * graph.vertex_mask.unsqueeze(-1)


class SpatialHypergraphConvolution(nn.Module):
    """Two-stage vertex-hyperedge-vertex message passing, Equations (14)-(16)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.theta = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, graph: Hypergraph) -> torch.Tensor:
        """Map ``[B,N,Cin]`` vertices to ``[B,N,Cout]`` vertices."""
        # Equation (16) is the matrix implementation of the hyperpath loop in
        # Equations (14)-(15): vertex -> incident hyperedge -> vertex.
        output = propagate_vertex_features(
            self.theta(graph.vertex_features), graph.incidence, graph.edge_weights, symmetric_vertex_norm=False
        ) + self.bias
        return F.relu(output) * graph.vertex_mask.unsqueeze(-1)


class SqueezeExcitationBlock(nn.Module):
    """Channel balancing SEB used after dual-branch concatenation in Figure 4."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reweight a ``[B,N,C]`` tensor and preserve its shape."""
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = mask.to(x.dtype).unsqueeze(-1)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        scale = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))
        return x * scale.unsqueeze(1)


class HyperGraphConvolutionBlock(nn.Module):
    """HGCB: parallel spectral/spatial convolution, concatenation and SEB.

    This directly implements Equation (3) and Figure 4.  The split of output
    channels between the two branches is an implementation detail because the
    manuscript does not specify branch widths.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # TODO: Not explicitly specified in the paper
        spectral_channels = out_channels // 2
        spatial_channels = out_channels - spectral_channels
        self.spectral = SpectralHypergraphConvolution(in_channels, spectral_channels)
        self.spatial = SpatialHypergraphConvolution(in_channels, spatial_channels)
        self.seb = SqueezeExcitationBlock(out_channels)

    def forward(self, graph: Hypergraph) -> torch.Tensor:
        """Return fused vertices with shape ``[B,N,Cout]``."""
        spectral = self.spectral(graph)
        spatial = self.spatial(graph)
        fused = torch.cat((spectral, spatial), dim=-1)
        return self.seb(fused, graph.vertex_mask) * graph.vertex_mask.unsqueeze(-1)


def pad_hypergraphs(graphs: list[Hypergraph]) -> Hypergraph:
    """Pad single-sample hypergraphs to one dense batch."""
    if not graphs:
        raise ValueError("At least one graph is required")
    max_vertices = max(graph.vertex_features.shape[1] for graph in graphs)
    max_edges = max(graph.incidence.shape[2] for graph in graphs)
    channels = graphs[0].vertex_features.shape[2]
    batch = len(graphs)
    device = graphs[0].vertex_features.device
    dtype = graphs[0].vertex_features.dtype

    vertices = torch.zeros(batch, max_vertices, channels, device=device, dtype=dtype)
    incidence = torch.zeros(batch, max_vertices, max_edges, device=device, dtype=dtype)
    edge_weights = torch.zeros(batch, max_edges, device=device, dtype=dtype)
    vertex_mask = torch.zeros(batch, max_vertices, device=device, dtype=torch.bool)
    for index, graph in enumerate(graphs):
        n = graph.vertex_features.shape[1]
        m = graph.incidence.shape[2]
        vertices[index, :n] = graph.vertex_features[0]
        incidence[index, :n, :m] = graph.incidence[0]
        edge_weights[index, :m] = graph.edge_weights[0]
        vertex_mask[index, :n] = graph.vertex_mask[0]
    return Hypergraph(vertices, incidence, edge_weights, vertex_mask)
