from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .hypergraph import HyperGraphConvolutionBlock, Hypergraph


class SinePositionEmbedding2D(nn.Module):
    """Two-dimensional sine/cosine position embedding from Equation (17)."""

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        if embedding_dim != 256:
            raise ValueError("The manuscript explicitly specifies position dimension d=256")
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional features ``[B,256,H,W]`` for an image tensor."""
        batch, _, height, width = x.shape
        quarter = self.embedding_dim // 4
        omega = torch.arange(quarter, device=x.device, dtype=x.dtype)
        omega = 1.0 / (10000 ** (omega / max(quarter, 1)))
        y = torch.arange(height, device=x.device, dtype=x.dtype)[:, None] * omega[None, :]
        x_pos = torch.arange(width, device=x.device, dtype=x.dtype)[:, None] * omega[None, :]
        y_embed = torch.cat((y.sin(), y.cos()), dim=-1)[:, None, :].expand(height, width, -1)
        x_embed = torch.cat((x_pos.sin(), x_pos.cos()), dim=-1)[None, :, :].expand(height, width, -1)
        embedding = torch.cat((y_embed, x_embed), dim=-1).permute(2, 0, 1)
        return embedding.unsqueeze(0).expand(batch, -1, -1, -1)


def _window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    batch, height, width, channels = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    x = x.view(batch, padded_h // window_size, window_size, padded_w // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)
    return windows, (padded_h, padded_w)


def _window_reverse(
    windows: torch.Tensor,
    window_size: int,
    padded_size: tuple[int, int],
    original_size: tuple[int, int],
    batch: int,
) -> torch.Tensor:
    padded_h, padded_w = padded_size
    channels = windows.shape[-1]
    x = windows.view(batch, padded_h // window_size, padded_w // window_size, window_size, window_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, padded_h, padded_w, channels)
    return x[:, : original_size[0], : original_size[1]]


class WindowSelfAttention(nn.Module):
    """Local multi-head attention used inside the manuscript's SwinBlock."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Attend within windows ``[Bwin, W*W, C]``."""
        batch, tokens, channels = windows.shape
        qkv = self.qkv(windows).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(output)


class SwinBlock(nn.Module):
    """Minimal conventional Swin block used where Figure 6 specifies SwinBlock.

    Window size, head count, MLP ratio, relative-position bias and exact stage
    depth are not provided by the manuscript.  The implementation uses a plain
    window/shifted-window block without relative bias.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int, shifted: bool = False) -> None:
        super().__init__()
        # TODO: Not explicitly specified in the paper
        self.window_size = window_size
        self.shift_size = window_size // 2 if shifted else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        # TODO: Not explicitly specified in the paper
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a feature map ``[B,C,H,W]`` to the same shape."""
        batch, channels, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1)
        shortcut = tokens
        normalized = self.norm1(tokens)
        if self.shift_size:
            normalized = torch.roll(normalized, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows, padded_size = _window_partition(normalized, self.window_size)
        windows = self.attention(windows)
        attended = _window_reverse(windows, self.window_size, padded_size, (height, width), batch)
        if self.shift_size:
            attended = torch.roll(attended, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        tokens = shortcut + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.permute(0, 3, 1, 2)


class FeatureAdaptiveHyperGraphConstruction(nn.Module):
    """FA-HGC with dynamic prototypes and continuous incidence, (25)-(27)."""

    def __init__(self, channels: int, num_hyperedges: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("FA-HGC channels must be divisible by heads")
        self.channels = channels
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.prototype = nn.Parameter(torch.randn(num_hyperedges, channels) * 0.02)
        self.offset = nn.Linear(channels * 2, num_hyperedges * channels)
        self.vertex_projection = nn.Linear(channels, channels)

    def forward(self, feature_map: torch.Tensor) -> Hypergraph:
        """Map ``[B,C,H,W]`` to a continuous-incidence hypergraph.

        Output vertices are ``[B,H*W,C]`` and H is ``[B,H*W,M]``.
        """
        batch, channels, height, width = feature_map.shape
        vertices = feature_map.flatten(2).transpose(1, 2)
        avg = feature_map.mean(dim=(2, 3))
        maximum = feature_map.amax(dim=(2, 3))
        context = torch.cat((avg, maximum), dim=-1)  # Equation (25)
        delta = self.offset(context).view(batch, self.num_hyperedges, channels)
        prototypes = self.prototype.unsqueeze(0) + delta
        queries = self.vertex_projection(vertices)

        head_dim = channels // self.num_heads
        queries = queries.view(batch, height * width, self.num_heads, head_dim)
        prototypes = prototypes.view(batch, self.num_hyperedges, self.num_heads, head_dim)
        similarity = torch.einsum("bnhd,bmhd->bnmh", queries, prototypes) / math.sqrt(head_dim)
        similarity = similarity.mean(dim=-1)  # Equation (26), arithmetic mean across heads
        incidence = similarity.softmax(dim=1)  # Equation (27), normalize over vertices
        edge_weights = torch.ones(batch, self.num_hyperedges, device=feature_map.device, dtype=feature_map.dtype)
        return Hypergraph(vertices, incidence, edge_weights)


class VertexLevelFeatureSelfAttention(nn.Module):
    """VL-FSA vertex-to-hyperedge aggregation from Equations (28)-(29)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(channels, channels, bias=False)
        self.a1 = nn.Linear(channels, 1, bias=False)

    def forward(self, vertices: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        """Return attended hyperedges ``[B,M,C]`` from vertices ``[B,N,C]``."""
        u = F.relu(self.w1(vertices))
        logits = self.a1(u).expand(-1, -1, incidence.shape[2])
        logits = logits + incidence.clamp_min(1e-8).log()
        alpha = logits.softmax(dim=1)
        return F.relu(torch.bmm(alpha.transpose(1, 2), u))


class EdgeLevelFeatureSelfAttention(nn.Module):
    """EL-FSA hyperedge-to-vertex aggregation from Equations (30)-(31)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.w2 = nn.Linear(channels, channels, bias=False)
        self.vertex_projection = nn.Linear(channels, channels, bias=False)
        self.a2 = nn.Linear(channels, 1, bias=False)

    def forward(
        self,
        vertices: torch.Tensor,
        hyperedges: torch.Tensor,
        incidence: torch.Tensor,
    ) -> torch.Tensor:
        """Return attended vertices ``[B,N,C]``."""
        edge_values = self.w2(hyperedges)
        vertex_values = self.vertex_projection(vertices)
        # Equation (31) combines an edge and its focal vertex.  The manuscript
        # does not disambiguate the printed operator/dimensional projection.
        # TODO: Not explicitly specified in the paper
        joint = F.relu(edge_values[:, None, :, :] + vertex_values[:, :, None, :])
        logits = self.a2(joint).squeeze(-1) + incidence.clamp_min(1e-8).log()
        beta = logits.softmax(dim=2)
        return F.relu(torch.einsum("bnm,bmc->bnc", beta, edge_values))


class FeatureStage(nn.Module):
    """One Swin -> FA-HGC -> HGCB -> VL-FSA -> EL-FSA stage."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        depth: int,
        window_size: int,
        num_hyperedges: int,
        fa_heads: int,
    ) -> None:
        super().__init__()
        self.swin_blocks = nn.ModuleList(
            [SwinBlock(channels, num_heads, window_size, shifted=index % 2 == 1) for index in range(depth)]
        )
        self.fa_hgc = FeatureAdaptiveHyperGraphConstruction(channels, num_hyperedges, fa_heads)
        self.hgcb = HyperGraphConvolutionBlock(channels, channels)
        self.vl_fsa = VertexLevelFeatureSelfAttention(channels)
        self.el_fsa = EdgeLevelFeatureSelfAttention(channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stage feature map ``[B,C,H,W]`` and incidence ``[B,HW,M]``."""
        for block in self.swin_blocks:
            x = block(x)
        graph = self.fa_hgc(x)
        hgcb_vertices = self.hgcb(graph)
        hyperedges = self.vl_fsa(hgcb_vertices, graph.incidence)
        vertices = self.el_fsa(hgcb_vertices, hyperedges, graph.incidence)
        batch, _, height, width = x.shape
        output = vertices.transpose(1, 2).reshape(batch, -1, height, width)
        return output, graph.incidence


class FeatureLevelTopologyAwareness(nn.Module):
    """Four-stage f-LTA in Figure 6 and Algorithm 1, lines 6-13."""

    def __init__(self, in_channels: int, config: dict) -> None:
        super().__init__()
        dims = list(config["stage_dims"])
        heads = list(config["stage_heads"])
        depths = list(config["stage_depths"])
        hyperedges = list(config["fa_num_hyperedges"])
        fa_heads = list(config["fa_heads"])
        if not all(len(values) == 4 for values in (dims, heads, depths, hyperedges, fa_heads)):
            raise ValueError("f-LTA requires exactly four stage settings")

        self.position = SinePositionEmbedding2D(int(config["position_dim"]))
        self.position_projection = nn.Conv2d(256, in_channels, 1, bias=False)
        self.patch_embedding = nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4)
        self.stages = nn.ModuleList(
            [
                FeatureStage(
                    dims[index],
                    heads[index],
                    depths[index],
                    int(config["window_size"]),
                    hyperedges[index],
                    fa_heads[index],
                )
                for index in range(4)
            ]
        )
        self.downsamples = nn.ModuleList(
            [nn.Conv2d(dims[index], dims[index + 1], kernel_size=2, stride=2) for index in range(3)]
        )
        self.out_channels = dims[-1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Map ``F_CACN [B,C,H,W]`` to ``F4 [B,8C,H/32,W/32]``."""
        x = x + self.position_projection(self.position(x))
        x = self.patch_embedding(x)
        features = []
        incidences = []
        for index, stage in enumerate(self.stages):
            x, incidence = stage(x)
            features.append(x)
            incidences.append(incidence)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return features[-1], incidences

