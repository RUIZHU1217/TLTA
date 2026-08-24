from __future__ import annotations

import math

import numpy as np
import torch
from skimage import color
from skimage.segmentation import slic
from torch import nn
from torch.nn import functional as F

from .cacn import CrossAttentionCollaborativeNetwork
from .hypergraph import HyperGraphConvolutionBlock, Hypergraph, pad_hypergraphs


class SuperPixelSegmentationModule(nn.Module):
    """SPSM following Algorithm 2 and Equations (18)-(19).

    SLIC implements the stated CIELAB/spatial iterative clustering, movement to
    a low-gradient center and connectivity enforcement.  Segmentation is a
    non-differentiable preprocessing operation; tensors are therefore detached
    for this module.
    """

    def __init__(self, num_superpixels: int, compactness: float, max_iter: int) -> None:
        super().__init__()
        self.num_superpixels = num_superpixels
        self.compactness = compactness
        self.max_iter = max_iter

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Segment ``[B,3,H,W]`` images; return contiguous labels ``[B,H,W]``."""
        labels = []
        for image in images.detach().cpu():
            array = image.permute(1, 2, 0).clamp(0, 1).numpy()
            # TODO: Not explicitly specified in the paper
            # scikit-image supplies the conventional connectivity postprocess
            # from Algorithm 2; the exact convergence tolerance is not reported.
            label = slic(
                array,
                n_segments=self.num_superpixels,
                compactness=self.compactness,
                max_num_iter=self.max_iter,
                convert2lab=True,
                enforce_connectivity=True,
                start_label=0,
                channel_axis=-1,
            )
            _, label = np.unique(label, return_inverse=True)
            labels.append(torch.from_numpy(label.reshape(array.shape[:2])).long())
        return torch.stack(labels).to(images.device)


def _superpixel_statistics(image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return per-superpixel mean Lab and normalized centroid features ``[N,5]``."""
    height, width = labels.shape
    count = int(labels.max().item()) + 1
    lab = torch.from_numpy(
        color.rgb2lab(image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()).astype(np.float32)
    ).to(image.device)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height, device=image.device),
        torch.linspace(0, 1, width, device=image.device),
        indexing="ij",
    )
    pixels = torch.cat((lab, xx.unsqueeze(-1), yy.unsqueeze(-1)), dim=-1).reshape(-1, 5)
    flat_labels = labels.reshape(-1)
    sums = pixels.new_zeros(count, 5).index_add_(0, flat_labels, pixels)
    counts = pixels.new_zeros(count).index_add_(0, flat_labels, torch.ones_like(flat_labels, dtype=pixels.dtype))
    return sums / counts.clamp_min(1.0).unsqueeze(-1)


class SuperPixelHyperGraphConstruction(nn.Module):
    """SP-HGC using multi-bandwidth agglomerative mean-shift, Equations (20)-(21)."""

    def __init__(self, out_channels: int, bandwidths: list[float], max_iter: int = 10) -> None:
        super().__init__()
        self.bandwidths = bandwidths
        # TODO: Not explicitly specified in the paper
        self.max_iter = max_iter
        self.vertex_projection = nn.Linear(5, out_channels)

    def _mean_shift_incidence(self, features: torch.Tensor, bandwidth: float) -> torch.Tensor:
        normalized = (features - features.mean(0, keepdim=True)) / features.std(
            0, keepdim=True, unbiased=False
        ).clamp_min(1e-5)
        queries = normalized.clone()
        for _ in range(self.max_iter):
            distances = torch.cdist(queries, normalized).square()
            weights = torch.exp(-distances / (2.0 * bandwidth * bandwidth))
            updated = weights @ normalized / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            if torch.linalg.vector_norm(updated - queries, dim=1).max() <= 1e-3:
                queries = updated
                break
            queries = updated

        # Iterative query-set compression is stated but not numerically defined.
        # TODO: Not explicitly specified in the paper
        modes: list[torch.Tensor] = []
        assignments: list[int] = []
        merge_radius = bandwidth * 0.5
        for query in queries:
            if not modes:
                modes.append(query)
                assignments.append(0)
                continue
            mode_tensor = torch.stack(modes)
            distance, index = torch.linalg.vector_norm(mode_tensor - query, dim=1).min(dim=0)
            if distance <= merge_radius:
                assignments.append(int(index.item()))
            else:
                modes.append(query)
                assignments.append(len(modes) - 1)
        assignment = torch.tensor(assignments, device=features.device)
        incidence = F.one_hot(assignment, num_classes=len(modes)).to(features.dtype)
        return incidence

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> Hypergraph:
        """Build a padded graph from images ``[B,3,H,W]`` and labels ``[B,H,W]``."""
        graphs = []
        for image, label in zip(images, labels):
            raw_vertices = _superpixel_statistics(image, label)
            incidence_parts = [
                self._mean_shift_incidence(raw_vertices, float(bandwidth)) for bandwidth in self.bandwidths
            ]
            incidence = torch.cat(incidence_parts, dim=1)
            projected = self.vertex_projection(raw_vertices)
            graphs.append(
                Hypergraph(
                    projected.unsqueeze(0),
                    incidence.unsqueeze(0),
                    torch.ones(1, incidence.shape[1], device=image.device, dtype=projected.dtype),
                )
            )
        return pad_hypergraphs(graphs)


class SuperPixelFeaturePropagation(nn.Module):
    """Propagate HGCB super-pixel vertices back to their pixel regions."""

    def forward(
        self,
        vertex_features: torch.Tensor,
        labels: torch.Tensor,
        vertex_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Map ``[B,N,C]`` vertices to ``[B,C,H,W]`` feature maps."""
        maps = []
        for features, label, mask in zip(vertex_features, labels, vertex_mask):
            valid = features[mask]
            pixel_features = valid[label]
            maps.append(pixel_features.permute(2, 0, 1))
        return torch.stack(maps)


class DenseContextualFeatureExtraction(nn.Module):
    """DCFE dense contextual expansion with dilation rates 3, 6 and 12, (22)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.d3 = nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3, bias=False)
        self.d6 = nn.Conv2d(out_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.d12 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=12, dilation=12, bias=False)
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Map ``[B,Cin,H,W]`` to ``[B,Cout,H,W]``."""
        first = F.relu(self.d3(image))
        second_new = F.relu(self.d6(first))
        second = torch.cat((first, second_new), dim=1)
        third_new = F.relu(self.d12(second))
        third = torch.cat((second, third_new), dim=1)
        return F.relu(self.norm(self.fuse(third)))


class InputLevelTopologyAwareness(nn.Module):
    """Complete i-LTA path in Figure 6 and Algorithm 1, lines 1-5."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        channels = int(config["i_lta_channels"])
        self.spsm = SuperPixelSegmentationModule(
            int(config["num_superpixels"]),
            float(config["superpixel_compactness"]),
            int(config["superpixel_max_iter"]),
        )
        self.sp_hgc = SuperPixelHyperGraphConstruction(channels, list(config["sp_bandwidths"]))
        self.hgcb = HyperGraphConvolutionBlock(channels, channels)
        self.propagate = SuperPixelFeaturePropagation()
        self.dcfe = DenseContextualFeatureExtraction(int(config["in_channels"]), channels)
        self.cacn = CrossAttentionCollaborativeNetwork(channels, num_blocks=5)
        self.out_channels = channels * 2

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Map SAR images ``[B,3,H,W]`` to ``F_CACN [B,2C,H,W]``."""
        labels = self.spsm(images)
        graph = self.sp_hgc(images, labels)
        sp_vertices = self.hgcb(graph)
        sp_features = self.propagate(sp_vertices, labels, graph.vertex_mask)
        pixel_features = self.dcfe(images)
        output = self.cacn(pixel_features, sp_features)
        return output, {"superpixel_labels": labels, "superpixel_incidence": graph.incidence}
