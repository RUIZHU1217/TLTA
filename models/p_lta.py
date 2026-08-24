from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou, nms, roi_align

from .cdn_fd import ContrastiveDeNoisingFeatureDecoding
from .hypergraph import HyperGraphConvolutionBlock, Hypergraph, pad_hypergraphs


def _xyxy_to_cxcywh_pixels(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)


def _cxcywh_to_xyxy_pixels(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


class AnchorGenerator:
    """Conventional dense anchors required by the RPN-like PPN.

    Anchor scales and ratios are absent from the manuscript and are supplied by
    configuration as implementation details.
    """

    def __init__(self, sizes: list[float], ratios: list[float]) -> None:
        self.sizes = sizes
        self.ratios = ratios
        self.num_anchors = len(sizes) * len(ratios)

    def __call__(
        self,
        feature_size: tuple[int, int],
        image_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        feature_h, feature_w = feature_size
        image_h, image_w = image_size
        stride_y, stride_x = image_h / feature_h, image_w / feature_w
        yy, xx = torch.meshgrid(
            (torch.arange(feature_h, device=device, dtype=dtype) + 0.5) * stride_y,
            (torch.arange(feature_w, device=device, dtype=dtype) + 0.5) * stride_x,
            indexing="ij",
        )
        centers = torch.stack((xx, yy, xx, yy), dim=-1).reshape(-1, 1, 4)
        base = []
        for size in self.sizes:
            for ratio in self.ratios:
                width = size / math.sqrt(ratio)
                height = size * math.sqrt(ratio)
                base.append([-width / 2, -height / 2, width / 2, height / 2])
        base_anchors = torch.tensor(base, device=device, dtype=dtype).view(1, -1, 4)
        return (centers + base_anchors).reshape(-1, 4)


def encode_box_deltas(anchors: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    anchors_c = _xyxy_to_cxcywh_pixels(anchors)
    boxes_c = _xyxy_to_cxcywh_pixels(boxes)
    tx = (boxes_c[:, 0] - anchors_c[:, 0]) / anchors_c[:, 2].clamp_min(1e-6)
    ty = (boxes_c[:, 1] - anchors_c[:, 1]) / anchors_c[:, 3].clamp_min(1e-6)
    tw = torch.log(boxes_c[:, 2].clamp_min(1e-6) / anchors_c[:, 2].clamp_min(1e-6))
    th = torch.log(boxes_c[:, 3].clamp_min(1e-6) / anchors_c[:, 3].clamp_min(1e-6))
    return torch.stack((tx, ty, tw, th), dim=-1)


def decode_box_deltas(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    anchors_c = _xyxy_to_cxcywh_pixels(anchors)
    cx = deltas[:, 0] * anchors_c[:, 2] + anchors_c[:, 0]
    cy = deltas[:, 1] * anchors_c[:, 3] + anchors_c[:, 1]
    width = deltas[:, 2].clamp(max=math.log(1000.0 / 16)).exp() * anchors_c[:, 2]
    height = deltas[:, 3].clamp(max=math.log(1000.0 / 16)).exp() * anchors_c[:, 3]
    return _cxcywh_to_xyxy_pixels(torch.stack((cx, cy, width, height), dim=-1))


def clip_boxes(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    return boxes


class ProposalPredictionNetwork(nn.Module):
    """PPN from Section 3.6.1: 3x3 reduction, REG/CLS branches and NMS."""

    def __init__(self, in_channels: int, config: dict) -> None:
        super().__init__()
        # TODO: Not explicitly specified in the paper
        hidden_channels = max(32, in_channels // 2)
        self.reduction = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.anchor_generator = AnchorGenerator(list(config["anchor_sizes"]), list(config["anchor_ratios"]))
        anchors_per_location = self.anchor_generator.num_anchors
        self.regression = nn.Conv2d(hidden_channels, anchors_per_location * 4, 1)
        # The manuscript explicitly describes an FC classification pathway.
        self.classification = nn.Linear(hidden_channels, anchors_per_location * 2)
        self.pre_nms_topk = int(config["proposal_pre_nms_topk"])
        self.post_nms_topk = int(config["proposal_post_nms_topk"])
        self.nms_threshold = float(config["proposal_nms_threshold"])
        self.iou_threshold = float(config["proposal_iou_threshold"])
        if self.iou_threshold != 0.5:
            raise ValueError("Algorithm 1 explicitly uses proposal IoU threshold 0.5")

    def _losses(
        self,
        logits: torch.Tensor,
        deltas: torch.Tensor,
        anchors: torch.Tensor,
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        cls_losses = []
        reg_losses = []
        for batch_index, target in enumerate(targets):
            gt_boxes = target["boxes"]
            if gt_boxes.numel() == 0:
                cls_losses.append(F.cross_entropy(logits[batch_index], torch.zeros(len(anchors), dtype=torch.long, device=anchors.device)))
                reg_losses.append(deltas[batch_index].sum() * 0.0)
                continue
            ious = box_iou(anchors, gt_boxes)
            best_iou, matched_index = ious.max(dim=1)
            labels = (best_iou >= self.iou_threshold).long()
            # TODO: Not explicitly specified in the paper
            # CE and Smooth-L1 are conventional RPN losses; PPN loss details and
            # anchor subsampling are not reported.
            cls_losses.append(F.cross_entropy(logits[batch_index], labels))
            positive = labels.bool()
            if positive.any():
                encoded = encode_box_deltas(anchors[positive], gt_boxes[matched_index[positive]])
                reg_losses.append(F.smooth_l1_loss(deltas[batch_index, positive], encoded))
            else:
                reg_losses.append(deltas[batch_index].sum() * 0.0)
        return {"loss_ppn_cls": torch.stack(cls_losses).mean(), "loss_ppn_reg": torch.stack(reg_losses).mean()}

    def forward(
        self,
        feature_map: torch.Tensor,
        image_size: tuple[int, int],
        targets: list[dict] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, torch.Tensor]]:
        """Return proposal boxes/scores lists and PPN training losses.

        Input is ``F4 [B,C,Hf,Wf]``. Each proposal tensor is ``[P,4]`` in
        resized-image xyxy coordinates.
        """
        hidden = F.relu(self.reduction(feature_map))
        batch, channels, height, width = hidden.shape
        deltas = self.regression(hidden)
        deltas = deltas.view(batch, self.anchor_generator.num_anchors, 4, height, width)
        deltas = deltas.permute(0, 3, 4, 1, 2).reshape(batch, -1, 4)
        flattened = hidden.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        logits = self.classification(flattened).view(batch, height * width, self.anchor_generator.num_anchors, 2)
        logits = logits.reshape(batch, -1, 2)
        anchors = self.anchor_generator((height, width), image_size, feature_map.device, feature_map.dtype)

        proposals: list[torch.Tensor] = []
        proposal_scores: list[torch.Tensor] = []
        foreground_scores = logits.softmax(dim=-1)[..., 1]
        for batch_index in range(batch):
            topk = min(self.pre_nms_topk, len(anchors))
            scores, indices = foreground_scores[batch_index].topk(topk)
            boxes = decode_box_deltas(anchors[indices], deltas[batch_index, indices])
            boxes = clip_boxes(boxes, image_size)
            valid = ((boxes[:, 2] - boxes[:, 0]) > 1) & ((boxes[:, 3] - boxes[:, 1]) > 1)
            boxes, scores = boxes[valid], scores[valid]
            keep = nms(boxes, scores, self.nms_threshold)[: self.post_nms_topk]
            proposals.append(boxes[keep])
            proposal_scores.append(scores[keep])

        losses: dict[str, torch.Tensor] = {}
        if self.training and targets is not None:
            losses = self._losses(logits, deltas, anchors, targets)
        return proposals, proposal_scores, losses


class ProposalSampler(nn.Module):
    """Generate P+ and P- according to Algorithm 1 line 15."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        super().__init__()
        if iou_threshold != 0.5:
            raise ValueError("The manuscript explicitly defines the split at IoU 0.5")
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def forward(
        self,
        proposals: list[torch.Tensor],
        scores: list[torch.Tensor],
        targets: list[dict] | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        positive, negative = [], []
        for index, boxes in enumerate(proposals):
            if targets is not None:
                gt = targets[index]["boxes"]
                if gt.numel():
                    best_iou = box_iou(boxes, gt).max(dim=1).values
                    positive_mask = best_iou >= self.iou_threshold
                else:
                    positive_mask = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
            else:
                # GT is unavailable at inference. Objectness is the minimal
                # conventional surrogate for the paper's GT-based split.
                # TODO: Not explicitly specified in the paper
                positive_mask = scores[index] >= 0.5
            positive.append(boxes[positive_mask])
            negative.append(boxes[~positive_mask])
        return positive, negative


class ProposalFeatureProjector(nn.Module):
    """RoIAlign proposal feature subsets and project each subset to one vertex."""

    def __init__(self, channels: int, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size
        # TODO: Not explicitly specified in the paper
        self.projection = nn.Linear(channels * output_size * output_size, channels)

    def forward(
        self,
        feature_map: torch.Tensor,
        proposal_lists: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> list[torch.Tensor]:
        image_h, image_w = image_size
        feature_h, feature_w = feature_map.shape[-2:]
        outputs = []
        for index, boxes in enumerate(proposal_lists):
            if boxes.numel() == 0:
                outputs.append(feature_map.new_zeros((0, feature_map.shape[1])))
                continue
            scaled = boxes.clone()
            scaled[:, [0, 2]] *= feature_w / image_w
            scaled[:, [1, 3]] *= feature_h / image_h
            regions = roi_align(
                feature_map[index : index + 1],
                [scaled],
                output_size=self.output_size,
                spatial_scale=1.0,
                aligned=True,
            )
            outputs.append(self.projection(regions.flatten(1)))
        return outputs


class ProposalHyperGraphConstruction(nn.Module):
    """Base PG-HGC implementation of Equations (32)-(34)."""

    def __init__(self, channels: int, k: int) -> None:
        super().__init__()
        self.projection = nn.Linear(channels, channels, bias=False)
        self.k = k
        self.channels = channels

    def _single(self, features: torch.Tensor) -> Hypergraph:
        count = features.shape[0]
        if count == 0:
            dummy_features = features.new_zeros(1, self.channels)
            incidence = features.new_ones(1, 1)
            mask = torch.zeros(1, 1, dtype=torch.bool, device=features.device)
            return Hypergraph(dummy_features.unsqueeze(0), incidence.unsqueeze(0), features.new_ones(1, 1), mask)
        if count == 1:
            incidence = features.new_ones(1, 1)
            return Hypergraph(features.unsqueeze(0), incidence.unsqueeze(0), features.new_ones(1, 1))

        projected = F.normalize(self.projection(features), dim=-1)
        similarity = projected @ projected.transpose(0, 1)  # Equation (32)
        similarity.fill_diagonal_(-torch.inf)
        k = min(self.k, count - 1)
        neighbors = similarity.topk(k, dim=1).indices  # Equation (33)
        incidence = features.new_zeros(count, count)
        for center in range(count):
            incidence[neighbors[center], center] = 1.0

        gamma = 1.0 / self.channels  # paper default in Section 3.6.2
        weights = []
        for edge in range(count):
            edge_features = features[incidence[:, edge].bool()]
            pairwise_squared = torch.cdist(edge_features, edge_features).square()
            weights.append(torch.exp(-gamma * pairwise_squared).sum() / max(len(edge_features), 1))
        edge_weights = torch.stack(weights).unsqueeze(0)  # Equation (34)
        return Hypergraph(features.unsqueeze(0), incidence.unsqueeze(0), edge_weights)

    def forward(self, feature_lists: list[torch.Tensor]) -> Hypergraph:
        return pad_hypergraphs([self._single(features) for features in feature_lists])


class PositiveProposalHyperGraphConstruction(ProposalHyperGraphConstruction):
    """PP-HGC for positive proposal vertices."""


class NegativeProposalHyperGraphConstruction(ProposalHyperGraphConstruction):
    """NP-HGC for negative proposal vertices."""


class ProposalGuidedHyperGraphConstruction(nn.Module):
    """PG-HGC wrapper that constructs positive and negative pathways in parallel."""

    def __init__(self, channels: int, k: int) -> None:
        super().__init__()
        self.pp_hgc = PositiveProposalHyperGraphConstruction(channels, k)
        self.np_hgc = NegativeProposalHyperGraphConstruction(channels, k)

    def forward(
        self,
        positive_features: list[torch.Tensor],
        negative_features: list[torch.Tensor],
    ) -> tuple[Hypergraph, Hypergraph]:
        return self.pp_hgc(positive_features), self.np_hgc(negative_features)


class SequencePositionEmbedding(nn.Module):
    """Conventional sine position encoding for the topology feature sequence."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, channels = x.shape
        position = torch.arange(length, device=x.device, dtype=x.dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, channels, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / channels))
        embedding = torch.zeros(length, channels, device=x.device, dtype=x.dtype)
        embedding[:, 0::2] = torch.sin(position * div)
        embedding[:, 1::2] = torch.cos(position * div[: embedding[:, 1::2].shape[1]])
        return embedding.unsqueeze(0)


class TopologyEncoder(nn.Module):
    """Encoder for F_ED in Algorithm 1, lines 20-21."""

    def __init__(self, channels: int, config: dict) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            channels,
            int(config["transformer_heads"]),
            int(config["ff_dim"]),
            float(config["dropout"]),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, int(config["encoder_layers"]))
        self.position = SequencePositionEmbedding()

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(sequence + self.position(sequence), src_key_padding_mask=~mask)


class ProposalLevelTopologyAwareness(nn.Module):
    """Complete p-LTA including PPN, PP/NP-HGC, encoder and CDN-FD decoder."""

    def __init__(self, in_channels: int, num_classes: int, model_config: dict, cdn_config: dict) -> None:
        super().__init__()
        self.num_queries = int(model_config["num_queries"])
        self.ppn = ProposalPredictionNetwork(in_channels, model_config)
        self.sampler = ProposalSampler(float(model_config["proposal_iou_threshold"]))
        self.roi_projector = ProposalFeatureProjector(in_channels, int(model_config["roi_output_size"]))
        self.pg_hgc = ProposalGuidedHyperGraphConstruction(
            in_channels, int(model_config["proposal_hypergraph_k"])
        )
        self.positive_hgcb = HyperGraphConvolutionBlock(in_channels, in_channels)
        self.negative_hgcb = HyperGraphConvolutionBlock(in_channels, in_channels)
        self.encoder = TopologyEncoder(in_channels, model_config)
        self.cdn_fd = ContrastiveDeNoisingFeatureDecoding(
            in_channels, num_classes, cdn_config, model_config
        )

    def _pack_sequence(
        self,
        positive: torch.Tensor,
        positive_mask: torch.Tensor,
        negative: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels = positive.shape
        sequence = positive.new_zeros(batch, self.num_queries, channels)
        mask = torch.zeros(batch, self.num_queries, device=positive.device, dtype=torch.bool)
        for index in range(batch):
            values = torch.cat((positive[index, positive_mask[index]], negative[index, negative_mask[index]]), dim=0)
            # TODO: Not explicitly specified in the paper
            values = values[: self.num_queries]
            count = len(values)
            if count:
                sequence[index, :count] = values
                mask[index, :count] = True
            else:
                mask[index, 0] = True
        return sequence, mask

    def forward(
        self,
        feature_map: torch.Tensor,
        image_size: tuple[int, int],
        targets: list[dict] | None = None,
    ) -> dict:
        proposals, scores, ppn_losses = self.ppn(feature_map, image_size, targets)
        positive_boxes, negative_boxes = self.sampler(proposals, scores, targets)
        positive_features = self.roi_projector(feature_map, positive_boxes, image_size)
        negative_features = self.roi_projector(feature_map, negative_boxes, image_size)
        positive_graph, negative_graph = self.pg_hgc(positive_features, negative_features)

        positive_output = self.positive_hgcb(positive_graph)
        negative_output = self.negative_hgcb(negative_graph)
        sequence, sequence_mask = self._pack_sequence(
            positive_output,
            positive_graph.vertex_mask,
            negative_output,
            negative_graph.vertex_mask,
        )
        memory = self.encoder(sequence, sequence_mask)
        decoded = self.cdn_fd(memory, sequence_mask, targets)
        decoded["proposal_boxes"] = proposals
        decoded["positive_proposals"] = positive_boxes
        decoded["negative_proposals"] = negative_boxes
        if self.training and targets is not None:
            decoded.setdefault("losses", {}).update(ppn_losses)
        return decoded

