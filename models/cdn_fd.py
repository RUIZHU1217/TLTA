from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou, generalized_box_iou

from utils.boxes import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    canonicalize_boxes,
    normalize_xyxy,
)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dimensions[index], dimensions[index + 1]) for index in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < len(self.layers) - 1 else layer(x)
        return x


class DetectionFFN(nn.Module):
    """Final classification and box-regression FFN shown in Figure 6."""

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes + 1)  # index 0 is background
        self.box_regressor = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, hidden: torch.Tensor, anchors_cxcywh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits ``[B,Q,K+1]`` and normalized xyxy boxes ``[B,Q,4]``."""
        logits = self.classifier(hidden)
        boxes_cxcywh = torch.sigmoid(self.box_regressor(hidden) + inverse_sigmoid(anchors_cxcywh))
        boxes = canonicalize_boxes(box_cxcywh_to_xyxy(boxes_cxcywh))
        return logits, boxes


class HungarianMatcher(nn.Module):
    """One-to-one matching used by the detection decoder.

    The manuscript names Hungarian matching but does not provide matching-cost
    coefficients.  Equal classification/L1/GIoU costs are used here.
    """

    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        targets: list[dict],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        probabilities = logits.softmax(-1)
        assignments = []
        for batch_index, target in enumerate(targets):
            target_boxes = normalize_xyxy(target["boxes"], target["size"])
            target_labels = target["labels"]
            if target_boxes.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                assignments.append((empty, empty))
                continue
            # TODO: Not explicitly specified in the paper
            class_cost = -probabilities[batch_index][:, target_labels]
            l1_cost = torch.cdist(boxes[batch_index], target_boxes, p=1)
            giou_cost = -generalized_box_iou(boxes[batch_index], target_boxes)
            cost = (class_cost + l1_cost + giou_cost).detach().cpu().numpy()
            query_indices, target_indices = linear_sum_assignment(cost)
            assignments.append(
                (
                    torch.as_tensor(query_indices, dtype=torch.long, device=logits.device),
                    torch.as_tensor(target_indices, dtype=torch.long, device=logits.device),
                )
            )
        return assignments


class SetDetectionLoss(nn.Module):
    """Conventional set-prediction loss for ordinary (non-CDN) queries.

    The ordinary detector loss weights are not specified in the manuscript;
    equal CE/L1/GIoU weights are the minimal implementation detail.
    """

    def __init__(self, matcher: HungarianMatcher) -> None:
        super().__init__()
        self.matcher = matcher

    def forward(self, logits: torch.Tensor, boxes: torch.Tensor, targets: list[dict]) -> dict[str, torch.Tensor]:
        assignments = self.matcher(logits, boxes, targets)
        class_targets = torch.zeros(logits.shape[:2], dtype=torch.long, device=logits.device)
        predicted_boxes = []
        target_boxes = []
        for batch_index, (query_indices, target_indices) in enumerate(assignments):
            if query_indices.numel() == 0:
                continue
            class_targets[batch_index, query_indices] = targets[batch_index]["labels"][target_indices]
            predicted_boxes.append(boxes[batch_index, query_indices])
            target_boxes.append(
                normalize_xyxy(targets[batch_index]["boxes"][target_indices], targets[batch_index]["size"])
            )
        loss_cls = F.cross_entropy(logits.flatten(0, 1), class_targets.flatten())
        if predicted_boxes:
            predicted = torch.cat(predicted_boxes)
            expected = torch.cat(target_boxes)
            loss_reg = F.l1_loss(predicted, expected)
            loss_giou = (1.0 - torch.diag(generalized_box_iou(predicted, expected))).mean()
        else:
            loss_reg = boxes.sum() * 0.0
            loss_giou = boxes.sum() * 0.0
        return {"loss_det_cls": loss_cls, "loss_det_reg": loss_reg, "loss_det_giou": loss_giou}


@dataclass
class DenoisingMetadata:
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    valid_mask: torch.Tensor
    target_boxes: torch.Tensor
    target_labels: torch.Tensor
    dn_count: int


class CDNFLoss(nn.Module):
    """CDN-FD regression, classification and negative loss in Equation (38)."""

    def __init__(
        self,
        lambda_reg: float = 1.0,
        lambda_cls: float = 2.0,
        lambda_neg: float = 0.5,
        gradient_iou_threshold: float = 0.3,
        negative_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        self.lambda_neg = lambda_neg
        self.gradient_iou_threshold = gradient_iou_threshold
        self.negative_threshold = negative_threshold

    def forward(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        metadata: DenoisingMetadata,
    ) -> dict[str, torch.Tensor]:
        zero = boxes.sum() * 0.0
        positive = metadata.positive_mask & metadata.valid_mask
        negative = metadata.negative_mask & metadata.valid_mask

        if positive.any():
            predicted_positive = boxes[positive]
            expected_positive = metadata.target_boxes[positive]
            pair_iou = torch.diag(box_iou(predicted_positive, expected_positive))
            # Section 3.6.3: renovate gradients only for positives with IoU > 0.3.
            gradient_mask = pair_iou > self.gradient_iou_threshold
            if gradient_mask.any():
                loss_reg = F.l1_loss(predicted_positive[gradient_mask], expected_positive[gradient_mask])
                loss_cls = F.cross_entropy(
                    logits[positive][gradient_mask], metadata.target_labels[positive][gradient_mask]
                )
            else:
                loss_reg = zero
                loss_cls = zero
        else:
            loss_reg = zero
            loss_cls = zero

        if negative.any():
            negative_background_probability = logits[negative].softmax(-1)[:, 0]
            loss_neg = F.relu(self.negative_threshold - negative_background_probability).mean()
        else:
            loss_neg = zero
        return {
            "loss_cdn_reg": self.lambda_reg * loss_reg,
            "loss_cdn_cls": self.lambda_cls * loss_cls,
            "loss_cdn_neg": self.lambda_neg * loss_neg,
        }


class ContrastiveDeNoisingFeatureDecoding(nn.Module):
    """CDN-FD decoder with Equations (35)-(38) and Figure 9.

    During training, each GT box yields a positive perturbation sampled from
    U(0, 0.1) and a negative perturbation from U(0.1, 0.7).  Denoising groups
    are isolated with an attention mask.  Inference uses only ordinary queries.
    """

    def __init__(self, hidden_dim: int, num_classes: int, config: dict, model_config: dict) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = int(model_config["num_queries"])
        self.lambda_1 = float(config["lambda_1"])
        self.lambda_2 = float(config["lambda_2"])
        self.denoising_groups = int(config["denoising_groups"])
        if (self.lambda_1, self.lambda_2) != (0.1, 0.7):
            raise ValueError("The manuscript explicitly sets CDN-FD thresholds lambda_1=0.1, lambda_2=0.7")

        layer = nn.TransformerDecoderLayer(
            hidden_dim,
            int(model_config["transformer_heads"]),
            int(model_config["ff_dim"]),
            float(model_config["dropout"]),
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, int(model_config["decoder_layers"]))
        self.query_embedding = nn.Embedding(self.num_queries, hidden_dim)
        # Learned anchors are displayed in Figure 9; count/initialization are absent.
        # TODO: Not explicitly specified in the paper
        self.learned_anchors = nn.Embedding(self.num_queries, 4)
        nn.init.zeros_(self.learned_anchors.weight)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.box_embedding = MLP(4, hidden_dim, hidden_dim, 2)
        self.ffn = DetectionFFN(hidden_dim, num_classes)
        self.matcher = HungarianMatcher()
        self.detector_loss = SetDetectionLoss(self.matcher)
        self.cdn_loss = CDNFLoss(
            float(config["lambda_reg"]),
            float(config["lambda_cls"]),
            float(config["lambda_neg"]),
            float(config["gradient_iou_threshold"]),
            float(config["negative_threshold"]),
        )

    def _build_denoising_queries(
        self,
        targets: list[dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DenoisingMetadata]:
        batch = len(targets)
        max_gt = max((len(target["boxes"]) for target in targets), default=0)
        dn_count = max_gt * 2 * self.denoising_groups
        if dn_count == 0:
            empty_query = torch.zeros(batch, 0, self.hidden_dim, device=device, dtype=dtype)
            empty_mask = torch.zeros(batch, 0, device=device, dtype=torch.bool)
            metadata = DenoisingMetadata(
                empty_mask, empty_mask, empty_mask, torch.zeros(batch, 0, 4, device=device, dtype=dtype),
                torch.zeros(batch, 0, device=device, dtype=torch.long), 0
            )
            return empty_query, torch.zeros(batch, 0, 4, device=device, dtype=dtype), empty_mask, metadata

        boxes = torch.zeros(batch, dn_count, 4, device=device, dtype=dtype)
        target_boxes = torch.zeros_like(boxes)
        labels = torch.zeros(batch, dn_count, device=device, dtype=torch.long)
        positive_mask = torch.zeros(batch, dn_count, device=device, dtype=torch.bool)
        negative_mask = torch.zeros_like(positive_mask)
        valid_mask = torch.zeros_like(positive_mask)

        for batch_index, target in enumerate(targets):
            gt_boxes = normalize_xyxy(target["boxes"], target["size"]).to(dtype)
            gt_labels = target["labels"]
            count = gt_boxes.shape[0]
            for group in range(self.denoising_groups):
                offset = group * max_gt * 2
                positive_slice = slice(offset, offset + count)
                negative_slice = slice(offset + max_gt, offset + max_gt + count)
                positive_noise = torch.empty_like(gt_boxes).uniform_(0.0, self.lambda_1)
                negative_noise = torch.empty_like(gt_boxes).uniform_(self.lambda_1, self.lambda_2)
                # Equations (35)-(36). Canonicalization is an implementation
                # detail needed to keep perturbed coordinates as valid boxes.
                boxes[batch_index, positive_slice] = canonicalize_boxes(gt_boxes + positive_noise)
                boxes[batch_index, negative_slice] = canonicalize_boxes(gt_boxes + negative_noise)
                target_boxes[batch_index, positive_slice] = gt_boxes
                target_boxes[batch_index, negative_slice] = gt_boxes
                labels[batch_index, positive_slice] = gt_labels
                labels[batch_index, negative_slice] = 0
                positive_mask[batch_index, positive_slice] = True
                negative_mask[batch_index, negative_slice] = True
                valid_mask[batch_index, positive_slice] = True
                valid_mask[batch_index, negative_slice] = True

        query = self.label_embedding(labels) + self.box_embedding(boxes)
        anchors = box_xyxy_to_cxcywh(boxes).clamp(1e-5, 1 - 1e-5)
        metadata = DenoisingMetadata(
            positive_mask, negative_mask, valid_mask, target_boxes, labels, dn_count
        )
        return query, anchors, ~valid_mask, metadata

    def _attention_mask(self, dn_count: int, device: torch.device) -> torch.Tensor | None:
        if dn_count == 0:
            return None
        total = dn_count + self.num_queries
        mask = torch.zeros(total, total, device=device, dtype=torch.bool)
        # Matching queries cannot read denoising queries.
        mask[dn_count:, :dn_count] = True
        group_size = dn_count // self.denoising_groups
        for group in range(self.denoising_groups):
            start, end = group * group_size, (group + 1) * group_size
            mask[start:end, :start] = True
            mask[start:end, end:dn_count] = True
        return mask

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        targets: list[dict] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Decode ``memory [B,Q,C]`` into normalized boxes and class logits."""
        batch = memory.shape[0]
        regular_query = memory[:, : self.num_queries] + self.query_embedding.weight.unsqueeze(0)
        regular_anchors = torch.sigmoid(self.learned_anchors.weight).unsqueeze(0).expand(batch, -1, -1)

        metadata = None
        if self.training and targets is not None:
            dn_query, dn_anchors, dn_padding, metadata = self._build_denoising_queries(
                targets, memory.device, memory.dtype
            )
            query = torch.cat((dn_query, regular_query), dim=1)
            anchors = torch.cat((dn_anchors, regular_anchors), dim=1)
            query_padding = torch.cat(
                (dn_padding, torch.zeros(batch, self.num_queries, device=memory.device, dtype=torch.bool)), dim=1
            )
            attention_mask = self._attention_mask(metadata.dn_count, memory.device)
        else:
            query = regular_query
            anchors = regular_anchors
            query_padding = None
            attention_mask = None

        decoded = self.decoder(
            query,
            memory,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=query_padding,
            memory_key_padding_mask=~memory_mask,
        )
        logits, boxes = self.ffn(decoded, anchors)
        dn_count = metadata.dn_count if metadata is not None else 0
        result: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "pred_logits": logits[:, dn_count:],
            "pred_boxes": boxes[:, dn_count:],
        }
        losses: dict[str, torch.Tensor] = {}
        if self.training and targets is not None:
            losses.update(self.detector_loss(result["pred_logits"], result["pred_boxes"], targets))
            if metadata is not None and dn_count:
                losses.update(self.cdn_loss(logits[:, :dn_count], boxes[:, :dn_count], metadata))
            else:
                zero = boxes.sum() * 0.0
                losses.update({"loss_cdn_reg": zero, "loss_cdn_cls": zero, "loss_cdn_neg": zero})
            result["losses"] = losses
        return result

