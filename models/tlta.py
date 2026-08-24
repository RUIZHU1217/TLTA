from __future__ import annotations

import torch
from torch import nn

from utils.boxes import denormalize_xyxy

from .f_lta import FeatureLevelTopologyAwareness
from .i_lta import InputLevelTopologyAwareness
from .p_lta import ProposalLevelTopologyAwareness


class TLTA(nn.Module):
    """Triple-Level Topology Awareness detector from Figure 6 and Algorithm 1.

    Information flow:
        image -> i-LTA -> f-LTA -> PPN -> PP/NP-HGC -> encoder -> CDN-FD
        decoder -> classification and box regression.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        self.num_classes = int(model_config["num_classes"])
        self.i_lta = InputLevelTopologyAwareness(model_config)
        self.f_lta = FeatureLevelTopologyAwareness(self.i_lta.out_channels, model_config)
        self.p_lta = ProposalLevelTopologyAwareness(
            self.f_lta.out_channels,
            self.num_classes,
            model_config,
            config["cdn_fd"],
        )
        test_config = config.get("test", {})
        # TODO: Not explicitly specified in the paper
        self.score_threshold = float(test_config.get("score_threshold", 0.05))
        # TODO: Not explicitly specified in the paper
        self.max_detections = int(test_config.get("max_detections", 100))

    def _postprocess(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        image_size: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
        probabilities = pred_logits.softmax(dim=-1)[..., 1:]
        scores, labels = probabilities.max(dim=-1)
        labels = labels + 1
        detections = []
        for sample_scores, sample_labels, sample_boxes in zip(scores, labels, pred_boxes):
            keep = sample_scores >= self.score_threshold
            sample_scores = sample_scores[keep]
            sample_labels = sample_labels[keep]
            sample_boxes = denormalize_xyxy(sample_boxes[keep], image_size)
            order = sample_scores.argsort(descending=True)[: self.max_detections]
            detections.append(
                {
                    "boxes": sample_boxes[order],
                    "scores": sample_scores[order],
                    "labels": sample_labels[order],
                }
            )
        return detections

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        """Run TLTA.

        Args:
            images: resized SAR batch ``[B,3,H,W]``.
            targets: training targets with pixel xyxy ``boxes`` and 1-based
                ``labels``. Required in training, optional in evaluation.
        Returns:
            In training, a dictionary of scalar losses. In evaluation, a list
            of ``boxes/scores/labels`` detection dictionaries.
        """
        if self.training and targets is None:
            raise ValueError("Training TLTA requires targets")
        image_size = (images.shape[-2], images.shape[-1])
        input_features, _ = self.i_lta(images)
        feature_level, _ = self.f_lta(input_features)
        proposal_level = self.p_lta(feature_level, image_size, targets if self.training else None)
        if self.training:
            return proposal_level["losses"]
        return self._postprocess(
            proposal_level["pred_logits"],
            proposal_level["pred_boxes"],
            image_size,
        )


def build_tlta(config: dict) -> TLTA:
    return TLTA(config)
