"""
End‑to‑end multimodal pipeline combining images, tabular data, and patient context.

"""

import logging
import torch
from torch import nn

import timm
from models.patient_context import create_patient_context

logger = logging.getLogger(__name__)


class FullPipeline(nn.Module):
    """Main model that fuses image embeddings, optional tabular features, and optional patient context."""
    def __init__(
        self,
        backbone_name="efficientnet_b0",
        pretrained=True,
        tabular_dim=0,
        use_patient_context=False,
        context_aggregation="mean",
        context_fusion="residual",
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features
        self.tabular_dim = tabular_dim
        self.use_patient_context = use_patient_context
        self.context_aggregation = context_aggregation
        self.context_fusion = context_fusion

        if self.use_patient_context:
            self.context_module = create_patient_context(context_aggregation, self.embed_dim)

        classifier_dim = self.embed_dim + self.tabular_dim
        if self.use_patient_context:
            if context_fusion == "concat":
                classifier_dim += self.embed_dim
            elif context_fusion == "residual":
                classifier_dim += 2 * self.embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        ctx_str = f"yes ({context_aggregation}/{context_fusion})" if use_patient_context else "no"
        logger.info(
            f"FullPipeline: {backbone_name} (embed_dim={self.embed_dim}), "
            f"tabular_dim={tabular_dim}, context={ctx_str}, "
            f"classifier_input={classifier_dim}"
        )

    def encode(self, image):
        return self.backbone(image)

    def forward(self, batch):
        image = batch["image"]
        embedding = self.backbone(image)
        features = [embedding]

        if self.tabular_dim > 0 and "tabular" in batch:
            features.append(batch["tabular"])

        if self.use_patient_context and self.context_fusion != "none":
            if "patient_context" in batch and "patient_context_mask" in batch:
                context_embeddings = batch["patient_context"]
                context_mask = batch["patient_context_mask"]
                if self.context_aggregation == "attention":
                    context = self.context_module(embedding, context_embeddings, context_mask)
                else:
                    context = self.context_module(context_embeddings, context_mask)
            else:
                batch_size = embedding.size(0)
                context = self.context_module.default_context.unsqueeze(0).expand(batch_size, -1)

            if self.context_fusion == "concat":
                features.append(context)
            elif self.context_fusion == "residual":
                residual = embedding - context
                features.extend([context, residual])

        combined = torch.cat(features, dim=-1)
        return self.classifier(combined).squeeze(-1)


def pipeline_factory(config, tabular_dim=0):
    """Factory to instantiate a FullPipeline from a configuration dictionary."""
    use_tabular = config.get("return_tabular", False)
    return FullPipeline(
        backbone_name=config.get("backbone", "efficientnet_b0"),
        pretrained=config.get("pretrained", True),
        tabular_dim=tabular_dim if use_tabular else 0,
        use_patient_context=config.get("return_patient_context", False),
        context_aggregation=config.get("context_aggregation", "mean"),
        context_fusion=config.get("context_fusion", "residual"),
    )