"""
Image backbone classifiers for multimodal deep learning.

"""

import logging
from typing import Dict

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SUPPORTED_BACKBONES = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "vit_tiny": "vit_tiny_patch16_224",
}


class BackboneClassifier(nn.Module):
    """Combines an image backbone with optional tabular features for binary classification."""
    def __init__(self, backbone_name: str = "resnet50", tabular_dim: int = 0, num_classes: int = 1, pretrained: bool = True):
        super().__init__()

        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone_name}. Choose from: {list(SUPPORTED_BACKBONES.keys())}")

        timm_name = SUPPORTED_BACKBONES[backbone_name]
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            img_size=128 if "vit" in backbone_name else None,
        )
        self.embed_dim = self.backbone.num_features

        head_input_dim = self.embed_dim + tabular_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        logger.info(f"BackboneClassifier: {backbone_name} (embed_dim={self.embed_dim}, tabular_dim={tabular_dim})")

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = self.encode(batch["image"])
        if "tabular" in batch:
            embedding = torch.cat([embedding, batch["tabular"]], dim=1)
        return self.head(embedding).squeeze(-1)


def backbone_factory(config: Dict, tabular_dim: int = 0) -> BackboneClassifier:
    """Factory function to instantiate a BackboneClassifier from a config dict."""
    backbone_name = config.get("backbone", "resnet50")
    pretrained = config.get("pretrained", True)
    return BackboneClassifier(
        backbone_name=backbone_name,
        tabular_dim=tabular_dim,
        num_classes=1,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    for name in SUPPORTED_BACKBONES:
        model = BackboneClassifier(backbone_name=name, tabular_dim=9, num_classes=1, pretrained=False)
        n_params = sum(p.numel() for p in model.parameters())
        dummy_image = torch.randn(2, 3, 128, 128)
        dummy_tabular = torch.randn(2, 9)
        output = model({"image": dummy_image, "tabular": dummy_tabular})
        print(f"{name}: {n_params:,} params, output shape={output.shape}")