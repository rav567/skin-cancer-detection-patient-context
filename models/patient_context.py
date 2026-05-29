"""
Patient context aggregation methods: mean, max, and attention over historical embeddings.

"""
import torch
from torch import nn


class MeanPatientContext(nn.Module):
    """Averages context embeddings across a patient's lesions, falling back to a learnable default."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.default_context = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.default_context, std=0.02)

    def forward(self, context_embeddings, mask):
        batch_size = context_embeddings.size(0)
        default = self.default_context.unsqueeze(0).expand(batch_size, self.embed_dim)

        #no lesion has context yet (e.g. epoch 0 before the cache is built), so the collated context collapses to a placeholder feature dim. Fall back entirely to the learned default.
        if context_embeddings.size(-1) != self.embed_dim:
            return default

        mask_f = mask.unsqueeze(-1).float()
        context_sum = (context_embeddings * mask_f).sum(dim=1)
        context_count = mask_f.sum(dim=1).clamp(min=1.0)
        context_mean = context_sum / context_count

        has_context = (mask.sum(dim=1) > 0).float().unsqueeze(-1)
        return has_context * context_mean + (1 - has_context) * default


class MaxPatientContext(nn.Module):
    """Takes the elementwise maximum over context embeddings, ignoring masked positions."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.default_context = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, context_embeddings, context_mask):
        batch_size = context_embeddings.size(0)
        default = self.default_context.unsqueeze(0).expand(batch_size, self.embed_dim)

        if context_embeddings.size(-1) != self.embed_dim:
            return default

        mask = context_mask.unsqueeze(-1).bool()
        masked = context_embeddings.masked_fill(~mask, float("-inf"))
        max_pooled, _ = masked.max(dim=1)
        has_any = context_mask.sum(dim=1) > 0
        return torch.where(has_any.unsqueeze(-1), max_pooled, default)


class AttentionPatientContext(nn.Module):
    """Attends from current lesion embedding to all historical context embeddings."""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.default_context = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.default_context, std=0.02)

    def forward(self, current_embedding, context_embeddings, mask):
        batch_size = current_embedding.size(0)
        default = self.default_context.unsqueeze(0).expand(batch_size, self.embed_dim)

        if context_embeddings.size(-1) != self.embed_dim:
            return default

        query = current_embedding.unsqueeze(1)
        key_padding_mask = (mask == 0)

        all_padding = key_padding_mask.all(dim=1)
        safe_mask = key_padding_mask.clone()
        safe_mask[all_padding, 0] = False

        attended, _ = self.attention(
            query=query,
            key=context_embeddings,
            value=context_embeddings,
            key_padding_mask=safe_mask,
        )
        attended = attended.squeeze(1)

        has_context = (~all_padding).float().unsqueeze(-1)
        return has_context * attended + (1 - has_context) * default


def create_patient_context(aggregation, embed_dim):
    """Factory that returns the appropriate patient context module given an aggregation type."""
    if aggregation == "mean":
        return MeanPatientContext(embed_dim)
    if aggregation == "max":
        return MaxPatientContext(embed_dim)
    if aggregation == "attention":
        return AttentionPatientContext(embed_dim)
    raise ValueError(f"Unknown aggregation: {aggregation}")