import torch
import torch.nn as nn
from jaxtyping import Float

"""
resource accounting
forward FLOPS: 0, since tensor slicing is not a FLOP.
# parameters:
    num_embeddings * embedding_dim = 50257 * 1600 = 80,411,200
"""
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device|None = None, dtype: torch.dtype | None = None):
        super().__init__()
        emb = nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), mean=0., std=1., a=-3., b=3.)
        self.emb = nn.Parameter(emb)
    
    def forward(self, token_ids: Float[torch.Tensor, "batch seqlen"]) -> Float[torch.Tensor, "batch seqlen d_model"]:
        return self.emb[token_ids]
        