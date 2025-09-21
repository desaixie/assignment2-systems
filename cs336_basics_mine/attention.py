import torch
import torch.nn as nn
from jaxtyping import Float, Int, Bool
import einops
from cs336_basics.linear import Linear
from cs336_basics.RoPE import RoPE

def softmax(x: Float[torch.Tensor, "..."], dim=-1) -> Float[torch.Tensor, "..."]:
    maximum = torch.max(x, dim=dim, keepdim=True).values  # e: remember to add the .values, as the function turns a named tuple (values, indices)
    x = x - maximum
    exp_x = x.exp()
    normalizer = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / normalizer
    
"""
resource accounting
forward shapes
1.1 (batch num_heads seqlen d_k) x (batch num_heads d_k seqlen) -> (batch num_heads seqlen seqlen)
1.2 (batch num_heads seqlen seqlen) / (batch num_heads seqlen seqlen) 
2. (batch num_heads seqlen seqlen) + (batch num_heads seqlen seqlen)
3. softmax(batch num_heads seqlen seqlen). 4 FLOPs per element
4. (batch num_heads seqlen seqlen) x (batch num_heads seqlen d_k) -> (batch num_heads seqlen d_k)

forward FLOPs
4*batch*num_heads*seqlen*seqlen*d_k + 6*batch*num_heads*seqlen*seqlen
    two matmuls (same FLOPs) + 6 element-wise
"""
# passed test in one try
def scaled_dot_product_attention(
            Q: Float[torch.Tensor, "batch ... seqlen d_k"], 
            K: Float[torch.Tensor, "batch ... seqlen d_k"], 
            V: Float[torch.Tensor, "batch ... seqlen d_v"],
            mask: Bool[torch.Tensor, "seqlen seqlen"] | None = None,
        ) -> Float[torch.Tensor, "batch ... d_v"]:
    d_k = Q.shape[-1]
    affinity_logits = Q @ K.transpose(-1, -2) / torch.sqrt(torch.tensor(d_k, device=Q.device, dtype=Q.dtype)) # "batch ... seqlen seqlen"
    
    # mask
    if mask is not None:
        adding_mask = torch.where(mask, torch.zeros_like(mask), torch.ones_like(mask) * float('-inf')).to(device=Q.device, dtype=torch.float32)
        affinity_logits += adding_mask
    
    affinity_score = softmax(affinity_logits, dim=-1)  # "batch ...", normalized over num keys dim
    attention_value = affinity_score @ V
    return attention_value
    
"""
resource accounting
forward shapes
1. (batch seqlen d_model) x (d_model 3*d_model) -> (batch seqlen 3*d_model)
2. 2 RoPEs on (batch num_heads seqlen d_k)
3. attention
4. (batch seqlen d_model) x (d_model d_model) -> (batch seqlen d_model)

forward FLOPs
8*batch*seqlen*d_model*d_model + 4*batch*seqlen*d_model + 4*batch*d_model*seqlen*seqlen + 6*batch*num_heads*seqlen*seqlen = 8*BS*1600*1600 + 4*BS*1600 + 4*BSS*1600 + 6*BSS*25 = 20,486,400BS + 6,550BSS

# parameters:
    4 * d_model * d_model + 1 * RoPE = 10,371,072
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: int | None = None, max_seq_len: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None, disable_rope: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = Linear(d_model, 3 * self.d_model, device=device, dtype=dtype)
        self.out_proj = Linear(self.d_model, d_model, device=device, dtype=dtype)  # e: in_features should be d_model, not d_k
        if disable_rope:
            self.rope = None
        else:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device, dtype=dtype)
    
    def forward(self, x: Float[torch.Tensor, "batch seqlen d_model"], token_positions: Int[torch.Tensor, "... seqlen"] | None = None) -> Float[torch.Tensor, "batch seqlen d_model"]:
        seqlen = x.shape[1]

        qkv = self.qkv(x)
        Q, K, V = qkv.split([self.d_model] * 3, dim=-1)  # e: split takes input of a single or many sizes, not the numebr of splits
        
        # extract num_heads and move to batch-like dim
        Q = einops.rearrange(Q, "batch seqlen (num_heads d_k) -> batch num_heads seqlen d_k", num_heads=self.num_heads, d_k = self.d_k)
        K = einops.rearrange(K, "batch seqlen (num_heads d_k) -> batch num_heads seqlen d_k", num_heads=self.num_heads, d_k = self.d_k)
        V = einops.rearrange(V, "batch seqlen (num_heads d_k) -> batch num_heads seqlen d_k", num_heads=self.num_heads, d_k = self.d_k)

        # apply rope
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(0, seqlen, 1, device=x.device).unsqueeze(0)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        # causal mask
        causal_mask = torch.triu(torch.ones(seqlen, seqlen)).transpose(0, 1).to(device=x.device, dtype=torch.bool)  # lower triangle are True

        heads = scaled_dot_product_attention(Q, K, V, causal_mask)
        heads = einops.rearrange(heads, "batch num_heads seqlen d_k -> batch seqlen (num_heads d_k)")
        return self.out_proj(heads)
