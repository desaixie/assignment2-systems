import torch
import torch.nn as nn
from jaxtyping import Float, Int, Bool
import einops
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.attention import MultiHeadAttention
from cs336_basics.FFN_SwiGLU import FFN_SwiGLU

"""
resource accounting
forward shape:
1.1 RMSNorm(batch seqlen d_model) 
1.2 attn(batch seqlen d_model)
1.3 (batch seqlen d_model) + (batch seqlen d_model)
2.1 RMSNorm(batch seqlen d_model)
2.2 ffn(batch seqlen d_model)
2.3 (batch seqlen d_model) + (batch seqlen d_model)

forward FLOPs:
two RMSNorms (9,600BS), two element-wise (3,200BS), 1 attn (20,486,400BS + 6,550BSS) , 1 ffn (61,478,400BS)

# parameters:
2 * 1600 + 10,371,072 + 30,720,000 = 41,094,272
"""
class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: int | None = None, max_seq_len: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None, disable_rope: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype, disable_rope=disable_rope)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = FFN_SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: Float[torch.Tensor, "batch seqlen d_model"], token_positions: Int[torch.Tensor, "... seqlen"] | None = None) -> Float[torch.Tensor, "batch seqlen d_model"]:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
        
"""resource accounting
total forward FLOPs:
    n blocks, 1 embedding (0 FLOP), 1 RMSNorm, 1 linear(d_model, vocab_size)
    48 * (9,600BS + 3,200BS + 20,486,400BS + 6,550BSS + 61,478,400BS) + 4,800BS + 2*BS*d_model*vocab_size = 4,095,752,000BS + 314,400BSS

# parameters
    48 * 41,094,272 + 80,411,200 + 1600 + 1600 * 50257 = 2,133,349,056 (2B)
memory assuming bf16 (16 bits/2 bytes)
    2,133,349,056 * 2 bytes = 4,266,698,112 bytes (4GB)
"""
class Transformer_LM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int, theta: int | None = None, max_seq_len: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None, disable_rope: bool = False):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.blocks = nn.ModuleList([
                Block(d_model, num_heads, d_ff, theta, context_length, device, dtype) 
                for _ in range(num_layers)
            ])  # e: need nn.ModuleList instead of a plain list to hold the blocks, so they are visible to Module methods like load_state_dict and to optimizers
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
        self.vocab_size = vocab_size
    
    def forward(self, token_ids: Int[torch.Tensor, "batch seqlen"]) -> Float[torch.Tensor, "batch seqlen vocab_size"]:
        x = self.embedding(token_ids)
        for i, b in enumerate(self.blocks):
            x = b(x)
        # e: a final RMSNorm before lm_head!
        output = self.lm_head(self.ln_final(x))
        return output
