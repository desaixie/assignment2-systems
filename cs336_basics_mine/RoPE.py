import torch
import torch.nn as nn
from jaxtyping import Float, Int
import einops

"""
resource accounting:
pre-computed rope buffer can be ignored. slicing from self.rope and rearranging are not FLOPs.

forward shape:
    1. (... seqlen pairs 2 2) x (... seqlen pairs 2) -> (... seqlen pairs 2), from the einsuum

forward FLOPs:
    2*...*seqlen*(pairs*2)*2 = 4*...*seqlen*d_k

# parameters:
    context_len * d_k * 2 = 1024 * 64 * 2 = 131,072

"""
class RoPE(nn.Module):
    def __init__(self, theta: int, d_k: int, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        rope = self.compute_rope_buffer()
        self.register_buffer("rope", rope, persistent=False)  # different from parameters, not learnable
        self.rope = self.rope.to(device=device, dtype=dtype)
        self.device = device
        
    def compute_rope_buffer(self) -> Float[torch.Tensor, "max_seq_len pairs 2 2"]:
        def compute_2x2(i: int, k: int) -> Float[torch.Tensor, "2 2"]:
            # rope angle
            theta_ik = torch.tensor(i / self.theta ** ((2*k-0) / self.d_k))  # e: make sure to pass in torch.Tensor to torch math functions, not scalars
            # e: turns out the output mismatch error was cause by an error in the handout, which uses 2k-1 above and for k in range(1, pairs+1), which shouldve been either
            # 1. even indices, i.e. 2k, with d in range(pairs). this is the standard
            # 2. odd indices, i.e. 2(k-1) or 2k-2, with d in range(1, pairs+1)
            return torch.tensor([[torch.cos(theta_ik), -torch.sin(theta_ik)], 
                                 [torch.sin(theta_ik), torch.cos(theta_ik)]], dtype=torch.float32)
        pairs = self.d_k // 2
        ret = []
        for n in range(self.max_seq_len):
            R_i = []
            for d in range(pairs):
                R_i.append(compute_2x2(n, d))
            R_i = torch.stack(R_i)
            ret.append(R_i)
        ret = torch.stack(ret)
        return ret
    
    """chatgpt version, no for loop, already close to the standalone, no 2x2 matrix version"""
    def compute_rope_buffer_chatgpt(self, ) -> torch.Tensor:
        d_over_2 = self.d_k // 2
        k = torch.arange(0, self.d_k, 2, dtype=torch.float32) / self.d_k            # (d_over_2,)
        inv_freq = 1.0 / (self.theta ** k)                                # (d_over_2,)
        pos = torch.arange(self.max_seq_len, dtype=torch.float32)         # (seqlen,)
        angles = pos[:, None] * inv_freq[None, :]                         # (seqlen, d_over_2)
        cos, sin = torch.cos(angles), torch.sin(angles)                   # each (seqlen, d_over_2)

        # Build 2x2 blocks: [[cos, -sin], [sin, cos]]
        R = torch.stack((
                torch.stack((cos, -sin), dim=-1),
                torch.stack((sin,  cos), dim=-1)
            ), dim=-2)                                                    # (seqlen, d_over_2, 2, 2)
        return R

    def forward(self, x: Float[torch.Tensor, "... seqlen d_k"], token_positions: Int[torch.Tensor, "... seqlen"]) -> Float[torch.Tensor, "... seqlen d_k"]:
        """ rearrange d to get pairs 2, then apply pe"""
        # select tokens
        token_positions = token_positions.to(self.device)
        # advanced indexing. the ... dimensions in token_positiosn are broadcasted to self.rope at the beginning
        rope = self.rope[token_positions]  # "... seqlen pairs 2 2"

        # e: cannot use bare number 2; have to use variable with value, e.g. two=2
        x = einops.rearrange(x, "... seqlen (pairs two)-> ... seqlen pairs two", two=2)  # split
        # x = x @ rope  # or rope.T?
        x = einops.einsum(rope, x, "... seqlen pairs i j, ... seqlen pairs j -> ... seqlen pairs i")  # vec dot over the 2nd dim in the 2x2 R and the 2-pair in x
        x = einops.rearrange(x, "... seqlen pairs two-> ... seqlen (pairs two)", two=2)  # merge
        return x
