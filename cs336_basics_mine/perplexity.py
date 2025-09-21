import torch
import torch.nn as nn
from jaxtyping import Float, Int

def perplexity(losses: Float[torch.Tensor, "m"]):
    return losses.mean().exp()