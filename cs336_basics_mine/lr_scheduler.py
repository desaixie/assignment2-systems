import torch
import torch.nn as nn
from jaxtyping import Float, Int
from collections.abc import Callable
import math

def lr_cosine_schedule(t: int, lrmax: float, lrmin: float, num_warmup: int, num_cosine: int) -> float:
    if t < num_warmup:
        return t / num_warmup * lrmax
    if t <= num_cosine:
        return lrmin + (1 + math.cos((t - num_warmup)/(num_cosine - num_warmup) * math.pi)) / 2 * (lrmax - lrmin)
    if t > num_cosine:
        return lrmin