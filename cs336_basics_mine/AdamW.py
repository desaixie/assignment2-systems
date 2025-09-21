import torch
import torch.nn as nn
from jaxtyping import Float, Int
from collections.abc import Callable
from typing import List
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "decay_ratio": weight_decay}
        super().__init__(params, defaults)
        self.t = 1  # might be insecure
    
    def step(self, closure: Callable|None = None):
        loss = None if closure is None else closure()  # enable recomputing loss for optim.step, we don't need this
               
        for group in self.param_groups:
            lr = group["lr"]  # TODO what are groups and why they have correspondinr lrs?
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            decay_ratio = group["decay_ratio"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # t = state.get("t", 1)  # iteration number  TODO possible to store t elsewhere? 
                grad = p.grad.data
                m = state.get("m", 0)  # 1st moment
                m = beta1 * m + (1 - beta1) * grad

                v = state.get("v", 0)  # 2nd moment
                v = beta2 * v + (1 - beta2) * grad ** 2

                lr_t = lr * math.sqrt(1 - beta2**self.t) / (1 - beta1**self.t)

                p.data -= lr_t * m / (v.sqrt() + eps)
                p.data -= lr * decay_ratio * p.data
                
                # update state
                state["m"] = m
                state["v"] = v
            self.t += 1  # e: shouldn't be in the inner loop

        return loss

def gradient_clipping(param: List[nn.Parameter], max_grad_norm: float, eps: float = 1e-6):
    # for p in param:
    #     if p.grad is None:
    #         continue
    #     norm = torch.linalg.vector_norm(p.grad.data.flatten(), ord=2)
    #     # norm = p.grad.detach().data.norm(2)
    #     if norm > max_grad_norm:
    #         p.grad.data = p.grad.data * max_grad_norm / (norm + eps)
    
    # for p in param:
    #     if p.grad is None:
    #         continue
    #     print(f"norm_after: {torch.linalg.vector_norm(p.grad.data.flatten(), ord=2)}")

    # e: above implementation computes norm and scales norm per param. the correct implementation should compute norm and scale norm over all param together.
    param = [p for p in param if p.grad is not None]

    per_param_norm = torch.stack([p.grad.detach().norm(2) for p in param])
    norm = per_param_norm.norm(2)
    # norm = p.grad.detach().data.norm(2)
    if norm > max_grad_norm:
        for p in param:
            p.grad.data = p.grad.data * max_grad_norm / (norm + eps)
    for p in param:
        if p.grad is None:
            continue
    return norm