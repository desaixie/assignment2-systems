import torch
import torch.nn as nn
from jaxtyping import Float, Int
from collections.abc import Callable
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: int):
        if lr < 0:
            raise
        defaults = {"lr": lr}
        super().__init__(params, defaults)  # e: this is the right way to call super.init
        print(f"params: {type(params)}")  # list of nn.Parameters
    
    def step(self, closure: Callable|None = None):
        loss = None if closure is None else closure()  # enable recomputing loss for optim.step, we don't need this
        for group in self.param_groups:
            lr = group["lr"]  # TODO what are groups and why they have correspondinr lrs?
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)  # iteration number
                grad = p.grad.data
                p.data -= (lr / math.sqrt(t+1)) * grad  # SGD with lr decay
                state["t"] = t + 1
        return loss

"""
lr tuning:
    1e0, 1e1, 1e2 converge faster and faster, while 1e3 causes divergence
"""
def sgd_training_loop():
    weights = torch.nn.Parameter(5 * torch.randn((10,10)))
    opt = SGD([weights], lr=1e3)

    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(f"loss: {loss.cpu().item()}")
        loss.backward()
        opt.step()

if __name__ == "__main__":
    sgd_training_loop()