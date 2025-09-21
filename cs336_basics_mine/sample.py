import torch
import torch.nn as nn
import numpy as np
from jaxtyping import Float, Int
from collections.abc import Callable
from typing import List, Tuple
import math
import os
import typing
import argparse
from pathlib import Path
import datetime
from functools import partial

from cs336_basics.transformer import Transformer_LM
from cs336_basics.cross_entropy import cross_entropy_loss
from cs336_basics.perplexity import perplexity
from cs336_basics.AdamW import AdamW, gradient_clipping
from cs336_basics.lr_scheduler import lr_cosine_schedule

def softmax_with_temp(x: Float[torch.Tensor, "..."], dim: int = -1, temperature: float = 1.) -> Float[torch.Tensor, "..."]:
    x /= temperature
    maximum = torch.max(x, dim=dim, keepdim=True).values  # e: remember to add the .values, as the function turns a named tuple (values, indices)
    x = x - maximum
    exp_x = x.exp()
    normalizer = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / normalizer

def top_p_sampling(probs: Float[torch.Tensor, "batch classes"], dim: int = -1, top_p: float = 1.):
    batch_size, nclasses = probs.shape
    sorted_, indices = torch.sort(probs, dim=-1, descending=True)
    cur_p = torch.zeros((batch_size,)).to(probs.device, probs.dtype)
    keep_elements = torch.zeros_like(probs)  # batch classes
    
    for i in range(nclasses):
        ind = indices[:, i]  # a column of indices
        batch_compare = cur_p >= top_p
        if batch_compare.all():
            break
        
        # update keep_indices, set ind to True while skipping batch_compare 
        keep_elements[:, ind] = torch.where(batch_compare, 0., 1.)

        # update cur_p while skipping batch_compare
        cur_p = torch.where(batch_compare, cur_p, cur_p + probs[ind])
    return probs[keep_elements]  # boolean masking

"""TODO: no kv cache or skipping computation where eot is True for now"""
def sample(model: Transformer_LM, prompt: Int[torch.Tensor, "batch seqlen"], max_sample_tokens: int, temperature: float, top_p: float, eot: int):
    batch_size = prompt.shape[0]
    ret = torch.ones((batch_size, max_sample_tokens), device=prompt.device) * eot
    # stop until <|endoftext|> or max_tokens
    for token_i in range(max_sample_tokens):
        # logits
        output = model(prompt)  # batch seqlen vocab_size
        last_token_logits = output[:, -1, :]  # batch vocab_size
        
        # to token
        # softmax with temperature
        probs = softmax_with_temp(last_token_logits, dim=-1, temperature=temperature)

        # top-p sampling
        probs = top_p_sampling(probs, dim=-1, top_p=top_p)
        
        # sample token
        dist = torch.distributions.Categorical(probs=probs)
        tokens = dist.sample() 

        # check eot
        batch_is_eot = tokens == eot
        ret[:, token_i] = torch.where(batch_is_eot, eot, tokens)
        if batch_is_eot.all():
            break
    return ret
