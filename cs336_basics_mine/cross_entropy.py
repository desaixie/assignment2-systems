import torch
import torch.nn as nn
from jaxtyping import Float, Int

""" naive version"""
# e: should've used labels as indices into logits! this works since False classes would be zeroed out by * with the one-hot vector
def cross_entropy_loss_naive_0(logits: Float[torch.Tensor, "batch ... vocab_size"], labels: Int[torch.Tensor, "batch vocab_size"] | Int[torch.Tensor, " batch_size"]) -> float:
    if len(labels.shape) == 1:
        labels = nn.functional.one_hot(labels)
    # softmax
    max_logits = logits.max(dim=-1, keepdim=True).values
    logits -= max_logits
    e = logits.exp()
    normalizer = e.sum(dim=-1, keepdim=True)
    probs = e / normalizer  # "batch ... vocab_size"

    # cross entropy
    loss_all = - probs.log()  # e: - labels * probs.log() produces large mismatch...
    loss = loss_all.mean()  # TODO: handle all batch-like dimensions as batch?
    return loss

"""log(e/e.sum) -> log(e) - log(e.sum) -> logits - log(e.sum)"""
def cross_entropy_loss_canceled_0(logits: Float[torch.Tensor, "batch ... vocab_size"], labels: Int[torch.Tensor, "batch vocab_size"] | Int[torch.Tensor, " batch_size"]) -> float:
    if len(labels.shape) == 1:
        labels = nn.functional.one_hot(labels)
    # softmax
    max_logits = logits.max(dim=-1, keepdim=True).values
    logits -= max_logits
    normalizer = logits.exp().sum(dim=-1, keepdim=True)

    # cross entropy
    loss_all = - (logits - normalizer.log())
    loss = loss_all.mean()
    return loss

"""using labels as index version"""
def cross_entropy_loss(logits: Float[torch.Tensor, "batch seqlen vocab_size"], labels: Int[torch.Tensor, "batch seqlen"]) -> float:
    if logits.ndim == 3:
        batch_size, seqlen, vocab_size = logits.shape
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == seqlen
        assert labels.ndim == 2
        labels_view = (batch_size, seqlen, 1)
    elif logits.ndim == 2:
        batch_size, vocab_size = logits.shape
        assert labels.shape[0] == batch_size
        assert labels.ndim == 1
        labels_view = (batch_size, 1)
    else:
        raise
    # softmax
    # e: not an error, but improvement is to not -= max_logits but separate this operation
    max_logits = logits.max(dim=-1, keepdim=True).values
    # logits -= max_logits  # prevent exp float overflow
    # normalizer = logits.exp().sum(dim=-1, keepdim=True)
    normalizer = (logits-max_logits).exp().sum(dim=-1, keepdim=True)


    # true_logits = logits[..., labels].unsqueeze(-1)  # e: could cause problems. use gather!
    true_logits = torch.gather(logits, -1, labels.view(*labels_view))

    # cross entropy
    # loss_all = - (true_logits - normalizer.log())
    loss_all = - (true_logits - (max_logits + normalizer.log()))
    loss = loss_all.mean()
    return loss
