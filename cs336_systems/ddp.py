
import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics_mine.transformer import Transformer_LM
from cs336_basics_mine.AdamW import AdamW

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def ddp(rank, world_size, args):
    setup(rank, world_size)
    # scatter data
    if rank == 0:
        data = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length))
        scatter_list = list(torch.chunk(data, world_size, dim=0))  # e: need to convert from tuple to list
        tensor = torch.empty((args.batch_size // world_size, args.context_length), dtype=torch.int64) # Placeholder for received data 
    else:
        scatter_list = None # Only root provides the list
        tensor = torch.empty((args.batch_size // world_size, args.context_length), dtype=torch.int64) # e: need to specify dtype

    dist.scatter(tensor, scatter_list if rank == 0 else None, src=0)

    model = Transformer_LM(args.d_model, args.num_heads, args.d_ff, args.vocab_size, args.context_length, args.num_layers, args.theta, args.context_length, args.device, args.model_dtype)

    # sync initial weights
    if rank == 0:
        state_dict = [model.state_dict()]  # e: broadcast_object_list expects lists of the same size, so wrap state_dict in a list of 1 element to ensure this
    else:
        state_dict = [None]
    dist.broadcast_object_list(state_dict, src=0)  # e: broadcast only takes a tensor. this works for a dict
    if rank != 0:
        model.load_state_dict(state_dict[0])


    optimizer = AdamW(model.parameters(), 0.1, (args.beta1, args.beta2), args.eps, args.weight_decay)

    s = 0
    for param in model.parameters():
        s += param.sum()
    print(f"before training, rank {rank} param sum: {s}")  # simple way to check params are the same across ranks

    # train
    for i in range(args.train_steps):
        loss = model(tensor).mean()
        loss.backward()

        for param in model.parameters():  # e: remember to iteratre through params so we can call param.grad
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
            param.grad = param.grad / world_size  # e: ReduceOp.AVG not supported in gloo...
        
        optimizer.step()
        optimizer.zero_grad()
        
        s = 0
        for param in model.parameters():
            s += param.sum()
        print(f"before training, step {i} rank {rank} param sum: {s}")
    


if __name__ == "__main__":
    world_size = 4
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=80)
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--context_length', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--theta', type=float, default=10000.)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_dtype', type=str, default='float32')

    # optimizer
    parser.add_argument('--lrmax', type=float, default=1e-4)
    parser.add_argument('--lrmin', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_warmup', type=int, default=100)
    parser.add_argument('--num_cosine', type=int, default=5000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # training loop
    parser.add_argument('--batch_size', type=int, default=32)  # batch_size * train_steps * context_length = 40000000
    parser.add_argument('--train_steps', type=int, default=10)

    args = parser.parse_args()
    if args.model_dtype == 'bf16':
        args.model_dtype = torch.bfloat16
    elif args.model_dtype == 'float32': 
        args.model_dtype = torch.float32
    else:
        raise
    args.device = torch.device(args.device)

    mp.spawn(fn=ddp, args=(world_size, args), nprocs=world_size, join=True)  # e: only provide the 2+ args, while the 1st is assumed to be the rank and will be provided by mp