import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import numpy as np
from jaxtyping import Float, Int
from collections.abc import Callable
from typing import List, Tuple
import math
import os
import typing
import argparse
from pathlib import Path
from datetime import datetime
from functools import partial
import wandb
from tqdm.auto import tqdm
import time

from cs336_basics.transformer import Transformer_LM
from cs336_basics.cross_entropy import cross_entropy_loss
from cs336_basics.perplexity import perplexity
from cs336_basics.AdamW import AdamW, gradient_clipping
from cs336_basics.lr_scheduler import lr_cosine_schedule
from cs336_basics.simple_tokenizer import Tokenizer

def load_data_memmap(file_path, dtype=np.int32):
    """Load data using memory mapping for efficient memory usage."""
    print(f"Loading data from {file_path} using memory mapping...")
    return np.memmap(file_path, dtype=dtype, mode='r')

def load_data_regular(file_path, dtype=np.int32):
    """Load data into regular memory."""
    print(f"Loading data from {file_path} into regular memory...")
    data = np.load(file_path)
    print(f"Loaded {len(data)} tokens")
    return data

"""custom torch dataset: returns a single sample, len checks upper bound"""
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, x: Int[np.ndarray, "data_size"], context_length: int, device: str):
        super().__init__()
        self.data = x
        self.context_length = context_length
        self.valid_len = x.shape[0] - context_length
        # self.device = device

    # e: switched to torch custom Dataset + torch Dataloader
    def __getitem__(self, idx) -> Tuple[Int[torch.Tensor, "contextlen"], Int[torch.Tensor, "contextlen"]]:
        token_ids = torch.tensor(self.data[idx:idx+self.context_length]).to(dtype=torch.int64)  # e: delay move to device to training thread
        target_ids = torch.tensor(self.data[idx+1:idx+1+self.context_length]).to(dtype=torch.int64)
        return token_ids, target_ids
    
    def __len__(self):
        return self.valid_len

def collate_fn(tuple_list: List[Tuple[Int[torch.Tensor, "contextlen"], Int[torch.Tensor, "contextlen"]]]) -> Tuple[Int[torch.Tensor, "batch contextlen"], Int[torch.Tensor, "batch contextlen"]]:
    l = list(zip(*tuple_list))
    return torch.stack(l[0], dim=0), torch.stack(l[1], dim=0)

    
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(ckpt, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]
    
def train(args):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    print(f"time: {timestamp}")
    print(f'args: {args}')
    
    wandb.init(project="cs336_a1", config=args, name=args.name)

    # tokenizer
    # from https://github.com/kkaitlyn111/cs336-assignment1/blob/main/cs336_basics/training_loop.py
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=args.special_tokens)
    s = "Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, “no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players.” With a few clicks here and there, you can findout that Noah Syndergaard’s fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game’s highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. The rising ocean of data has empowered an increasingly important actor in baseball’s culture: the analytical hobbyist."
    ids = tokenizer.encode(s)
    print(tokenizer.decode(ids))
    
    if args.reuse_pretokens and os.path.exists(args.pretokens_train_path):
        print(f"Reusing existing pretokenized training data from: {args.pretokens_train_path}")
    else:
        print(f"Creating fresh pretokenized training data...")
        tokenizer.pretokenize_file(
            args.train_path,
            args.pretokens_train_path,
            use_parallel=args.use_parallel_pretokenize
        )
        print(f"Saved fresh pretokenized training data to: {args.pretokens_train_path}")

    if args.reuse_pretokens and os.path.exists(args.pretokens_valid_path):
        print(f"Reusing existing pretokenized validation data from: {args.pretokens_valid_path}")
    else:
        print(f"Creating fresh pretokenized validation data...")
        tokenizer.pretokenize_file(
            args.valid_path,
            args.pretokens_valid_path,
            use_parallel=args.use_parallel_pretokenize
        )
        print(f"Saved fresh pretokenized validation data to: {args.pretokens_valid_path}")
    
    # load data based on the specified method
    if not args.use_memmap:
        print("Loading data into regular memory...")
        train_data = load_data_regular(args.pretokens_train_path)
        valid_data = load_data_regular(args.pretokens_valid_path)
    else:
        print("Loading data using memory mapping...")
        train_data = load_data_memmap(args.pretokens_train_path)
        valid_data = load_data_memmap(args.pretokens_valid_path)

    # consturct model
    print("building model")
    model = Transformer_LM(args.d_model, args.num_heads, args.d_ff, args.vocab_size, args.context_length, args.num_layers, args.theta, args.context_length, args.device, args.model_dtype)
    model.to(args.device)
    model = torch.compile(model, backend="aot_eager")
    wandb.watch(model)

    # construct optimizer
    print("building optimizer")
    lr_scheduler = partial(lr_cosine_schedule, lrmax=args.lrmax, lrmin=args.lrmin, num_warmup=args.num_warmup, num_cosine=args.num_cosine)
    optimizer = AdamW(model.parameters(), 0.1, (args.beta1, args.beta2), args.eps, args.weight_decay)
    
    
    start_step = 0
    if args.resume_path:
        start_step = load_checkpoint(args.resume_path, model, optimizer)
    
    optimizer.param_groups[0]['lr'] = lr_scheduler(start_step)  # update lr in AdamW
        
    # load data
    print("creating dataloader")
    pin_memory = True if "cuda" in str(args.device) else False
    prefetch = 2 if args.num_workers > 0 else None

    dataset = TextDataset(train_data, args.context_length, args.device)
    sampler = RandomSampler(dataset, replacement=True, num_samples=args.train_steps * args.batch_size)
    dataloader = iter(DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=args.num_workers>0, prefetch_factor=prefetch))

    val_dataset = TextDataset(valid_data, args.context_length, args.device)
    val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=args.train_steps * args.batch_size // args.val_every_step)
    val_dataloader = iter(DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=args.num_workers>0, prefetch_factor=prefetch))

    # training loop
    progress_bar = tqdm(
        range(0, args.train_steps),
        initial=start_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=False
    )
    lr = lr_scheduler(start_step)
    for i in range(start_step, args.train_steps):
        # train_one_step
        start_time = time.time()
        batch = next(dataloader)
        input_batch, target_batch = batch[0].to(args.device, non_blocking=pin_memory), batch[1].to(args.device, non_blocking=pin_memory)
        
        output = model(input_batch)
        loss = cross_entropy_loss(output, target_batch)
        loss.backward()
        grad_norm = gradient_clipping(model.parameters(), max_grad_norm=args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step_time = time.time() - start_time
        
        wandb_dict = {"loss": loss.item(), "global_step": i, "grad_norm": grad_norm.item(), "step_time": step_time, "lr": lr}

        # update lr in AdamW
        lr = lr_scheduler(i)
        for group in optimizer.param_groups:
            group['lr'] = lr
        
        # validation_step
        if i % args.val_every_step == 0:
            with torch.no_grad():
                batch = next(val_dataloader)
                input_batch, target_batch = batch[0].to(args.device, non_blocking=pin_memory), batch[1].to(args.device, non_blocking=pin_memory)
                output = model(input_batch)
                val_loss = cross_entropy_loss(output, target_batch)
                val_perpl = perplexity(val_loss)
                wandb_dict["val_loss"] = val_loss.item()
                wandb_dict["val_perpl"] = val_perpl.item()

        # save checkpoint
        if i % args.save_every_step == 0:
            save_dir = Path(args.save_dir) / timestamp
            save_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, i, save_dir / f"step_{i}.ckpt")

        del wandb_dict["step_time"]
        progress_bar.set_postfix(wandb_dict)
        progress_bar.update(1)
        wandb.log(wandb_dict)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--theta', type=float, default=10000.)
    parser.add_argument('--device', type=str, default='mps')
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

    # tokenizer and data
    parser.add_argument("--train_path", type=str, default = "data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid_path", type=str, default = "data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab_path", type=str, default = "tinystories_vocab.pkl")
    parser.add_argument("--merges_path", type=str, default = "tinystories_merges.pkl")
    parser.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"])
    parser.add_argument("--pretokens_train_path", type=str, default="data/openweb-train-tokenized.npy", help="Path to pretokenized training data")
    parser.add_argument("--pretokens_valid_path", type=str, default="data/openweb-valid-tokenized.npy", help="Path to pretokenized validation data")
    parser.add_argument("--reuse_pretokens", action="store_true", default = True, help="Reuse existing pretokenized data if available")
    parser.add_argument("--use_memmap", type=bool, default=True)
    parser.add_argument("--use_parallel_pretokenize", type=bool, default=True)  # Default to parallel for full dataset
    parser.add_argument("--num_workers", type=int, default=0)  # 0 or 10 for m4 macbook
    
    # training loop
    parser.add_argument('--batch_size', type=int, default=32)  # batch_size * train_steps * context_length = 40000000
    parser.add_argument('--train_steps', type=int, default=5000)
    parser.add_argument('--val_every_step', type=int, default=10)
    parser.add_argument('--log_every_step', type=int, default=10)
    parser.add_argument('--save_every_step', type=int, default=50)
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--name', type=str)
    parser.add_argument('--save_dir', type=str, default='logs/')

    

    args = parser.parse_args()
    if args.model_dtype == 'bf16':
        args.model_dtype = torch.bfloat16
    elif args.model_dtype == 'float32': 
        args.model_dtype = torch.float32
    else:
        raise
    args.device = torch.device(args.device)
    train(args)