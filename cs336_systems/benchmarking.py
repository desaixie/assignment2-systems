import time
import argparse
import torch
from cs336_basics_mine.transformer import Transformer_LM

def main(args):
    model = Transformer_LM(args.d_model, args.num_heads, args.d_ff, args.vocab_size, args.context_length, args.num_layers, args.theta, args.context_length, args.device, args.model_dtype)
    
    data = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length))
    
    for _ in range(args.bench_markup_steps):
        model(data)
    
    start_time = time.time()
    for _ in range(args.bench_steps):
        loss = model(data).mean(dim=-1)
        if args.bench_backward:
            loss.backward()
        torch.cuda.synchronize()  # no cuda, stopped this part...
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # benchmarking
    parser.add_argument('--bench_warmup_steps', type=int, default=10)
    parser.add_argument('--bench_steps', type=int, default=100)
    parser.add_argument('--bench_backward', type=bool, default=False, action="store_true")

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
    main(args)