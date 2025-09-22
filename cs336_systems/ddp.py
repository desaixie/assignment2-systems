import time
import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics_mine.transformer import Transformer_LM
from cs336_basics_mine.AdamW import AdamW

def setup(rank, world_size, seed):
    torch.manual_seed(seed)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float|None = None):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # braodcast initial weights
        if self.rank == 0:
            state_dict = [module.state_dict()]
        else:
            state_dict = [None]
        dist.broadcast_object_list(state_dict, src=0)
        if self.rank != 0:
            self.module.load_state_dict(state_dict[0])
        
        # backward all reduce (bucketed or per-grad hooked)
        if bucket_size_mb is not None:
            # construct buckets in reverse order
            self.buckets = []
            cur_bucket = []
            cur_bucket_size_mb = 0.
            # bucket_i = 0
            for param in reversed(list(module.parameters())):  # e: need to converting to list
                if not param.requires_grad:  # e: skip no grad params
                    continue
                if cur_bucket_size_mb > bucket_size_mb:
                    # full, start new bucket
                    self.buckets.append(cur_bucket)
                    cur_bucket = []
                    cur_bucket_size_mb = 0.
                    # bucket_i += 1
                # regular
                # cur_bucket.append(grad)  # e: can't bucket grad first, they are still None!
                cur_bucket.append(param)
                # self.param_to_bucket[id(param)] = bucket_i
                cur_bucket_size_mb += param.numel() * param.element_size() / (1024. * 1024.)  # e: two 1024s in MB
            # append last bucket
            self.buckets.append(cur_bucket)  # e: forgot about this

            """update bucket status"""
            self.param_to_bucket = {id(p): bid for bid, bucket in enumerate(self.buckets) for p in bucket}
            self.bucket_grad_ready_sets = [set() for _ in range(len(self.buckets))]
            # print(f"nbuckets: {len(self.buckets)}, bucket_i: {bucket_i}")
            self.pending = []
            def _grad_hook(param: torch.Tensor) -> None:  
                grad = param.grad
                if grad is None:
                    return
                # how to find out which bucket this param belongs to and increment bucket_grad_ready_counters?
                bucket_i = self.param_to_bucket[id(param)]
                self.bucket_grad_ready_sets[bucket_i].add((id(param)))

                if len(self.bucket_grad_ready_sets[bucket_i]) == len(self.buckets[bucket_i]):  # bucket ready to all reduce
                    # debugging outputs
                    # if self.rank == 0:
                        # for name, p2 in self.module.named_parameters():
                        #     if id(param) == id(p2):
                        #         print(f"\n\ncurrent param name: {name}, shape: {p2.shape}")
                        #         break
                        # for p in self.buckets[bucket_i]:
                        #     for name, p2 in self.module.named_parameters():
                        #         if id(p) == id(p2):
                        #             print(f"bucket {bucket_i}, len {len(self.buckets[bucket_i])}, grad param name: {name}, shape: {p2.shape} is None: {p.grad is None}")
                        #             break
                    grads = [p.grad for p in self.buckets[bucket_i]]
                    flat_grads = torch._utils._flatten_dense_tensors(grads)
                    handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
                    self.pending.append((handle, self.buckets[bucket_i], flat_grads, grads))

                
        else:
    
            self.pending = []
            def _grad_hook(param: torch.Tensor) -> None:  
                # only comm grad of the current param tensor
                grad = param.grad
                if grad is None:  # e: defensive, might not be needed
                    return
                handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                self.pending.append((handle, grad))
                # param.grad = param.grad / self.world_size  # e: cannot do this with async_op! async requires keeping the same buffer alive until wait() returns
                # e: thus, solution is to keep grad and assign the grad/world_size after wait()
            
        for param in module.parameters():
            if param.requires_grad:  # e: error without this check
                param.register_post_accumulate_grad_hook(_grad_hook)
    
    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        if self.bucket_size_mb is not None:
            for handle, params, flat_grads, grads in self.pending:
                handle.wait()
                grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)  # use help() to see signature
                for j, param in enumerate(params):
                    # param.grad = grads[j] / self.world_size
                    param.grad.copy_(grads[j] / self.world_size)  # e: in-place alternative TODO any difference?
            self.pending.clear()
            self.bucket_grad_ready_sets = [set() for _ in range(len(self.buckets))]  # e: forgot to reset this after each optm step, causing grad is None error in the next step
        else:
            for handle, grad in self.pending:
                handle.wait()
                grad.div_(self.world_size)  # divide in place (all_reduce is also in-place, so grad still refers to the correct param.grad)
            self.pending.clear()


"""standalone function version"""
def ddp(rank, world_size, args):
    setup(rank, world_size, args.seed)
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

    if args.ddp_wrapper_class:
        ddp_model = DDP(model, args.bucket_size_mb)  # wrap
        optimizer = AdamW(ddp_model.module.parameters(), 0.1, (args.beta1, args.beta2), args.eps, args.weight_decay)

        s = 0
        for param in ddp_model.module.parameters():
            s += param.sum()
        print(f"before training, rank {rank} param sum: {s}")  # simple way to check params are the same across ranks

        start_time = time.time()
        for i in range(args.train_steps):
            loss = ddp_model(tensor).mean()
            loss.backward()
            
            ddp_model.finish_gradient_synchronization()

            optimizer.step()
            optimizer.zero_grad()
            
            s = 0
            for param in ddp_model.module.parameters():
                s += param.sum()
            print(f"after training, step {i} rank {rank} param sum: {s}")
        print(f"using ddp_wrapper, time used for {args.train_steps} steps: {time.time() - start_time}")  
        # 0.27s, slower than batched_all_reduce.
        # 0.095s with bucketed grad, slightly faster than batched_all_reduce
        # ChatGPT: this is because on gloo, both comm and compute fight for the same resource, thus overlapping doesn't help much while batching helps a lot. Use bucketed overlapping to enjoy benefits of both.
    else:
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
        start_time = time.time()
        for i in range(args.train_steps):
            loss = model(tensor).mean()
            loss.backward()

            if args.batched_all_reduce:  # use special functions to batch/unbatch the tensors of different sizes, so we can all_reduce a single tensor, reduce #comm
                grads = [p.grad for p in model.parameters()]
                flat_grads = torch._utils._flatten_dense_tensors(grads)
                dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=False)
                grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)  # use help() to see signature
                for j, param in enumerate(model.parameters()):
                    param.grad = grads[j] / world_size
            else:
                for param in model.parameters():  # e: remember to iteratre through params so we can call param.grad
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
                    param.grad = param.grad / world_size  # e: ReduceOp.AVG not supported in gloo...
            
            optimizer.step()
            optimizer.zero_grad()
            
            s = 0
            for param in model.parameters():
                s += param.sum()
            print(f"after training, step {i} rank {rank} param sum: {s}")
        print(f"batched_all_reduce: {args.batched_all_reduce}, time used for {args.train_steps} steps: {time.time() - start_time}")  
        # on M4 mac CPU, 0.1s with batched_all_reduce, 0.3s without.
    # with same seed, two methods arrive at the same result after 10 steps
    


if __name__ == "__main__":
    world_size = 4  # setting to 1 to check non-ddp result
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_size_mb', type=float, default=None)
    parser.add_argument('--ddp_wrapper_class', default=False, action='store_true')
    parser.add_argument('--batched_all_reduce', default=False, action='store_true')
    parser.add_argument('--seed', type=float, default=1)

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