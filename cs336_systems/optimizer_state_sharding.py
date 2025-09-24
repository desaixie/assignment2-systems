from typing import Type, Any, List
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics_mine.AdamW import AdamW

# e: mode.parameter() already returns a flattened list of nn.Parameter objects, so this is not needed
# def flatten_params_from_group(groups):
#     ret = []
#     for group in groups:
#         print(f"group keys: {group.keys()}")
#         ret.append(group["params"])
#     print(f"flattened len: {len(ret)}")
#     return ret

"""
greedily assign whole param tensors to each rank
sort params by size, then assign to each rank interleaved
returns sharded_params, list of each rank's param shard (a list of params)
"""
def shard_params_old(flattened_params: List[torch.Tensor], world_size: int):
    # if type(params) is dict:  # handle param groups (dict instead of list)
    #     # print(params.items())
    #     flattened_params = params["params"]
    #     # sorted_params = sorted(params.items(), reverse=True, key=lambda x: x[1].numel())  # sort by value  e: the dict is not like a named_parameters(), but like my AdamW's group
    #     sorted_params = sorted(flattened_params, reverse=True, key=lambda x: x.numel())  # sort by value
    # else:
    if type(flattened_params) is not List:  # e: type(v) is List instaed of v is List!
        flattened_params = list(flattened_params)
    params = [p for p in flattened_params if p.requires_grad]  # e: only keeping requires_grad params
    # print(f"len params: {len(params), len(flattened_params)}, world_size: {world_size}")
    assert len(params) >= world_size
    sorted_params = sorted(params, reverse=True, key=lambda x: x.numel())
    sharded_params = [[] for _ in range(world_size)]
    cur_rank = 0
    for p in sorted_params:
        if cur_rank == world_size:
            cur_rank = 0
        sharded_params[cur_rank].append(p)
        cur_rank += 1
    # if type(params) is dict:  # e: removed, since dict is group, not named_parameters()
    #     for i, _ in enumerate(sharded_params):
    #         sharded_params[i] = dict(sharded_params[i])
    return sharded_params

"""
greedily assign whole param tensors to each rank
sort params by size, then assign to each rank interleaved
returns sharded_params_indices, list of each rank's param shard indices (a list of params indices)
"""
def shard_params(flattened_params: List[torch.Tensor], world_size: int):
    if type(flattened_params) is not List:  # e: type(v) is List instaed of v is List!
        flattened_params = list(flattened_params)
    assert len(flattened_params) >= world_size
    sorted_param_indices = sorted(range(len(flattened_params)), reverse=True, key=lambda i: flattened_params[i].numel())
    sharded_param_indices = [[] for _ in range(world_size)]
    cur_rank = 0
    for ind in sorted_param_indices:
        if cur_rank == world_size:
            cur_rank = 0
        sharded_param_indices[cur_rank].append(ind)
        cur_rank += 1
    return sharded_param_indices

# class OptimizerStateSharding():
class OptimizerStateSharding(torch.optim.Optimizer):
    def __init__(self, params: List[torch.nn.Parameter], optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        # params is the complete set to be sharded
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        # self._flattened_params = [p for p in params if p.requires_grad]  # e: model.parameters() already gives flattened list of nn.Parameter objects!
        self._flattened_params = [p for p in params]  # keeping all params

        ### debugging
        # flattened_param_names = []
        # named_params = list(named_params)
        # for p1 in self._flattened_params:
        #     for name, p2 in named_params:
        #         if id(p1) == id(p2):
        #             flattened_param_names.append(name)
        #             break
        # print(f"rank {self.rank}, flattened_param_names: {flattened_param_names}")
        # print(f"len flattened: {len(list(self._flattened_params))}")  # e: avoid doing list(iterator) without storing it, otherwise it's left as an empty iterator

        # shard  # e: broadcasting params do not preserve the param identity across ranks. need to instead broadcast indices
        # if self.rank == 0:
        #     sharded_params = shard_params(self._flattened_params, self.world_size)
        # else:
        #     sharded_params = [None] * self.world_size
        # dist.broadcast_object_list(sharded_params, src=0)
        # self.sharded_params = sharded_params
        if self.rank == 0:
            sharded_param_indices = shard_params(self._flattened_params, self.world_size)
        else:
            sharded_param_indices = [None] * self.world_size
        dist.broadcast_object_list(sharded_param_indices, src=0)
        self.sharded_params = [[self._flattened_params[ind] for ind in sharded_param_indices[r]] for r in range(self.world_size)]

        ### debugging
        # sharded_param_names = []
        # for l in self.sharded_params:
        #     cur = []
        #     for p1 in l:
        #         for name, p2 in named_params:
        #             if id(p1) == id(p2):
        #                 cur.append(name)
        #                 break
        #     sharded_param_names.append(cur)
        #     cur = []
        # print(f"rank {self.rank}, sharded_param_names: {sharded_param_names}")
                    
        
        self.optim = optimizer_cls(params=self.sharded_params[self.rank], **kwargs)
        super().__init__(self._flattened_params, defaults=kwargs)
    
    def step(self, closure=None, **kwawrgs):
        # print(f"rank {self.rank}, before step, param sum: {sum([p.data.sum() for p in self._flattened_params])}")
        self.optim.step(closure, **kwawrgs)
        
        # broadcast these params to other ranks and receive
        # dist.broadcast(self.sharded_params[self.rank], src=self.rank)  # e: this is still a list. need to flatten/unflatten
        # print(f"rank {self.rank}, before broadcast, param sum: {sum([p.data.sum() for p in self._flattened_params])}")
        with torch.no_grad():  # e: ensure no_grad for synching
            for r in range(self.world_size):
                for p in self.sharded_params[r]:
                    # if r == self.rank:
                    #     tensor = p
                    # else:
                    #     tensor = torch.empty_like(p)
                    # shouldn't be creating placeholders. params should be broadcasted from the rank into themselves
                    dist.broadcast(p.data, src=r)  # in-place update parameters # e: broadcasting p causes error as it carries other info, instead p.data is correct
                    # e: this would hang if ps mismatch among ranks
        # print(f"rank {self.rank}, after broadcast param sum: {sum([p.data.sum() for p in self._flattened_params])}")



        # flat_params = torch._utils._flatten_dense_tensors(params)
        # dist.broadcast(flat_params, src=self.rank)  


    
    def add_param_group(self, param_group: dict[str, Any]):
        # if self.rank == 0:
        #     sharded_param_group = [shard_params(param_group, self.world_size)]
        # else:
        #     sharded_param_group = [None]
        # dist.broadcast_object_list(sharded_param_group, src=0)

        # return super().add_param_group(sharded_param_group[0][self.rank])
        return super().add_param_group(param_group)  # e: the super class sees the raw, unsharded param group in order to call optimizer.zero_grad(). self.optim handles the actual step and tracking optimizer state