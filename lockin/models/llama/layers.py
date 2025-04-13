from typing import Any
from torch import Tensor
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ProcessGroup


class MegatronF(torch.autograd.Function):
    """
    Implements the `F` function from https://arxiv.org/pdf/1909.08053.pdf.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, process_group: ProcessGroup) -> Tensor: # type: ignore
        ctx.process_group = process_group
        return x
    
    @staticmethod
    def backward(ctx: Any, dy: Tensor) -> tuple[Tensor, None]: # type: ignore
        ctx.mark_dirty(dy)
        dist.all_reduce(dy, group=ctx.process_group)
        return dy, None

class MegatronG(torch.autograd.Function):
    """
    Implements the `G` function from https://arxiv.org/pdf/1909.08053.pdf.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, process_group: ProcessGroup) -> Tensor: # type: ignore
        ctx.mark_dirty(x)
        dist.all_reduce(x, group=process_group)
        return x

    @staticmethod
    def backward(ctx: Any, dy: Tensor) -> tuple[Tensor, None]: # type: ignore
        return dy, None
    
class AllGather(torch.autograd.Function):
    """
    Implements an all-gather operation along the last dimension
    of the input tensor.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, process_group: ProcessGroup) -> Tensor: # type: ignore
        ctx.rank, ctx.size = dist.get_rank(group=process_group), dist.get_world_size(group=process_group)
        ys = [torch.zeros_like(x) for _ in range(ctx.size)]
        dist.all_gather(ys, x, group=process_group)
        return torch.cat(ys, dim=-1)
    
    @staticmethod
    def backward(ctx: Any, dy: Tensor) -> tuple[Tensor, None]: # type: ignore
        rank, size = ctx.rank, ctx.size
        stride = dy.shape[-1] // size
        return dy[...,rank*stride:(rank+1)*stride], None


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: Tensor, process_group: ProcessGroup) -> Tensor: # type: ignore
        ctx.mark_dirty(x)
        ctx.process_group = process_group
        dist.all_reduce(x, group=process_group)
        return x

    @staticmethod
    def backward(ctx: Any, dy: Tensor) -> tuple[Tensor, None]: # type: ignore
        ctx.mark_dirty(dy)
        dist.all_reduce(dy, group=ctx.process_group)
        return dy, None

def megatron_f(x: Tensor, process_group: ProcessGroup) -> Tensor:
    return MegatronF.apply(x, process_group) # type: ignore

def megatron_g(x: Tensor, process_group: ProcessGroup) -> Tensor:
    return MegatronG.apply(x, process_group) # type: ignore

def all_reduce(x: Tensor, process_group: ProcessGroup) -> Tensor:
    return AllReduce.apply(x, process_group) # type: ignore

def all_gather(x: Tensor, process_group: ProcessGroup) -> Tensor:
    return AllGather.apply(x, process_group) # type: ignore
    
class ColumnParallelLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.process_group = process_group
        
        size = dist.get_world_size(group=process_group)
        assert self.out_features % size == 0, "`out_features` must be divisible by tensor parallel size in `ColumnParallelLinear`"
        self.local_out_features = out_features // size

        self.weight = nn.Parameter(torch.zeros(
            size=(self.local_out_features, self.in_features),
            device=device,
            dtype=dtype
        ))

    def forward(self, x: Tensor) -> Tensor:
        x = megatron_f(x, self.process_group)
        x = F.linear(x, self.weight)
        return x

class RowParallelLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.process_group = process_group
        
        size = dist.get_world_size(group=process_group)
        assert self.in_features % size == 0, "`in_features` must be divisible by tensor parallel size in `RowParallelLinear`"
        self.local_in_features = in_features // size

        self.weight = nn.Parameter(torch.zeros(
            size=(self.out_features, self.local_in_features),
            device=device,
            dtype=dtype
        ))

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.weight)
        x = megatron_g(x, self.process_group)
        return x

class VocabParallelEmbedding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        process_group: ProcessGroup,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False
    ) -> None:
        
        super().__init__()

        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        self.process_group = process_group
        
        tp = dist.get_world_size(group=process_group)
        per_partition_vocab_size = self.num_embeddings // tp
        self.vocab_start_index = dist.get_rank(group=process_group) * per_partition_vocab_size
        self.vocab_end_index = self.vocab_start_index + per_partition_vocab_size

        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        self.weight = torch.nn.Parameter(torch.zeros(self.num_embeddings_per_partition, self.embedding_dim))

    def forward(self, input_: Tensor) -> Tensor:  # type: ignore
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        output_parallel = torch.nn.functional.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # x = nn.Embedding
        output_parallel[input_mask, :] = 0.0
        return all_reduce(output_parallel, self.process_group)
    
    
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)