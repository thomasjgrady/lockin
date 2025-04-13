import json
import os
from pathlib import Path
from typing import cast

import torch

from .layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, all_gather
from .. import Model
import torch.nn as nn
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed import ProcessGroup
from pydantic import BaseModel
import torch.distributed as dist
from torch.distributed import ProcessGroup
from flash_attn import flash_attn_varlen_func_with_kvcache
from normalization import FusedRMSNorm
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


class ModelArgs(BaseModel, extra="forbid"):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048
    max_batch_len: int = 8192
    use_scaled_rope: bool = False

def apply_scaling(freqs: Tensor) -> Tensor:
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * torch.pi / freqs
    new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    return torch.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs,
        new_freqs,
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # cos = freqs_cis.real.contiguous().float()
    # sin = freqs_cis.imag.contiguous().float()
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return freqs_cis, cos, sin

def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: Tensor,
    xk: Tensor,
    freqs_cis: Tensor,
) -> tuple[Tensor, Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs, tp_group: ProcessGroup) -> None:
        
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = dist.get_world_size(group=tp_group)
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.tp_group = tp_group

        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, process_group=tp_group)
        self.wk = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, process_group=tp_group)
        self.wv = ColumnParallelLinear(args.dim, self.n_kv_heads * self.head_dim, process_group=tp_group)
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, process_group=tp_group)
    
    def forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
        freqs_cis: Tensor
    ) -> Tensor:

        xq = self.wq.forward(x)
        xk = self.wk.forward(x)
        xv = self.wv.forward(x)
        
        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xkv = torch.stack([xk, xv], dim=1) # (n_tokens, 2, n_local_kv_heads, head_dim)

        xo: Tensor = flash_attn_varlen_kvpacked_func(
            q=xq,
            kv=xkv,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True
        ) # type: ignore
        xo = xo.contiguous().view(-1, self.n_local_heads * self.head_dim)
        
        return self.wo.forward(xo)

    @torch.inference_mode()
    def inference_forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
        rotary_sin: Tensor,
        rotary_cos: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_seqlens: Tensor,
        cache_indices: Tensor
    ) -> Tensor:

        xq = self.wq.forward(x)
        xk = self.wk.forward(x)
        xv = self.wv.forward(x)
        
        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        xo: Tensor = flash_attn_varlen_func_with_kvcache(
            q=xq,
            cu_seqlens_q=cu_seqlens, # type: ignore
            max_seqlen_q=max_seqlen, # type: ignore
            cu_seqlens_k=cu_seqlens, # type: ignore
            k_cache=k_cache,
            v_cache=v_cache,
            k=xk,
            v=xv,
            rotary_sin=rotary_sin,
            rotary_cos=rotary_cos,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_indices,
            causal=True
        )

        xo = xo.view(-1, self.n_local_heads * self.head_dim)
        return self.wo.forward(xo)

class FeedForward(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        tp_group: ProcessGroup
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.tp_group = tp_group

        self.w1 = ColumnParallelLinear(dim, hidden_dim, process_group=tp_group)
        self.w2 = RowParallelLinear(hidden_dim, dim, process_group=tp_group)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, process_group=tp_group)

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.w1.forward(x)
        y3 = self.w3.forward(x)
        x2 = F.silu(y1) * y3
        return self.w2.forward(x2)
    
    @torch.inference_mode()
    def inference_forward(self, x: Tensor) -> Tensor:
        y1 = self.w1.forward(x)
        y3 = self.w3.forward(x)
        x2 = F.silu(y1) * y3
        return self.w2.forward(x2)
    
class TransformerBlock(nn.Module):
    
    def __init__(self, layer_id: int, args: ModelArgs, tp_group: ProcessGroup) -> None:
        
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, tp_group=tp_group)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            tp_group=tp_group
        )
        self.layer_id = layer_id
        self.attention_norm = FusedRMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = FusedRMSNorm(args.dim, eps=args.norm_eps)
        self.tp_group = tp_group

    def forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
        freqs_cis: Tensor
    ) -> Tensor:
        h = x + self.attention.forward(self.attention_norm(x), cu_seqlens, max_seqlen, freqs_cis)
        y = h + self.feed_forward.forward(self.ffn_norm(h))
        return y

    @torch.inference_mode()
    def inference_forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
        rotary_sin: Tensor,
        rotary_cos: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_seqlens: Tensor,
        cache_indices: Tensor
    ) -> Tensor:
        h = x + self.attention.inference_forward(
            self.attention_norm(x),
            cu_seqlens,
            max_seqlen,
            rotary_sin,
            rotary_cos,
            k_cache,
            v_cache,
            cache_seqlens,
            cache_indices
        )
        y = h + self.feed_forward.inference_forward(self.ffn_norm(h))
        return y

class Llama(nn.Module, Model):

    def __init__(self, params: ModelArgs, mesh: DeviceMesh) -> None:
        
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.mesh = mesh
        self.tp_group = mesh.get_group("tp")

        self.tok_embeddings = VocabParallelEmbedding(params.vocab_size, params.dim, process_group=self.tp_group)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, tp_group=self.tp_group))

        self.norm = FusedRMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(params.dim, params.vocab_size, process_group=self.tp_group)

        self.freqs_cis = torch.tensor(0.0)
        self.cos = torch.tensor(0.0)
        self.sin = torch.tensor(0.0)

    def set_freqs_cis(self) -> None:
        self.freqs_cis, self.cos, self.sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2,
            self.params.rope_theta,
        )

    def forward(
        self,
        tokens: Tensor,
        cu_seqlens: Tensor
    ) -> Tensor:
        
        h = self.tok_embeddings.forward(tokens)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = torch.cat([self.freqs_cis[:b-a] for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:])], dim=0)
        max_seqlen = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])

        for layer in self.layers:
            checkpointed_wrapper = cast(TransformerBlock, lambda *args, **kwargs: checkpoint( # type: ignore
                layer,
                *args,
                **kwargs,
                use_reentrant=False
            ))           
            h = checkpointed_wrapper(h, cu_seqlens, max_seqlen, freqs_cis)
        
        h = self.norm.forward(h)
        y = self.output.forward(h).float()
        return all_gather(y, self.tp_group)

    @torch.inference_mode()
    def inference_forward(
        self,
        tokens: Tensor,
        cu_seqlens: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_seqlens: Tensor,
        cache_indices: Tensor
    ) -> Tensor:
        
        h = self.tok_embeddings.forward(tokens)

        self.freqs_cis = self.freqs_cis.to(h.device)
        self.cos = self.cos.to(h.device, h.dtype)
        self.sin = self.sin.to(h.device, h.dtype)
        max_seqlen = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])

        for layer in self.layers:
            typed_layer = cast(TransformerBlock, layer)
            h = typed_layer.inference_forward(
                h,
                cu_seqlens,
                max_seqlen,
                self.sin,
                self.cos,
                k_cache,
                v_cache,
                cache_seqlens,
                cache_indices
            )
        
        h = self.norm.forward(h)
        y = self.output.forward(h).float()
        return all_gather(y, self.tp_group)
    
    def get_mesh(self) -> DeviceMesh:
        return self.mesh
    
    def init_cache(self, cache_size: int) -> Tensor:
        assert self.params.n_kv_heads is not None
        return torch.empty(
            size=[
                self.params.n_layers,
                cache_size,
                self.params.max_seq_len,
                self.params.n_kv_heads,
                self.params.dim // self.params.n_heads
            ],
            requires_grad=False
        )
    
    @staticmethod
    def get_tp_size_from_checkpoint_directory(checkpoint_dir: str | Path) -> int:
        return len([p for p in os.listdir(checkpoint_dir) if p.endswith(".pth") and "consolidated" in p])

    @staticmethod
    def from_meta_checkpoint(checkpoint_dir: str | Path, mesh: DeviceMesh, init_fsdp: bool) -> "Llama":
        
        d = Path(checkpoint_dir)
        shard_paths = sorted([d / p for p in os.listdir(d) if p.endswith(".pth") and "consolidated" in p])
        
        tp_group = mesh.get_group("tp")
        rank = dist.get_rank(group=tp_group)
        size = dist.get_world_size(group=tp_group)
        assert size == len(shard_paths), f"Expected tensor parallel size ({size}) to be equal to number of shards {len(shard_paths)}"

        with open(d / "params.json", "r") as f:
            params = ModelArgs(**json.load(f))

        device_before = torch.get_default_device()
        dtype_before = torch.get_default_dtype()
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.bfloat16)
        model = Llama(params, mesh)

        if init_fsdp:
            model = FSDP(
                model,
                process_group=mesh.get_group("dp"),
                sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                use_orig_params=True
            )

        torch.set_default_device(device_before)
        torch.set_default_dtype(dtype_before)

        weights = torch.load(shard_paths[rank], weights_only=True, map_location=device_before)
        model.set_freqs_cis()
        model.load_state_dict(weights, assign=True)

        return cast(Llama, model)