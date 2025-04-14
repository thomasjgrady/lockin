from typing import Literal
from torch import Tensor
from pydantic import BaseModel
import torch
from .models import Model
from .utils import pack
import torch.distributed as dist
from tqdm.auto import tqdm


class Generation(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    tokens: Tensor
    start: int
    prefix_len: int
    batch_idx: int

@torch.inference_mode()
def generate(
    model: Model,
    prompts: list[Tensor],
    temperature: float,
    max_new_tokens: int,
    max_seq_len,
    max_total_tokens: int,
    eos_id: int,
    k_cache: Tensor,
    v_cache: Tensor,
    progress: bool = False,
    pbar_position: int = 0
) -> list[Tensor]:
    
    tp_group = model.get_mesh().get_group("tp")
    
    assert k_cache.shape == v_cache.shape
    cache_size = k_cache.shape[1]
    
    queue: dict[int, Generation] = {}
    for i, p in enumerate(prompts):
        assert p.dim() == 1, f"Expected 1 dimensional inputs, but batch item {i} had dimension {p.dim()}"
        assert p.dtype == torch.long, f"Expected long dtype for all inputs, but batch item {i} had dtype {p.dtype}"
        assert len(p) <= max_seq_len, f"Expected all inputs to have at most {max_seq_len} tokens, but batch item {i} has {len(p)} tokens"
        queue[i] = Generation(
            tokens=p,
            start=0,
            prefix_len=len(p),
            batch_idx=i
        )

    active: dict[int, Generation] = {}
    completed: dict[int, Tensor] = {}

    def active_token_count() -> int:
        return sum(len(x.tokens) for x in active.values())
    
    pbar = tqdm(total=len(queue), desc="generating", disable=not progress, position=pbar_position)
    
    while True:

        # Remove items from active if done
        to_remove: list[int] = []
        for i, x in active.items():
            if len(x.tokens) >= max_seq_len:
                to_remove.append(i)
            elif len(x.tokens) - x.prefix_len >= max_new_tokens:
                to_remove.append(i)
            elif x.tokens[-1] == eos_id:
                to_remove.append(i)
        
        for i in to_remove:
            x = active.pop(i)
            completed[x.batch_idx] = x.tokens
            pbar.update()

        # Enqueue longest items until active is saturated
        while len(queue) > 0 and active_token_count() < max_total_tokens and len(active) < cache_size:

            key_to_remove: int | None = None
            for i, x in queue.items():
                if len(x.tokens) + active_token_count() <= max_total_tokens:
                    key_to_remove = i
                    break
            
            if key_to_remove is None:
                break

            x = queue.pop(key_to_remove)
            next_slot = min(i for i in range(cache_size) if i not in active)
            active[next_slot] = x

        if len(queue) == 0 and len(active) == 0:
            break
        
        # Peform a generation step
        cache_indices = list(sorted(active.keys()))
        cache_seqlens = [active[i].start for i in cache_indices]
        tokens = [active[i].tokens[active[i].start:] for i in cache_indices]
        packed_tokens, cu_seqlens = pack(tokens)

        packed_logits = model.inference_forward(
            packed_tokens,
            cu_seqlens,
            k_cache,
            v_cache,
            torch.tensor(cache_seqlens, dtype=torch.int32),
            torch.tensor(cache_indices, dtype=torch.int32)
        )

        # Sampling
        next_logits = torch.concat([packed_logits[x-1].view(1, -1) for x in cu_seqlens[1:]], dim=0)
        if temperature == 0:
            next_tokens = torch.argmax(next_logits, dim=-1)
        else:
            next_probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tokens = torch.multinomial(next_probs, num_samples=1)

        next_tokens = next_tokens.reshape(-1)
        dist.broadcast(next_tokens, group=tp_group, group_src=0)

        for i, t in enumerate(next_tokens):
            slot = cache_indices[i]
            active[slot].start = len(active[slot].tokens)
            active[slot].tokens = torch.concat([active[slot].tokens, t.view(1)])
            
    
    output = list(x[1] for x in sorted(completed.items(), key=lambda x: x[0]))
    assert len(output) == len(prompts)
    return output