from torch import Tensor
from pydantic import BaseModel
import torch
from .models import Model
from .utils import pack


class Generation(BaseModel, extra="forbid"):
    tokens: Tensor
    start: int
    end: int
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
    v_cache: Tensor
) -> list[Tensor]:
    
    assert k_cache.shape == v_cache.shape
    cache_size = k_cache.shape[1]
    
    queue: dict[int, Generation] = {}
    for i, p in enumerate(prompts):
        assert p.dim() == 1, f"Expected 1 dimensional inputs, but batch item {i} had dimension {p.dim()}"
        assert p.dtype == torch.long, f"Expected long dtype for all inputs, but batch item {i} had dtype {p.dtype}"
        assert len(p) <= max_seq_len, f"Expected all inputs to have at most {max_seq_len} tokens, but batch item {i} has {len(p)} tokens"
        buf = torch.empty(size=[max_seq_len], dtype=torch.long)
        buf[:len(p)].copy_(p)
        queue[i] = Generation(
            tokens=buf,
            start=0,
            end=len(p),
            prefix_len=len(p),
            batch_idx=i
        )

    active: dict[int, Generation] = {}
    completed: dict[int, Tensor] = {}

    def active_token_count() -> int:
        return sum(len(x.tokens) for x in active.values())
    
    while len(queue) > 0 and len(active) > 0:

        # Remove items from active if done
        to_remove: list[int] = []
        for i, x in active.items():
            if x.end >= max_seq_len:
                to_remove.append(i)
            elif x.end - x.prefix_len >= max_new_tokens:
                to_remove.append(i)
            elif x.tokens[x.end-1] == eos_id:
                to_remove.append(i)
        
        for i in to_remove:
            x = active.pop(i)
            completed[x.batch_idx] = x.tokens

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
        
        # Peform a generation step
        cache_indices = list(sorted(active.keys()))
        cache_seqlens = [active[i].start for i in cache_indices]
        tokens = [active[i].tokens[active[i].start:active[i].end] for i in cache_indices]
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
        next_logits = torch.concat([packed_logits[x-1].view(1) for x in cu_seqlens[1:]], dim=0)
        if temperature == 0:
            next_tokens = torch.argmax(next_logits, dim=-1)
        else:
            next_probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tokens = torch.multinomial(next_probs, num_samples=1)

        next_tokens = next_tokens.reshape(-1)
        for i, t in enumerate(next_tokens):
            slot = cache_indices[i]
            next_token = int(t.item())
            active[i].tokens[active[slot].end] = next_token
            active[slot].end += 1
            active[slot].start = active[slot].end - 1
    
    return list(x[1] for x in sorted(completed.items(), key=lambda x: x[0]))