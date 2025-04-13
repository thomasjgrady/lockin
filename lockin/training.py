from typing import Callable

from torch import Tensor
from lockin.models import Model
from torch.optim import Optimizer
from torch.distributed.device_mesh import DeviceMesh
import torch.nn.functional as F


def step(
    mesh: DeviceMesh,
    model: Model,
    optim: Optimizer,
    lr: float,
    max_microbatch_size: int
) -> None:
    ...

def make_linear_schedule(
    num_steps: int,
    warmup_frac: float,
    min_val: float,
    max_val: float
) -> Callable[[int], float]:
    warmup_steps = int(num_steps * warmup_frac)
    def f(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps * max_val
        else:
            frac = (step - warmup_steps) / (num_steps - warmup_steps)
            return (1 - frac) * max_val + frac * min_val

    return f

def sft_loss(
    tokens: Tensor,
    logits: Tensor,
    mask: Tensor
) -> Tensor:
    return F.cross_entropy(logits[mask], tokens[mask], reduction="mean")