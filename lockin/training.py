from typing import Callable

from torch import Tensor
from lockin.models import Model
from torch.optim import Optimizer
from torch.distributed.device_mesh import DeviceMesh


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
    ...

def sft_loss(
    tokens: Tensor,
    logits: Tensor,
    mask: Tensor
) -> Tensor:
    ...