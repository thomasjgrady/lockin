from typing import Protocol
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh


class Model(Protocol):

    def forward(
        self,
        tokens: Tensor,
        cu_seqlens: Tensor
    ) -> Tensor:
        ...

    def inference_forward(
        self,
        tokens: Tensor,
        cu_seqlens: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_seqlens: Tensor,
        cache_indices: Tensor,
    ) -> Tensor:
        ...

    def get_mesh(self) -> DeviceMesh: ...