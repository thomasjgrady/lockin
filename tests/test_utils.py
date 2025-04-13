import os
import torch
import torch.distributed as dist
from lockin.utils import init_distributed, pack, unpack
import torch.multiprocessing as tmp


def test_pack_unpack():
    xs = [torch.randn(size=[i], dtype=torch.float32) for i in range(1, 10)]
    x, s = pack(xs)
    ys = unpack(x, s)
    assert all(torch.all(x == y) for x, y in zip(xs, ys))

def _test_init_distributed(rank: int, world_size: int):

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "22500"

    mesh = init_distributed(dp_size=2)
    assert dist.get_world_size(mesh.get_group("dp")) == 2
    assert dist.get_world_size(mesh.get_group("tp")) == 2

def test_init_distributed():
    world_size = 4
    tmp.spawn(fn=_test_init_distributed, args=(world_size,), nprocs=world_size, join=True)