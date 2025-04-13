from torch import Tensor
from torch.distributed import P2POp
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from typing import Callable, Literal
import math
import torch
import torch.distributed as dist


def pack(xs: list[Tensor]) -> tuple[Tensor, Tensor]:
    seqlens = [len(x) for x in xs]
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    return torch.concat(xs, dim=0), cu_seqlens

def unpack(x: Tensor, cu_seqlens: Tensor) -> list[Tensor]:
    return [x[a:b] for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:])]

def init_distributed(
    backend: Literal["nccl", "gloo", "mpi", "infer"] = "infer",
    tp_size: int | Literal["infer"] = "infer",
    dp_size: int | Literal["infer"] = "infer"
) -> DeviceMesh:
    
    use_cuda = torch.cuda.is_available()
    if backend == "infer":
        backend = "nccl" if use_cuda else "gloo"

    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()

    if tp_size == "infer":
        assert dp_size != "infer", "Cannot infer tp_size and dp_size"
        assert world_size % dp_size == 0, "Expected world size to be divisible by dp_size"
        tp_size = world_size // dp_size
    
    elif dp_size == "infer":
        assert tp_size != "infer", "Cannot infer tp_size and dp_size"
        assert world_size % tp_size == 0, "Expected world size to be divisible by tp_size"
        dp_size = world_size // tp_size

    mesh = init_device_mesh(
        device_type="cuda" if use_cuda else "cpu",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp")
    )

    if use_cuda:
        gpus_per_node = torch.cuda.device_count()
        device_idx = dist.get_rank() % gpus_per_node
        torch.set_default_device(f"cuda:{device_idx}")
        torch.cuda.set_device(device_idx)

    return mesh

def get_balance_communication_pattern(
    xs: list[Tensor],
    dp_group: ProcessGroup,
    bucket_size: float,
    by: Callable[[Tensor], float] = lambda x: float(x.numel())
) -> tuple[int, list[dict[int, list[int]]], list[dict[int, list[tuple[int, int]]]]]:
    
    if len(xs) > 0:
        assert all(x.dtype == xs[0].dtype for x in xs), "All tensors must be same dtype"
        assert all(x.device == xs[0].device for x in xs), "All tensors must be same device"
        assert all(x.dim() == 1 for x in xs), "Only balancing of 1-dimensional tensors is supported"
    
    dp_size = dist.get_world_size(dp_group)
    
    sizes = [[by(x), len(x)] for x in xs]
    assert all(x[0] <= bucket_size for x in sizes)
    all_sizes: list[list[tuple[float, int]]] = [[] for _ in range(dp_size)]
    dist.all_gather_object(all_sizes, sizes, dp_group)

    total_size = sum(sum(x[0] for x in sizes) for sizes in all_sizes)
    num_buckets = int(math.ceil(total_size / bucket_size))
    if num_buckets % dp_size != 0:
        num_buckets = num_buckets + dp_size - (num_buckets % dp_size)
    assert num_buckets % dp_size == 0

    ideal_bucket_size = total_size / num_buckets
    
    # list of index -> rank
    send_to: list[dict[int, list[int]]] = [{ r: list() for r in range(dp_size) } for _ in range(dp_size)]
    
    # list of rank -> (bucket index, size)
    recv_from: list[dict[int, list[tuple[int, int]]]] = [{ r: list() for r in range(dp_size) } for _ in range(dp_size)]

    all_sizes_sorted = [sorted(enumerate(x), key=lambda x: x[1][0], reverse=True) for x in all_sizes]

    found_allocation = False
    max_iter = 100

    for i in range(max_iter):

        if all(sum(len(x) for x in st.values()) == len(szs) for st, szs in zip(send_to, all_sizes_sorted)):
            found_allocation = True
            break

        for bucket_idx in range(num_buckets):
            dst = bucket_idx % dp_size
            local_bucket_idx = bucket_idx // dp_size
            found_bucket = False
            for cap in [ideal_bucket_size, bucket_size]:
                for offset in range(dp_size):
                    src = (dst + offset) % dp_size
                    for j, (s, l) in all_sizes_sorted[src]:
                        if any(j in send_to[src][k] for k in range(dp_size)):
                            continue
                        current_size = 0
                        for src_ in range(dp_size):
                            current_size += sum(x[1] for x in recv_from[dst][src_] if x[0] == local_bucket_idx)
                        src_start = dst if current_size <= ideal_bucket_size else (dst + 1) % dp_size
                        src = (src_start + offset) % dp_size
                        if current_size + s <= cap:
                            send_to[src][dst].append(j)
                            recv_from[dst][src].append((local_bucket_idx, l))
                            found_bucket = True
                            break
                    if found_bucket:
                        break
                if found_bucket:
                        break

    if not found_allocation:
        raise RuntimeError(f"Could not find balanced allocation satisfying constraints. Current best guess:\n{send_to=}\n{recv_from=}")

    return num_buckets // dp_size, send_to, recv_from


def balance(
    xs: list[Tensor],
    dp_group: ProcessGroup,
    comm_pattern: tuple[int, list[dict[int, list[int]]], list[dict[int, list[tuple[int, int]]]]]
) -> list[list[Tensor]]:
    
    if len(xs) > 0:
        assert all(x.dtype == xs[0].dtype for x in xs), "All tensors must be same dtype"
        assert all(x.device == xs[0].device for x in xs), "All tensors must be same device"
        assert all(x.dim() == 1 for x in xs), "Only balancing of 1-dimensional tensors is supported"
    
    dp_rank = dist.get_rank(dp_group)
    
    num_buckets, send_to, recv_from = comm_pattern
    ops: list[P2POp] = []

    for dst, idxs in send_to[dp_rank].items():
        if len(idxs) > 0:
            x = torch.concat([xs[i] for i in idxs])
            ops.append(P2POp(op=dist.isend, tensor=x, group=dp_group, group_peer=dst))

    recv_data: list[tuple[list[int], list[int], Tensor]] = []
    
    for src, l in recv_from[dp_rank].items():
        if len(l) > 0:
            bucket_idxs = [x[0] for x in l]
            sizes = [x[1] for x in l]
            cu_sizes: list[int] = torch.cumsum(torch.tensor([0] + sizes, dtype=torch.int32), dim=0).tolist()
            buf = torch.empty(size=[cu_sizes[-1]], device=xs[0].device, dtype=xs[0].dtype)
            recv_data.append((bucket_idxs, cu_sizes, buf))
            ops.append(P2POp(op=dist.irecv, tensor=buf, group=dp_group, group_peer=src))

    work = dist.batch_isend_irecv(ops)
    for w in work:
        w.wait()

    buckets: list[list[Tensor]] = [list() for _ in range(num_buckets)]
    for bucket_idxs, cu_sizes, buf in recv_data:
        assert len(bucket_idxs) == len(cu_sizes) - 1
        for i, (a, b) in enumerate(zip(cu_sizes[:-1], cu_sizes[1:])):
            buckets[bucket_idxs[i]].append(buf[a:b])

    return buckets

def normalize(x: Tensor, eps: float = 1e-10) -> Tensor:
    return (x - x.mean(dim=-1)) / (x.std(dim=-1) + eps)