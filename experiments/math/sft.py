from lockin.benchmarks.math import parse_answer, score_answer
from lockin.datasets.math import MATHDataset
from lockin.inference import generate
from lockin.models.llama import Llama
from lockin.tokenizers import ChatMessage, tokenize_chat_data
from lockin.tokenizers.llama import LlamaChatFormat, LlamaTokenizer
from lockin.training import make_linear_schedule, sft_loss
from lockin.utils import balance, get_balance_communication_pattern, init_distributed, pack
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
import itertools
import json
import math
import os
import torch
import torch.distributed as dist
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam


# Load model
model_name = "llama-3.2-3b-instruct"
checkpoint_dir = Path(f"/workspace/{model_name}")
tp_size = Llama.get_tp_size_from_checkpoint_directory(checkpoint_dir)
mesh = init_distributed(tp_size=tp_size, dp_size="infer")
torch.set_default_dtype(torch.bfloat16)
model = Llama.from_meta_checkpoint(checkpoint_dir, mesh, init_fsdp=False)
default_device = torch.get_default_device()

# Get distribution information
dp_group = mesh.get_group("dp")
dp_size = dist.get_world_size(dp_group)
dp_rank = dist.get_rank(dp_group)

tp_group = mesh.get_group("tp")

output_dir = Path(f"/workspace/experiments/MATH/sft/{model_name}")
if dist.get_rank() == 0:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
dist.barrier()

# Load tokenizer
tokenizer = LlamaTokenizer(checkpoint_dir / "tokenizer.model")
chat_format = LlamaChatFormat(tokenizer)

# Zero-2 is automatically applied when we set init_fsdp=True
# optim = torch.optim.AdamW(params=model.parameters(), betas=(0.9, 0.95), weight_decay=0.1)
optim = DistributedFusedAdam(
    params=model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    adam_w_mode=True,
    weight_decay=0.1,
    distributed_process_group=dp_group,
    redundant_process_group=tp_group,
    dtype=torch.float32,
    store_params=False,
    bucket_cap_mb=8
)
optim.zero_grad()

train_dataset_dir = Path("/workspace/data/MATH/train")
train_dataset = MATHDataset(train_dataset_dir)

# Batch size across all workers
global_batch_size = 128
assert global_batch_size % dp_size == 0
batch_size_per_replica = global_batch_size // dp_size

# Number of epochs and steps
num_epochs = 1
num_steps_per_epoch = int(math.ceil(len(train_dataset) / batch_size_per_replica))
num_steps = num_steps_per_epoch * num_epochs

# Learning rate schedule
lr_schedule = make_linear_schedule(num_steps, warmup_frac=0.1, min_val=1e-6, max_val=1e-4)

# maximum number of tokens per microbatch
max_microbatch_size = 4096

# Set default device to cpu for sampler
torch.set_default_device("cpu")

# Run training
pbar = tqdm(total=num_steps, desc="", disable=dist.get_rank() != 0)
step = 0
metrics: list[dict] = []

for epoch in range(num_epochs):
    sampler = DistributedSampler(train_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True, seed=1234 + epoch)
    loader = DataLoader(train_dataset, batch_size=batch_size_per_replica, collate_fn=lambda x: x)
    for batch_idx, batch in enumerate(loader):
        with torch.device(default_device):

            tokens, masks = tokenize_chat_data(batch, chat_format, add_final_eot=True)
            comm_pattern = get_balance_communication_pattern(tokens, dp_group, bucket_size=max_microbatch_size)

            token_microbatches = balance(tokens, dp_group, comm_pattern)
            mask_microbatches = balance(masks, dp_group, comm_pattern)
            num_microbatches = len(token_microbatches)

            loss: float = 0.0
            for tokens, masks in zip(token_microbatches, mask_microbatches):
                packed_tokens, cu_seqlens = pack(tokens)
                packed_masks, _ = pack(masks)
                packed_logits = model.forward(packed_tokens, cu_seqlens)
                microbatch_loss = sft_loss(packed_tokens[1:], packed_logits[:-1], packed_masks[1:]) / num_microbatches
                microbatch_loss.backward()
                loss += microbatch_loss.item()

            lr = lr_schedule(step)
            for g in optim.param_groups:
                g["lr"] = lr

            optim.step()
            optim.zero_grad()
            
            # Average loss across all replicas
            loss_tensor = torch.tensor(loss, dtype=torch.float32)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG, group=dp_group)
            loss = loss_tensor.item()

            metrics.append({
                "step": step,
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss,
                "lr": lr
            })
            pbar.set_description(f"{step=:08d}, {loss=:.6f}, {lr=:.6f}")
            pbar.update()
            step += 1

        if batch_idx >= 10:
            break
    break
                
# Run evaluations
test_dataset_dir = Path("/workspace/data/MATH/test")
test_dataset = MATHDataset(test_dataset_dir)
print(f"{len(test_dataset)=}")

global_batch_size = 1024
assert global_batch_size % dp_size == 0
batch_size_per_replica = global_batch_size // dp_size

results: list[dict] = []
sampler = DistributedSampler(test_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=False)
loader = DataLoader(test_dataset, batch_size=batch_size_per_replica, collate_fn=lambda x: x)

model.requires_grad_(False)
cache_size = 4
with torch.device(default_device):
    k_cache = model.init_cache(cache_size=cache_size)
    v_cache = model.init_cache(cache_size=cache_size)

pbar = tqdm(loader, desc="run benchmark", disable=dist.get_rank() != 0)

for batch in pbar:
    with torch.device(default_device):
        batch_for_gen: list[list[ChatMessage]] = [x[:-1] + [ChatMessage(role="assistant", content="")] for x in batch]
        ground_truth_responses = [x[-1].content for x in batch_for_gen]
        ground_truth_answers = [parse_answer(x) for x in ground_truth_responses]

        tokens, masks = tokenize_chat_data(batch, chat_format, add_final_eot=False)
        seqlens = [len(x) for x in tokens]

        generations = generate(
            model=model,
            prompts=tokens,
            temperature=0,
            max_new_tokens=2048,
            max_seq_len=model.config.max_seq_len,
            max_total_tokens=8192,
            eos_id=chat_format.eot_id(),
            k_cache=k_cache,
            v_cache=v_cache,
            progress=dist.get_rank() == 0,
            pbar_position=1
        )

        responses = [chat_format.decode(x[n:].tolist()) for x, n in zip(generations, seqlens)]
        answers = [parse_answer(x) for x in responses]
        scores = [score_answer(a, gt) for a, gt in zip(answers, ground_truth_answers)]

        batch_results = [
            {
                "prompt": [m.model_dump(mode="python") for m in b[:-1]],
                "ground_truth_response": gtr,
                "ground_truth_answer": gta,
                "response": r,
                "answer": a,
                "score": s
            }
            for b, gtr, gta, r, a, s in zip(
                batch_for_gen,
                ground_truth_responses,
                ground_truth_answers,
                responses,
                answers,
                scores
            )
        ]

        all_batch_results: list[list[dict]] = [list() for _ in range(dp_size)]
        dist.gather_object(
            obj=batch_results,
            object_gather_list=all_batch_results if dist.get_rank(group=dp_group) == 0 else None,
            group_dst=0,
            group=dp_group
        )
        results.extend(list(itertools.chain.from_iterable(all_batch_results)))

if dist.get_rank() == 0:
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f)
    print("Saved metrics and results")

dist.destroy_process_group()