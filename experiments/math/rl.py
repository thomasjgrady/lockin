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
    lr=1e-6,
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
global_batch_size = 512
assert global_batch_size % dp_size == 0
batch_size_per_replica = global_batch_size // dp_size

# Number of rollouts per dataset sample
rollouts_per_sample = 16

# Number of epochs and steps
num_steps = 1000
checkpoint_every: int = 100

# maximum number of tokens per microbatch
max_microbatch_size = 4096

# Set default device to cpu for sampler
torch.set_default_device("cpu")

# Run training
pbar = tqdm(total=num_steps, desc="", disable=dist.get_rank() != 0)
epoch = 0
step = 0
metrics: list[dict] = []

# System message for prompting
system_message = r"Think carefully and answer the given problem step by step. Keep your reasoning concise and clear. At the end of your response, output your answer in LaTeX like $\boxed{your answer}$"

while True:
    sampler = DistributedSampler(train_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True, seed=1234 + epoch)
    loader = DataLoader(train_dataset, batch_size=batch_size_per_replica, collate_fn=lambda x: x)
    for batch_idx, batch in enumerate(loader):
        with torch.device(default_device):
            
            # Inference
            inference_batch: list[ChatMessage] = [
                [ChatMessage(role="system", content=system_message)] + x[:-1] + [ChatMessage(role="assistant", content="")]
                for x in batch
            ]
            tokens, masks = tokenize_chat_data(batch, chat_format, add_final_eot=True)
            comm_pattern = get_balance_communication_pattern(tokens, dp_group, bucket_size=max_microbatch_size)

            token_microbatches = balance(tokens, dp_group, comm_pattern)
            mask_microbatches = balance(masks, dp_group, comm_pattern)
            num_microbatches = len(token_microbatches)