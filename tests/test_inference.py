import os
from pathlib import Path
import torch
import torch.distributed as dist
from lockin.inference import generate
from lockin.models.llama import Llama
from lockin.tokenizers import ChatMessage, tokenize_chat_data
from lockin.tokenizers.llama import LlamaChatFormat, LlamaTokenizer
from lockin.utils import init_distributed
import torch.multiprocessing as tmp


def _test_inference(rank: int, world_size: int):

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "22500"

    model_name = "llama-3.2-3b-instruct"
    checkpoint_dir = Path(f"/workspace/{model_name}")
    tp_size = Llama.get_tp_size_from_checkpoint_directory(checkpoint_dir)
    mesh = init_distributed(tp_size=tp_size, dp_size="infer")
    torch.set_default_dtype(torch.bfloat16)
    model = Llama.from_meta_checkpoint(checkpoint_dir, mesh, init_fsdp=False)
    default_device = torch.get_default_device()

    tokenizer = LlamaTokenizer(checkpoint_dir / "tokenizer.model")
    chat_format = LlamaChatFormat(tokenizer)

    model.requires_grad_(False)
    cache_size = 16
    with torch.device(default_device):
        k_cache = model.init_cache(cache_size=cache_size)
        v_cache = model.init_cache(cache_size=cache_size)

    xs = list(range(16))
    ys = [x + 1 for x in xs]

    batch = [
        [
            ChatMessage(role="user", content=f"What is {x} + 1?"),
            ChatMessage(role="assistant", content="")
        ]
        for x in xs
    ]

    tokens, masks = tokenize_chat_data(batch, chat_format, add_final_eot=False)
    seqlens = [len(x) for x in tokens]

    generations = generate(
        model=model,
        prompts=tokens,
        temperature=0,
        max_new_tokens=16,
        max_seq_len=model.config.max_seq_len,
        max_total_tokens=8192,
        eos_id=chat_format.eot_id(),
        k_cache=k_cache,
        v_cache=v_cache
    )

    responses = [chat_format.decode(x[n:].tolist()) for x, n in zip(generations, seqlens)]
    for x, r in zip(xs, responses):
        assert f"{x + 1}" in r

def test_inference():
    world_size = 1
    tmp.spawn(fn=_test_inference, args=(world_size,), nprocs=world_size, join=True)