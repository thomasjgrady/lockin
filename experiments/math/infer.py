import itertools
import json
import os
from pathlib import Path

import torch
from lockin.benchmarks.math import parse_answer, score_answer
from lockin.datasets.math import MATHDataset
from lockin.inference import generate
from lockin.models.llama import Llama
from lockin.tokenizers import ChatMessage, tokenize_chat_data
from lockin.tokenizers.llama import LlamaTokenizer, LlamaChatFormat
from lockin.utils import init_distributed
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm.auto import tqdm


def main(model_name: str) -> None:

    model_path = Path(f"/workspace/{model_name}")
    tp_size = Llama.get_tp_size_from_checkpoint_directory(model_path)
    mesh = init_distributed(tp_size=tp_size, dp_size="infer")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    torch.set_default_dtype(torch.bfloat16)
    dp_group = mesh.get_group("dp")

    model = Llama.from_meta_checkpoint(model_path, mesh, init_fsdp=False).requires_grad_(False)
    model.config.max_seq_len = 8192
    model.set_freqs_cis()

    tokenizer = LlamaTokenizer(model_path / "tokenizer.model")
    chat_format = LlamaChatFormat(tokenizer)

    dataset = MATHDataset("/workspace/data/MATH/train")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(dp_group),
        shuffle=False,
        seed=1234
    )
    loader = DataLoader(dataset, batch_size=128, sampler=sampler, collate_fn=lambda x: x)
    system_message = r"Think carefully and answer the given problem step by step. At the end of your response, output your answer in LaTeX like $\boxed{your answer}$"

    results: list[dict] = []
    cache_size = 16
    k_cache = model.init_cache(cache_size)
    v_cache = model.init_cache(cache_size)

    for batch in tqdm(loader, desc="Run benchmark", disable=dist.get_rank() != 0):

        typed_batch: list[list[ChatMessage]] = batch
        questions = [x[0].content for x in typed_batch]
        ground_truth_responses = [x[1].content for x in typed_batch]
        ground_truth_answers = [parse_answer(x[1].content) for x in typed_batch]
        
        prompts = [
            [
                ChatMessage(role="system", content=system_message),
                ChatMessage(role="user", content=q),
                ChatMessage(role="assistant", content="")
            ]
            for q in questions
        ]
        tokens, _ = tokenize_chat_data(prompts, chat_format, add_final_eot=False)
        seqlens = [len(x) for x in tokens]
        generations = generate(
            model,
            tokens,
            temperature=0,
            max_new_tokens=4096,
            max_seq_len=model.config.max_seq_len,
            max_total_tokens=model.config.max_seq_len,
            eos_id=chat_format.eot_id(),
            k_cache=k_cache,
            v_cache=v_cache,
            progress=dist.get_rank() == 0,
            pbar_position=1
        )
        responses = [chat_format.decode(x[n:].tolist()) for x, n in zip(generations, seqlens)]
        answers = [parse_answer(x) for x in responses]
        scores = [score_answer(a, gt) for a, gt in zip(answers, ground_truth_answers)]

        results.extend([
            {
                "question": q,
                "ground_truth_response": gtr,
                "ground_truth_answer": gta,
                "response": r,
                "answer": a,
                "score": s
            }
            for q, gtr, gta, r, a, s in zip(
                questions,
                ground_truth_responses,
                ground_truth_answers,
                responses,
                answers,
                scores
            )
        ])

    all_results: list[list[dict]] = [list() for _ in range(dist.get_world_size(dp_group))]
    dist.all_gather_object(all_results, results, dp_group)

    output_dir = Path(f"/workspace/traces/MATH/train/{model_name}")
    if dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / "results.jsonl", "w") as f:
            for r in itertools.chain.from_iterable(all_results):
                f.write(json.dumps(r) + "\n")

        with open(output_dir / "metadata.jsonl", "w") as f:
            json.dump(
                {
                    "system_message": system_message,
                    "model_name": model_name
                },
                f
            )

    dist.destroy_process_group()

if __name__ == "__main__":
    import fire
    fire.Fire(main)