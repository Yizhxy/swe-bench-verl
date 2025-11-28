
import os
import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import hydra
from verl import DataProto

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size (num GPUs)")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    # 让 vLLM 只看见你想用的 GPU（可选）
    # 例如：CUDA_VISIBLE_DEVICES=0,1 python a.py --tp 2
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible:
        gpus = [g for g in visible.split(",") if g.strip() != ""]
        if args.tp > len(gpus):
            raise ValueError(f"--tp={args.tp} exceeds visible GPUs={len(gpus)}")
    else:
        if torch.cuda.is_available() and args.tp > torch.cuda.device_count():
            raise ValueError(f"--tp={args.tp} exceeds total GPUs={torch.cuda.device_count()}")

    print(f"Loading {args.model} with TP={args.tp} ...")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=args.trust_remote_code,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    prompts = [
        tok.apply_chat_template(
            [{"role":"user","content":"讲讲 RMSNorm 是什么，和 LayerNorm 的区别？"}],
            tokenize=False, add_generation_prompt=True
        ),
        tok.apply_chat_template(
            [{"role":"user","content":"写一个两行的 Python 函数，返回 x 的平方。"}],
            tokenize=False, add_generation_prompt=True
        ),
    ]

    params = SamplingParams(max_tokens=128, temperature=0.7, top_p=0.9)
    outputs = llm.generate(prompts, params)

    for i, out in enumerate(outputs):
        print(f"\n--- Prompt {i+1} ---")
        print(out.outputs[0].text)

def test_path():
    import os
    from pathlib import Path

    path = "$HOME/data/gsm8k/train.parquet"

    # 展开 $HOME 环境变量和 ~
    expanded = os.path.expanduser(os.path.expandvars(path))
    print(expanded)
    exit(0)

from verl import DataProto
from pathlib import Path



def test_pkl():
    save_dir = Path("./data")
    all_batches = []

    for pkl_file in sorted(save_dir.glob("training_step_*.pkl")):
        batch = DataProto.load_from_disk(str(pkl_file))
        responses = batch.batch["responses"]
        print(responses.shape)
        all_batches.append(batch)

    print(f"Loaded {len(all_batches)} rollout steps")







if __name__ == "__main__":
    test_pkl()
