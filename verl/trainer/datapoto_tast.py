from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import numpy as np
import torch

from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

import os
import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
import hydra
from omegaconf import DictConfig
os.environ["CUDA_VISIBLE_DEVICES"] = "2,7"
def _get_free_port():
    import socket
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p

def init_dist():
    # 这些环境变量应该由 torchrun 设置，这里作为后备
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_get_free_port()))
    
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available")
        
    if torch.distributed.is_initialized():
        return

    backend = "nccl"
    
    # 获取本地rank（由torchrun设置）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 设置当前进程的GPU设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # 初始化分布式
    torch.distributed.init_process_group(
        backend=backend, 
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    if rank == 0:
        print(f"[Dist] Initialized. rank={rank}/{world_size}, local_rank={local_rank}")
        if torch.cuda.is_available():
            print(f"[Dist] Using GPU: {torch.cuda.current_device()}, GPU name: {torch.cuda.get_device_name()}")

def finalize_dist():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management."""
    
    # 初始化分布式环境
    init_dist()
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    try:
        # 获取配置中的张量并行大小
        original_tp = config.actor_rollout_ref.rollout.tensor_model_parallel_size
        
        print(f"[Rank {rank}] World size: {world_size}, Original tensor parallel size: {original_tp}")
        
        # 检查配置是否匹配
        if world_size < original_tp:
            if rank == 0:
                print(f"Warning: WORLD_SIZE({world_size}) < tensor_parallel_size({original_tp}), adjusting...")
            # 如果世界大小小于配置的TP，使用世界大小作为TP
            adjusted_tp = world_size
        else:
            adjusted_tp = original_tp
        
        # 关键修复：在创建配置对象之前修改OmegaConf配置
        # 创建一个配置的深拷贝，避免修改原始配置
        import copy
        rollout_config_dict = copy.deepcopy(config.actor_rollout_ref.rollout)
        
        # 修改张量并行大小
        rollout_config_dict.tensor_model_parallel_size = adjusted_tp
        
        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config_dict)
        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model, dataclass_type=HFModelConfig)
        
        print(f"[Rank {rank}] Initializing vLLMRollout with TP={adjusted_tp}...")
        
        # 单进程情况下，device_mesh设为None
        device_mesh = None
        
        # 如果是多进程，可能需要创建设备网格
        if world_size > 1:
            try:
                # 尝试导入并创建设备网格
                from verl.utils.device_mesh import create_device_mesh
                device_mesh = create_device_mesh(
                    mesh_shape=(adjusted_tp,),
                    axis_names=("tensor",)
                )
                print(f"[Rank {rank}] Created device mesh for tensor parallelism")
            except ImportError:
                print(f"[Rank {rank}] Warning: Could not import create_device_mesh, using None")
                device_mesh = None
        
        # 初始化vLLM Rollout
        llm = vLLMRollout(
            config=rollout_config, 
            model_config=model_config, 
            device_mesh=device_mesh
        )
        
        print(f"[Rank {rank}] vLLMRollout initialized successfully")
        
        try:
            results = llm.inference_engine.generate(["你好，给我一句测试用的短话。"])
            
            # 打印结果
            for i, result in enumerate(results):
                generated_text = result.outputs[0].text if hasattr(result, 'outputs') else result
                print(f"Result {i}: {generated_text}")
                
            print("Inference test completed successfully")
        except Exception as e:
            print(f"Error during generation on rank 0: {e}")
            import traceback
            traceback.print_exc()
    
        # 同步所有进程
        torch.distributed.barrier()
        
        if rank == 0:
            print("All ranks completed successfully")
        
    except Exception as e:
        print(f"[Rank {rank}] Error during execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理资源
        try:
            if 'llm' in locals():
                if hasattr(llm, "shutdown"):
                    llm.shutdown()
                elif hasattr(llm, "release"):
                    import asyncio
                    if asyncio.iscoroutinefunction(llm.release):
                        asyncio.run(llm.release())
                    else:
                        llm.release()
                
                print(f"[Rank {rank}] vLLMRollout cleanup completed")
        except Exception as e:
            print(f"[Rank {rank}] Error during shutdown: {e}")
            
        # 同步并清理分布式环境
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        finalize_dist()
        print(f"[Rank {rank}] Distributed environment finalized")

if __name__ == "__main__":
    main()