import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from tensordict import TensorDict
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length

import os
import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import asyncio
import getpass
import inspect
import logging
import os
import pickle
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator
from pathlib import Path
import os

import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel, LoRAConfig
from vllm.lora.request import LoRARequest

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
import hydra

def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids
def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]



params = SamplingParams(max_tokens=2048, temperature=1, top_p=1,top_k=-1)
class RolloutDataCollector:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        data_config,
        generation_config,
        tokenizer,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.data_config = data_config
        self.generation_config = generation_config
        self.save_path='./data/'
        



        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler


        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.data_config["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.data_config.get("gen_batch_size", self.data_config.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.data_config.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.data_config.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * 1 ###bs=1


        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")


    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        # if self.async_rollout_mode:
        #     gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    # def _validate(self):
    #     data_source_lst = []
    #     reward_extra_infos_dict: dict[str, list] = defaultdict(list)

    #     # Lists to collect samples for the table
    #     sample_inputs = []
    #     sample_outputs = []
    #     sample_gts = []
    #     sample_scores = []
    #     sample_turns = []
    #     sample_uids = []

    #     for test_data in self.val_dataloader:
    #         test_batch = DataProto.from_single_dict(test_data)

    #         if "uid" not in test_batch.non_tensor_batch:
    #             test_batch.non_tensor_batch["uid"] = np.array(
    #                 [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
    #             )

    #         # repeat test batch
    #         test_batch = test_batch.repeat(
    #             repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
    #         )

    #         # we only do validation on rule-based rm
    #         if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
    #             return {}

    #         # Store original inputs
    #         input_ids = test_batch.batch["input_ids"]
    #         # TODO: Can we keep special tokens except for padding tokens?
    #         input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
    #         sample_inputs.extend(input_texts)
    #         sample_uids.extend(test_batch.non_tensor_batch["uid"])

    #         ground_truths = [
    #             item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
    #         ]
    #         sample_gts.extend(ground_truths)

    #         test_gen_batch = self._get_gen_batch(test_batch)
    #         test_gen_batch.meta_info = {
    #             "eos_token_id": self.tokenizer.eos_token_id,
    #             "pad_token_id": self.tokenizer.pad_token_id,
    #             "recompute_log_prob": False,
    #             "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
    #             "validate": True,
    #             "global_steps": self.global_steps,
    #         }
    #         print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

    #         # pad to be divisible by dp_size
    #         size_divisor = (
    #             self.actor_rollout_wg.world_size
    #             if not self.async_rollout_mode
    #             else self.config.actor_rollout_ref.rollout.agent.num_workers
    #         )
    #         test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
    #         if not self.async_rollout_mode:
    #             test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
    #         else:
    #             test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

    #         # unpad
    #         test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

    #         print("validation generation end")

    #         # Store generated outputs
    #         output_ids = test_output_gen_batch.batch["responses"]
    #         output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    #         sample_outputs.extend(output_texts)

    #         test_batch = test_batch.union(test_output_gen_batch)
    #         test_batch.meta_info["validate"] = True

    #         # evaluate using reward_function
    #         if self.val_reward_fn is None:
    #             raise ValueError("val_reward_fn must be provided for validation.")
    #         result = self.val_reward_fn(test_batch, return_dict=True)
    #         reward_tensor = result["reward_tensor"]
    #         scores = reward_tensor.sum(-1).cpu().tolist()
    #         sample_scores.extend(scores)

    #         reward_extra_infos_dict["reward"].extend(scores)
    #         print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
    #         if "reward_extra_info" in result:
    #             for key, lst in result["reward_extra_info"].items():
    #                 reward_extra_infos_dict[key].extend(lst)
    #                 print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

    #         # collect num_turns of each prompt
    #         if "__num_turns__" in test_batch.non_tensor_batch:
    #             sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

    #         data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

    #     self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

    #     # dump generations
    #     val_data_dir = self.config.trainer.get("validation_data_dir", None)
    #     if val_data_dir:
    #         self._dump_generations(
    #             inputs=sample_inputs,
    #             outputs=sample_outputs,
    #             gts=sample_gts,
    #             scores=sample_scores,
    #             reward_extra_infos_dict=reward_extra_infos_dict,
    #             dump_path=val_data_dir,
    #         )

    #     for key_info, lst in reward_extra_infos_dict.items():
    #         assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

    #     data_sources = np.concatenate(data_source_lst, axis=0)

    #     data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
    #     metric_dict = {}
    #     for data_source, var2metric2val in data_src2var2metric2val.items():
    #         core_var = "acc" if "acc" in var2metric2val else "reward"
    #         for var_name, metric2val in var2metric2val.items():
    #             n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
    #             for metric_name, metric_val in metric2val.items():
    #                 if (
    #                     (var_name == core_var)
    #                     and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
    #                     and (f"@{n_max}" in metric_name)
    #                 ):
    #                     metric_sec = "val-core"
    #                 else:
    #                     metric_sec = "val-aux"
    #                 pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
    #                 metric_dict[pfx] = metric_val

    #     if len(sample_turns) > 0:
    #         sample_turns = np.concatenate(sample_turns)
    #         metric_dict["val-aux/num_turns/min"] = sample_turns.min()
    #         metric_dict["val-aux/num_turns/max"] = sample_turns.max()
    #         metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

    #     return metric_dict

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    # def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
    #     """Compute rollout importance sampling weights and mismatch metrics, conditionally add weights to batch.

    #     This method computes IS weights to correct for distribution mismatch between
    #     rollout policy and training policy. It always computes metrics when enabled, but
    #     only adds weights to batch if algorithm.rollout_is is True.

    #     Args:
    #         batch: DataProto containing old_log_probs, rollout_log_probs, response_mask

    #     Returns:
    #         Tuple of (updated_batch, metrics) where:
    #             - updated_batch: Batch with rollout_is_weights added (if rollout_is=True)
    #             - metrics: Dictionary of IS and mismatch metrics (all with mismatch/ prefix)
    #     """
    #     # Compute rollout IS weights if enabled and data is available
    #     # rollout_is_threshold is the main on/off switch
    #     if self.config.algorithm.rollout_is_threshold is not None and "rollout_log_probs" in batch.batch:
    #         rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
    #             old_log_prob=batch.batch["old_log_probs"],
    #             rollout_log_prob=batch.batch["rollout_log_probs"],
    #             response_mask=batch.batch["response_mask"],
    #             rollout_is_level=self.config.algorithm.rollout_is_level,
    #             rollout_is_mode=self.config.algorithm.rollout_is_mode,
    #             rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
    #             rollout_is_threshold_lower=self.config.algorithm.rollout_is_threshold_lower,
    #             rollout_is_veto_threshold=self.config.algorithm.rollout_is_veto_threshold,
    #         )

    #         # Control: Should we apply weights to policy loss?
    #         # True = add weights to batch (actor will apply them)
    #         # False = don't add weights (metrics only, no loss modification)
    #         apply_weights = self.config.algorithm.get("rollout_is", False)

    #         if apply_weights:
    #             # Add IS weights to batch for distribution to workers
    #             batch = batch.union(rollout_is_weights)

    #         return batch, rollout_is_metrics

    #     # Return unchanged batch and empty metrics if IS is disabled
    #     return batch, {}

    def _save_rollout_dataproto(self, batch: DataProto, step: int):
        save_dir = Path(self.save_path)
    
        # 判断文件夹是否存在，如果不存在则创建
        if not save_dir.exists():
            os.makedirs(save_dir, exist_ok=True)
            print(f"[Info] Created directory: {save_dir}")
        
        # 构造文件名（零填充以避免空格）
        save_path = save_dir / f"training_step_{step:03d}.pkl"
        
        # 保存到磁盘
        batch.save_to_disk(str(save_path))
        print(f"[Info] Saved rollout data to: {save_path}")

    def fit(self,llm):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        # we start from step 1
        self.global_steps =0
        self.max_steps_duration = 0

        for epoch in range(1):
            for batch_dict in self.train_dataloader:
                self.global_steps += 1
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                
                input_s = batch.batch["input_ids"]
                print(input_s.shape)
                
                gen_batch = self._get_gen_batch(batch)

                input_s = gen_batch.batch["input_ids"]
                print(input_s.shape)
                
                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=4, interleave=True)

                input_s = gen_batch.batch["input_ids"]
                print(input_s.shape)
                
                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        print("============please wait for generation============")
                        # exit(0)
                        gen_batch_output =self.generate_sequences(gen_batch,llm)
                        input_s = gen_batch_output.batch["input_ids"]
                        print(input_s.shape)
                        # timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=4, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    self._save_rollout_dataproto(batch, step=self.global_steps)

    def generate_sequences(self,prompts: DataProto,llm) -> DataProto:
        """Generate sequences using the LLM model with the provided batch data.

        This function prepares the input data, invokes the LLM model to generate sequences,
        and returns the generated output in a DataProto format.

        Args:
            batch (DataProto): The input data containing prompts and other necessary information.
            llm (LLM): The LLM model used for sequence generation.

        Returns:
            DataProto: The output data containing the generated sequences.
        """
        prompts = prompts.to("cuda")
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        
        batch_size = idx.size(0)
        pad_token_id=prompts.meta_info["pad_token_id"]
        response_length = 512

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        
        vllm_inputs = [
            {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
        ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }

        lora_requests = None
        
        # users can customize different sampling_params at different run
        outputs = llm.generate(
            prompts=vllm_inputs,  # because we have already convert it to prompt token id
            sampling_params=params,
            lora_request=lora_requests,
            use_tqdm=False,
        )

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        response = []
        rollout_log_probs = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response_ids = output.outputs[sample_id].token_ids
                response.append(response_ids)
                if False:
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(curr_log_prob)

        response = pad_2d_list_to_length(response, pad_token_id, max_length=response_length).to(
            idx.device
        )
        if False:
            rollout_log_probs = pad_2d_list_to_length(
                rollout_log_probs, -1, max_length=response_length
            ).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if  False:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)