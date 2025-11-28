# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any, List, Optional

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

def default_simple_reward(
    reward: List[float],
    completion_ids: List[List[int]],
    gold_completion_ids: List[List[int]],
    **kwargs,
) -> List[float]:
    rewards = []
    for _reward, comp, gold_comp in zip(reward, completion_ids, gold_completion_ids):
        _reward = float(comp == gold_comp) * float(_reward)
        rewards.append(_reward)
    return rewards


@register("swe_bench_naive")
class SweRewardManager(AbstractRewardManager):
    """适配 SWE-bench offline 数据格式的 RewardManager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key: str = "data_source") -> None:
        """
        Args:
            tokenizer: 用于 decode 的 tokenizer。
            num_examine: 每个 data_source 最多打印多少条样本做调试。
            compute_score: 计算 reward 的函数
            reward_fn_key: 
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        # 使用上面定义的 simple_reward 逻辑
        self.compute_score = default_simple_reward
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        - 从 data.batch["responses"] 取出模型生成的 completion token。
        - 从 data.non_tensor_batch["gold_completion_ids"] 取出 gold 序列。
        - 从 data.non_tensor_batch["reward"] 取出基础 reward 标量。
        - 如果 completion_ids == gold_completion_ids → 给这个基础 reward；
          否则 reward 为 0。
        - reward 只打在最后一个有效 response token 上。
        """

        # 如果上游已经给了 rm_scores，就直接用（保持兼容）
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        responses = data.batch["responses"]            # [B, resp_len]
        attention_mask = data.batch["attention_mask"]  # [B, prompt_len + resp_len]
        prompts = data.batch["prompts"]                # [B, prompt_len]

        device = responses.device
        batch_size, resp_len = responses.shape

        # 初始化 reward_tensor，形状与 responses 一致
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32, device=device)

        reward_extra_info = defaultdict(list)

        already_print_data_sources: dict[Any, int] = {}

        base_rewards: List[float] = []
        completion_list: List[List[int]] = []
        gold_completion_list: List[List[int]] = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]           # [prompt_len]
            attn = data_item.batch["attention_mask"]          # [prompt_len + resp_len]
            response_ids = data_item.batch["responses"]       # [resp_len]

            prompt_len = prompt_ids.shape[-1]
            # 有效 prompt 长度
            valid_prompt_len = int(attn[:prompt_len].sum().item())
            # 有效 response 长度
            valid_resp_len = int(attn[prompt_len:].sum().item())

            # 取有效的 completion token
            valid_response_ids = response_ids[:valid_resp_len]  # tensor [valid_resp_len]

            # gold completion ids 在 non_tensor_batch 里
            # 假设是 python list[int] 或 1D tensor
            gold_completion_ids = data_item.non_tensor_batch["gold_completion_ids"]
            if isinstance(gold_completion_ids, torch.Tensor):
                gold_completion_ids = gold_completion_ids.tolist()

            # 基础 reward 标量
            base_reward = float(data_item.non_tensor_batch["reward"])

            # 收集给 compute_score 使用
            base_rewards.append(base_reward)
            completion_list.append(valid_response_ids.tolist())
            gold_completion_list.append(gold_completion_ids)

        # ==== 使用 simple_reward 逻辑统一计算所有样本的最终 reward ====
        final_rewards: List[float] = self.compute_score(
            reward=base_rewards,
            completion_ids=completion_list,
            gold_completion_ids=gold_completion_list,
        )

        # ==== 把 reward 写回 reward_tensor====
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            attn = data_item.batch["attention_mask"]
            response_ids = data_item.batch["responses"]

            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = int(attn[:prompt_len].sum().item())
            valid_resp_len = int(attn[prompt_len:].sum().item())
            valid_response_ids = response_ids[:valid_resp_len]

            gold_completion_ids = data_item.non_tensor_batch["gold_completion_ids"]
            if isinstance(gold_completion_ids, torch.Tensor):
                gold_completion_ids = gold_completion_ids.tolist()

            # 最终 reward（已经带了 “是否匹配 gold” 的逻辑）
            reward_value = float(final_rewards[i])

            # 只在最后一个有效 response token 上打 reward
            if valid_resp_len > 0:
                reward_tensor[i, valid_resp_len - 1] = reward_value

            # ==== 构造一些可选的 extra info ====
            # 完整 prompt / gold / response 字符串（优先用原始字符串字段）
            prompt_str = data_item.non_tensor_batch.get(
                "prompt",
                self.tokenizer.decode(prompt_ids[-valid_prompt_len:], skip_special_tokens=True),
            )
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            gold_str = data_item.non_tensor_batch.get(
                "gold_completion",
                self.tokenizer.decode(gold_completion_ids, skip_special_tokens=True),
            )

            # data_source 可选：如果你没有这个字段，下面两行可以删除/换成别的 key
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "default")

            matched = completion_list[i] == gold_completion_list[i]

            reward_extra_info["base_reward"].append(base_rewards[i])
            reward_extra_info["final_reward"].append(reward_value)
            reward_extra_info["matched_gold"].append(matched)

            # 打印调试
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[data_source] {data_source}")
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[gold_completion]", gold_str)
                print("[base_reward]", base_rewards[i])
                print("[matched_gold]", matched)
                print("[final_reward]", reward_value)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor