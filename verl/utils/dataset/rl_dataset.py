# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
# from verl.actor_rollout.vllm.utils import get_response_mask
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        print("Downloading dataset...")
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        print("reading files and tokenizing...")
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            print("appending a parquet file")
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    images = (
                        [process_image(image) for image in doc[image_key]]
                        if image_key in doc and doc[image_key]
                        else None
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]]
                        if video_key in doc and doc[video_key]
                        else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                        )
                    )
                
                def doc2len_2(doc) -> int:
                    prompt_key = "prompt_ids"
                    # print(len(doc[prompt_key]))
                    return len(
                            doc[prompt_key]
                    )
                def doc2len_3(doc) -> int:
                    return len(doc["gold_completion_ids"])

            dataframe = dataframe.filter(
                lambda doc: doc2len_2(doc) <= self.max_prompt_length and (doc2len_3(doc) < 4096),
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

class OfflineSweDataset(RLHFDataset):
    """
    """

    def __init__(self, data_files, tokenizer, config, processor: Optional[ProcessorMixin] = None):
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=processor)

        self.truncation = self.config.get("truncation", "error")
        self.max_prompt_length = self.config.get("max_prompt_length", 1024)

        self.offline_max_response_length = self.config.get(
            "offline_max_response_length",
            self.max_prompt_length,
        )
        self.offline_max_total_length = self.config.get(
            "offline_max_total_length",
            self.max_prompt_length + self.offline_max_response_length,
        )

    def __getitem__(self, idx: int) -> dict:
        row = self.dataframe[idx]

        # ===== 1. prompt：沿用原 RLHFDataset 的逻辑，左 pad 到 max_prompt_length =====
        prompt_ids = torch.tensor(row["prompt_ids"], dtype=torch.long).unsqueeze(0)  # [1, Lp]
        prompt_attn = torch.ones_like(prompt_ids)  # [1, Lp]

        prompt_ids_pad, prompt_attn_pad = verl_F.postprocess_data(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        prompt_ids_pad = prompt_ids_pad[0]      # [max_prompt_length]
        prompt_attn_pad = prompt_attn_pad[0]    # [max_prompt_length]

        # prompt-only 版本，直接作为 prompts 送进 DataProto
        prompts = prompt_ids_pad.clone()        # [max_prompt_length]

        # 计算有效 prompt token 数（去掉左 pad）
        prompt_valid_len = int(prompt_attn_pad.sum().item())
        prompt_tokens = prompt_ids_pad[-prompt_valid_len:]  # [prompt_valid_len]

        # ===== 2. response：从 gold_completion_ids 构造，保持与 generate_sequences 一致 =====
        resp_ids_raw = torch.tensor(row["gold_completion_ids"], dtype=torch.long)  # [R_raw]
        resp_len_raw = int(resp_ids_raw.size(0))

        # 使用与 online rollout 相同的 response_length
        # 注意：这里假设 data.max_response_length == actor_rollout_ref.rollout.response_length
        response_length = self.config.max_response_length

        # 先截断到 response_length
        if resp_len_raw > response_length:
            resp_ids = resp_ids_raw[:response_length]
        else:
            resp_ids = resp_ids_raw

        # 再右 pad 到 response_length（和 pad_2d_list_to_length 一样）
        if resp_ids.size(0) < response_length:
            pad_r = response_length - resp_ids.size(0)
            pad_tokens = torch.full((pad_r,), self.tokenizer.pad_token_id, dtype=torch.long)
            responses = torch.cat([resp_ids, pad_tokens], dim=0)  # [response_length]
        else:
            responses = resp_ids  # [response_length]

        # ===== 3. 构造完整 input_ids：左 pad prompt + 右 pad response =====
        # 对齐 generate_sequences 里的 seq = torch.cat([idx, response], dim=-1)
        input_ids_full = torch.cat([prompts, responses], dim=0)  # [max_prompt_length + response_length]

        # ===== 4. attention_mask：prompt 原样 + get_response_mask =====
        # prompt 部分 mask 就用 prompt_attn_pad
        attention_prompt = prompt_attn_pad  # [max_prompt_length]

        # get_response_mask: 根据 eos 截断 response 的有效部分，后面（含 pad）为 0
        response_attention_mask = get_response_mask(
            response_id=responses.unsqueeze(0),  # [1, response_length]
            eos_token=self.tokenizer.eos_token_id,
            dtype=attention_prompt.dtype,
        )[0]  # [response_length]

        attention_full = torch.cat(
            [attention_prompt, response_attention_mask], dim=0
        )  # [max_prompt_length + response_length]

        # ===== 5. position_ids：先算 prompt，再按“最后一个 + delta”扩展 =====
        position_prompt = compute_position_id_with_mask(
            attention_prompt.unsqueeze(0)
        )[0]  # [max_prompt_length]

        # response 部分 position_ids：最后一个 prompt 的 position + 1,2,3,...
        delta_position_id = torch.arange(
            1,
            response_length + 1,
            device=position_prompt.device,
            dtype=position_prompt.dtype,
        )  # [response_length]
        last_pos = position_prompt[-1]  # 标量
        response_position_ids = last_pos + delta_position_id  # [response_length]

        position_full = torch.cat(
            [position_prompt, response_position_ids], dim=0
        )  # [max_prompt_length + response_length]

        # ===== 6. response_mask：只对 response 段，形状 = [response_length] =====
        # ★★★ CHANGED: 原来是做成 [max_prompt_length + response_length]，现在改成与 entropys 对齐的 [response_length]
        # entropys / log_probs 的形状是 (bs, response_length)，所以 response_mask 也必须是 (bs, response_length)
        response_mask = (response_attention_mask > 0).long()   # [response_length]

        # ===== 7. rollout_log_probs：同样只对 response 段，形状 = [response_length] =====
        # ★★★ CHANGED: 在线 generate_sequences 里 rollout_log_probs 的 shape 也是 (bs, response_length)，
        # 所以这里也保持 [response_length]，而不是整段 seq_len。
        if "logprobs" in row and row["logprobs"] is not None:
            lp = torch.tensor(row["logprobs"], dtype=torch.float)  # [R_raw]
            # 截断到 response_length
            lp = lp[:response_length]
            # 右 pad 到 response_length（与 responses 对齐）
            if lp.size(0) < response_length:
                pad_lp = torch.zeros(response_length - lp.size(0), dtype=torch.float)
                lp = torch.cat([lp, pad_lp], dim=0)
        else:
            lp = torch.zeros(response_length, dtype=torch.float)

        rollout_log_probs = lp  # [response_length]  # ★★★ CHANGED: 不再扩展到 [max_prompt_length + response_length]

        # ===== 8. 组装 row_dict：tensor 字段 + 原始非 tensor 字段 =====
        row_dict = dict(row)

        # tensor 字段：进入 DataProto.batch
        row_dict["prompts"] = prompts                      # [max_prompt_length]，prompt-only
        row_dict["responses"] = responses                  # [response_length]，右 pad 后的 response
        row_dict["input_ids"] = input_ids_full             # [max_prompt_length + response_length]
        row_dict["attention_mask"] = attention_full        # [max_prompt_length + response_length]
        row_dict["position_ids"] = position_full           # [max_prompt_length + response_length]
        row_dict["response_mask"] = response_mask          # [response_length]  ★★★ 对齐 compute_log_prob
        row_dict["rollout_log_probs"] = rollout_log_probs  # [response_length]  ★★★ 对齐在线 rollout

        # 非 tensor 字段（messages / request_id / repo / instance_id / turn / prompt / gold_completion / reward ...）
        # 保留 HF 中原本的类型即可，DataProto.from_single_dict 会自动放到 non_tensor_batch
        return row_dict
