import io
import json
import os
import random
import shutil
import sys
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler



from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import AutoTokenizer

class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True, force=False):
        if self.on or force:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()
def get_train_ds_config(
    offload,
    adam_offload=True,
    stage=2,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    disable_trace_cache=False,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # ZeRO++
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    if disable_trace_cache:
        zero_opt_dict["stage3_prefetch_bucket_size"] = 0
        zero_opt_dict["stage3_max_live_parameters"] = 0
        zero_opt_dict["stage3_max_reuse_distance"] = 0

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"},
    }


def get_eval_ds_config(
    offload,
    stage=0,
    bf16=True,
):
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]

def get_sp_tokens(args):
    sp_tokens = dict()
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        sp_token = getattr(args, key, None)
        if sp_token is not None:
            sp_tokens[key] = sp_token
    return sp_tokens

def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    sp_tokens = get_sp_tokens(strategy.args)

    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, **sp_tokens)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196


    if "mistral" in pretrain.lower():
        template_tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta', trust_remote_code=True)
        tokenizer.apply_chat_template = template_tokenizer.apply_chat_template

    elif "llama" in pretrain.lower():
        tempalte_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
        tokenizer.apply_chat_template = tempalte_tokenizer.apply_chat_template
        tokenizer.eos_token_id = 128001
        tokenizer.eos_token = '<|end_of_text|>'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def longest_common_substring_dp(str1, str2):
    len1, len2 = len(str1), len(str2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    max_len = 0
    end_pos = 0

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0

    return str1[end_pos - max_len: end_pos]
class DeepspeedStrategy(ABC):
    """
    The strategy for training with Accelerator.
    """

    def __init__(
            self,
            seed: int = 42,
            max_norm: float = 0.0,
            micro_train_batch_size=1,
            train_batch_size=1,
            zero_stage=2,
            bf16=True,
            args=None,
    ) -> None:
        super().__init__()

        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.bf16 = bf16
        self.adam_offload = args.adam_offload
        self.zpg = args.zpg
        self.seed = seed
        self.max_norm = max_norm
        self.grad_accum_dtype = args.grad_accum_dtype
        # disable_trace_cache
        self.disable_trace_cache = args.disable_trace_cache

        self.is_rlhf = False
        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)

        if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:  # for slurm
            self.args.local_rank = int(os.environ["LOCAL_RANK"])

        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size // self.world_size

    def create_optimizer(self, model, **kwargs) -> Optimizer:

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:

        model.backward(loss)

    def optimizer_step(
            self,
            optimizer: optim.Optimizer,
            model: nn.Module,
            scheduler,
            name="model",
            **kwargs,
    ) -> None:

        model.step()

    def setup_dataloader(
            self,
            replay_buffer,
            batch_size: int,
            pin_memory: bool = False,
            shuffle=True,
            collate_fn=None,
            drop_last=True,
    ):
        # DDP only mode, replay buffers on each rank are different.
        sampler = DistributedSampler(
            replay_buffer,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            seed=self.seed,
            drop_last=drop_last,
        )
        return DataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    def _unwrap_model(self, model) -> nn.Module:

        if hasattr(model, "module"):
            return model.module
        else:
            return model

    def prepare(
            self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        self.is_rlhf = is_rlhf
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._ds_init_train_model(*arg))
            else:
                ret.append(self._ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _ds_init_train_model(self, model, optim, scheduler):

        ds_config = self.get_ds_train_config()

        engine, optim, _, scheduler = deepspeed.initialize(
            model= model,
            optimizer=optim,
            lr_scheduler=scheduler,
            config=ds_config,
            args={"local_rank": self.args.local_rank},
            dist_init_required=True,
        )

        model = engine

        return model, optim, scheduler

    def get_ds_train_config(self):
        # DS Config
        ds_config = get_train_ds_config(
            # offload=False,
            offload=True,
            adam_offload=self.adam_offload,
            stage=self.stage,
            bf16=self.bf16,
            max_norm=self.max_norm,
            zpg=self.zpg,
            grad_accum_dtype=self.grad_accum_dtype,
            disable_trace_cache=self.disable_trace_cache,
        )

        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        train_batch_size = self.train_batch_size
        # corner case for ptx loss (backward twice)
        ds_config["train_batch_size"] = train_batch_size

        return ds_config

    def _ds_init_eval_model(self, model):

        ds_config = self.get_ds_eval_config(offload=getattr(model, "_offload", False))

        engine, *_ = deepspeed.initialize(
            model=model,
            args={"local_rank": self.args.local_rank},
            config=ds_config,
            dist_init_required=True,
        )

        model = engine
        return model

    def get_ds_eval_config(self, offload=False):
        # DS Config
        ds_config = get_eval_ds_config(offload=offload, stage=self.stage if self.stage == 3 else 0, bf16=self.bf16)
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        ds_config["train_batch_size"] = self.train_batch_size

        return ds_config

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % self.accumulated_gradient == 0:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
                        if self.stage != 3:
                            data = param.data.to(device)
                            param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)
                        else:
                            # TODO: use prefiltering for efficiency
                            params_to_fetch = _z3_params_to_fetch([param, param_ema])
                            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                                data = param.data.to(device)
                                param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)

    def load_model(
            self,
            model: nn.Module,
            path: str,
            map_location="cpu",
            strict: bool = False,
            key_replace_fn=None,
    ) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        # save model weights for ZeRO2/3
        model_to_save = self._unwrap_model(model)

        # gather parameters
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if self.is_rank_0():
                    output_state_dict[k] = vv

        if self.is_rank_0():
            # copy named_buffers
            for k, v in model_to_save.named_buffers():
                vv = v.data.cpu()
                output_state_dict[k] = vv

            # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir, **kwargs)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(output_dir, "adapter_model.bin"),
                    )
            else:
                # save model
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict, **kwargs)

            # save config
            output_config_file = os.path.join(output_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)
            # save tokenizer
            tokenizer.save_pretrained(output_dir)

            # for models not in AutoModel, copy python module files
            train_from_model_path = model_to_save.config._name_or_path
            if os.path.exists(train_from_model_path):
                for filename in os.listdir(train_from_model_path):
                    if filename.endswith(".py"):
                        shutil.copy(os.path.join(train_from_model_path, filename), os.path.join(output_dir, filename))

    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def rank_0_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.rank_0_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"
            if self.is_rank_0():
                ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            else:
                ret = None
        dist.gather(data.to(torch.cuda.current_device()), ret, dst=0)
        if self.is_rank_0():
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)
        else:
            return None

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        return dist.get_rank()

    def save_ckpt(self, model, save_dir, tag=None, max_num=3, max_mem=1000, client_state={}, save_latest=True):
        if self.is_rank_0():
            # Check and create the directory
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            # max hard drive space limit
            MAX_SIZE = max_mem * 1024 * 1024 * 1024

            while True:
                # Get all subdirectory and modification time
                subdirs = [
                    (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                    for d in os.listdir(save_dir)
                    if os.path.isdir(os.path.join(save_dir, d))
                ]
                # Sort by modification time, oldest first
                subdirs.sort(key=lambda x: x[1])
                # Calculate the total size of all sub -directory
                total_size = 0
                for subdir, _ in subdirs:
                    for dirpath, dirnames, filenames in os.walk(subdir):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)

                # If the number of subdire directors is greater than equal to max_num or the total size is greater than max_mem, the oldest Checkpoint is deleted
                if len(subdirs) >= max_num or total_size > MAX_SIZE:
                    oldest_dir, _ = subdirs[0]  # The oldest directory
                    if os.path.exists(oldest_dir):  # Ensure that the directory exists
                        shutil.rmtree(oldest_dir)  # Delete directory
                        self.print(f"Deleted oldest ckpt {oldest_dir}")  # The standard print function is used here
                else:
                    break

        assert isinstance(model, deepspeed.DeepSpeedEngine)
        model.save_checkpoint(save_dir, tag=tag, client_state=client_state, save_latest=save_latest)

    def load_ckpt(
            self,
            model,
            load_dir,
            tag=None,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_only=False,
    ):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        # basic ckpt: reuse deepspeed.DeepSpeedEngine.load_checkpoint
        return model.load_checkpoint(
            load_dir,
            tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
        )



def get_strategy(args):
    # default args for deepspeed
    if "seed" not in args:
        args.seed = 42
    if "max_norm" not in args:
        args.max_norm = 1.0
    if "micro_train_batch_size" not in args:
        args.micro_train_batch_size = 1
    if "train_batch_size" not in args:
        args.train_batch_size = 8
    if "local_rank" not in args:
        args.local_rank = -1
    if "bf16" not in args:
        args.bf16 = True
    if "adam_offload" not in args:
        args.adam_offload = False
    if "zpg" not in args:
        args.zpg = 1
    if "grad_accum_dtype" not in args:
        args.grad_accum_dtype = "fp32"

    strategy = DeepspeedStrategy(
        seed=args.seed,
        max_norm=args.max_norm,
        micro_train_batch_size=args.micro_train_batch_size,
        train_batch_size=args.train_batch_size,
        zero_stage=args.zero_stage,
        args=args,
    )

    return strategy

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
