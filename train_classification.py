import argparse
import os
import random

import numpy as np
from tqdm import tqdm

# os.environ['HF_HOME'] = './'
# # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# access_token = ""
# os.environ['HF_TOKEN'] =access_token
import sys

import torch
import math

from transformers import AutoModelForCausalLM, get_scheduler

from sft_trainer import CLSSFTTrainer

sys.path.append('..')
import utils
from generation_dataset import ClassficationDataset, ClassificationEvalDataset




def train(args):
    strategy = utils.get_strategy(args)
    strategy.setup_distributed()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    logger = utils.Logger(args.log_file)
    if strategy.is_rank_0():
        logger.log(str(args))
    tokenizer = utils.get_tokenizer(args.model_name_or_path, model, "right", strategy)


    train_dataset = ClassficationDataset(args.instruction_train_data_path, args.context_train_data_path,
                                         tokenizer=tokenizer, inject_rate=args.inject_rate, head_rate=args.head_rate,
                                         tail_rate=args.tail_rate)
    is_injection_eval_dataset = ClassificationEvalDataset(args.eval_data_path, tokenizer=tokenizer, is_inject=True)
    no_injection_eval_dataset = ClassificationEvalDataset(args.eval_data_path, tokenizer=tokenizer, is_inject=False)
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    train_dataloader = strategy.setup_dataloader(
        train_dataset, args.micro_train_batch_size, True, True, train_dataset.collate_fn
    )
    is_injection_eval_dataloader = strategy.setup_dataloader(is_injection_eval_dataset, batch_size=args.eval_batch_size,
                                                             shuffle=False, collate_fn=is_injection_eval_dataset.collate_fn)
    no_injection_eval_dataloader = strategy.setup_dataloader(no_injection_eval_dataset, batch_size=args.eval_batch_size,
                                                             shuffle=False, collate_fn=no_injection_eval_dataset.collate_fn)

    num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # if args.filter_method == "generation":
        #     model.gradient_checkpointing_enable()
        # elif isinstance(model, ModelForTokenClassification):
        #     model.model.gradient_checkpointing_enable()
        # else:
        #     raise ValueError("Only support generation or classification model")

    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))
    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = CLSSFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        is_injection_eval_dataloader=is_injection_eval_dataloader,
        no_injection_eval_dataloader=no_injection_eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        logger=logger

    )

    trainer.fit(args)
    trainer.eval(True, is_injection_eval_dataset.position)
    trainer.eval(False, no_injection_eval_dataset.position)
    strategy.save_model(model, tokenizer, args.save_path)
    # if args.filter_method == "generation":
    #     strategy.save_model(model, tokenizer, args.save_path)
    # else:
    #     strategy.save_model(model.model, tokenizer, args.save_path)
    #     if strategy.is_rank_0():
    #         torch.save(strategy._unwrap_model(model).classifier.state_dict(), os.path.join(args.save_path, "classifier.pt"))
def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="bigscience/bloomz-1b7")
    parser.add_argument("--instruction_train_data_path", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--context_train_data_path", type=str, default="Dahoas/full-hh-rlhf")

    parser.add_argument("--eval_data_path", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")

    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=list, default=None)

    parser.add_argument("--bos_token", type=str, default=None)
    parser.add_argument("--eos_token", type=str, default=None)
    parser.add_argument("--pad_token", type=str, default=None)
    parser.add_argument("--unk_token", type=str, default=None)

    parser.add_argument("--log_file", type=str, default="./logs/0130-1721.txt")
    parser.add_argument("--train_fn_type", type=str, default="insert")
    parser.add_argument("--test_fn_type", type=str, default="insert")
    parser.add_argument("--add_initial_parameters", action="store_true", default=False)
    parser.add_argument("--initial_model", type=str, default="gpt-xl/")
    parser.add_argument("--eval_dataset", type=str, default="cais/mmlu")
    parser.add_argument("--augment_eos", action="store_true", default=False)
    parser.add_argument("--filter_method", type=str, default="generation")
    parser.add_argument("--attack_method", type=str, default="ignore")
    parser.add_argument("--inject_rate", type=float, default=0.5)
    parser.add_argument("--head_rate", type=float, default=0.25)
    parser.add_argument("--tail_rate", type=float, default=0.25)

    args = parser.parse_args()

    set_seeds(args)
    train(args)
