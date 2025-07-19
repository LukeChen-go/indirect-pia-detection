import json
import math
from abc import ABC
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm
# from transformers.trainer import get_scheduler


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class ExtractionSFTTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
            self,
            model,
            strategy,
            optim: Optimizer,
            train_dataloader,
            eval_dataloader,
            scheduler,
            max_norm: float = 1,
            pretrain_mode: bool = False,
            batch_size: int = 1,
            max_epochs: int = 2,
            tokenizer=None,
            logger=None
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.loss_fn = GPTLMLoss()
        self.logger = logger

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # wandb setting
        self._wandb = None


    def fit(self, args):
        # get eval and save steps

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            loss_mean = 0
            for inputs, attention_masks, labels in self.train_dataloader:
                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())
                output = self.model(inputs, attention_mask=attention_mask)
                labels = labels.to(torch.cuda.current_device())

                gpt_loss = self.loss_fn(output.logits, labels)

                bos_labels = labels.clone()
                bos_mask  = bos_labels != self.loss_fn.IGNORE_INDEX
                bos_position = bos_mask.float().argmax(dim=-1).view(-1,1)
                bos_mask = torch.ones_like(bos_labels).scatter(index=bos_position, dim=1, value=0)
                bos_labels[bos_mask.bool()] = self.loss_fn.IGNORE_INDEX
                bos_loss = self.loss_fn(output.logits, bos_labels)


                eos_labels = labels.clone()
                labels_mask = eos_labels != self.tokenizer.eos_token_id
                eos_labels[labels_mask] = self.loss_fn.IGNORE_INDEX
                eos_loss = self.loss_fn(output.logits, eos_labels)


                loss = gpt_loss + eos_loss + bos_loss
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict = {"gpt_loss": gpt_loss.item(), "loss_mean": loss_mean}
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1

            epoch_bar.update()

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

    def eval(self):
        self.model.eval()
        for batch in tqdm(self.eval_dataloader):
            batch = (b.to(self.model.device) for b in batch)
            input_ids, attention_masks = batch
            generation_config = self.model.generation_config
            generation_config.max_length = 8192
            generation_config.max_new_tokens = 32
            generation_config.do_sample = False
            generation_config.temperature = 0.0
            output = self.model.generate(
                input_ids,
                generation_config=generation_config
            )
            response = self.tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            response = response.strip()
            print(response)
        self.model.generation_config.do_sample = True  # to avoid a bug





class CLSSFTTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
            self,
            model,
            strategy,
            optim: Optimizer,
            train_dataloader,
            is_injection_eval_dataloader,
            no_injection_eval_dataloader,
            scheduler,
            max_norm: float = 1,
            pretrain_mode: bool = False,
            batch_size: int = 1,
            max_epochs: int = 2,
            tokenizer=None,
            logger=None
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.is_injection_eval_dataloader = is_injection_eval_dataloader
        self.no_injection_eval_dataloader = no_injection_eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.loss_fn = nn.CrossEntropyLoss()
        self.logger = logger

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # wandb setting
        self._wandb = None


    def fit(self, args):
        # get eval and save steps

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            loss_mean = 0
            for inputs, attention_masks, labels in self.train_dataloader:
                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())
                output = self.model(inputs, attention_mask=attention_mask)
                labels = labels.to(torch.cuda.current_device())
                target_position = (labels != -100).float().argmax(-1) - 1
                target_position = target_position.view(-1,1)
                target_mask = torch.zeros_like(inputs).scatter(dim=-1, src=torch.ones_like(inputs), index=target_position)
                target_logits = output.logits[target_mask.bool()]
                target_labels = inputs.gather(dim=-1, index=target_position+1).view(-1)
                # gpt_loss = self.loss_fn(output.logits, labels)
                gpt_loss = self.loss_fn(target_logits, target_labels)
                loss = gpt_loss
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict = {"gpt_loss": gpt_loss.item(), "loss_mean": loss_mean}
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1

            epoch_bar.update()

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

    def eval(self, is_injection, position):
        eval_dataloader = self.is_injection_eval_dataloader if is_injection \
            else self.no_injection_eval_dataloader
        label = 0 if not is_injection else 1
        check = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                batch = (b.to(self.model.device) for b in batch)
                input_ids, attention_masks = batch
                output = self.model(input_ids, attention_mask=attention_masks)
                logits = output.logits
                for attention_mask in attention_masks:
                    attention_mask[:attention_mask.sum() - 1] = 0
                selected_logits = logits[attention_masks.bool()]
                selected_position = torch.tensor(position).unsqueeze(0).repeat(logits.shape[0], 1).to(
                    self.model.device)
                target_logits = selected_logits.gather(index=selected_position, dim=-1)
                pre = target_logits.argmax(dim=-1).view(-1)
                check += (pre == label).tolist()
        acc = self.strategy.all_reduce(sum(check) / len(check))
        if self.strategy.is_rank_0():
            self.logger.log(f"Is injected:{is_injection}, Acc:{acc}")













class HeadTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
            self,
            model,
            strategy,
            optim: Optimizer,
            train_dataloader,
            scheduler,
            max_norm: float = 1,
            pretrain_mode: bool = False,
            batch_size: int = 1,
            max_epochs: int = 2,
            tokenizer=None,
            logger=None
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.loss_fn = nn.CrossEntropyLoss()
        self.logger = logger

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # wandb setting
        self._wandb = None


    def fit(self, args):
        # get eval and save steps

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            loss_mean = 0
            for inputs, attention_masks, labels in self.train_dataloader:
                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())
                output = self.model(inputs, attention_mask=attention_mask)
                labels = labels.to(torch.cuda.current_device())

                gpt_loss = self.loss_fn(output.logits, labels)
                # gpt_loss = self.loss_fn(target_logits, target_labels)
                loss = gpt_loss
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict = {"gpt_loss": gpt_loss.item(), "loss_mean": loss_mean}
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1

            epoch_bar.update()

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)






