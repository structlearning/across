import random
from abc import ABC, abstractmethod

import numpy as np
import pyfiglet
import torch
import torch.nn as nn
import tqdm
from loguru import logger
from torch.optim import Optimizer

import wandb

from ..utils.dataset_loader import SubgraphIsomorphismDataset
from .early_stopping import EarlyStoppingModule


class Trainer(ABC):
    def __init__(self, conf, model: nn.Module, optimizer: Optimizer, datasets=None):
        self.conf = conf
        self.wandb_enabled = conf.task.wandb_enabled

        self.model = model
        self.optimizer = optimizer
        self.device = conf.training.device

        if datasets is not None:
            train_dataset, val_dataset, test_dataset = datasets
        elif conf.dataset.rel_mode == "sub_iso":
            train_dataset = SubgraphIsomorphismDataset(conf, mode="train")
            val_dataset = SubgraphIsomorphismDataset(conf, mode="val")
            test_dataset = SubgraphIsomorphismDataset(conf, mode="test")
        else:
            raise NotImplementedError(f"rel_mode '{conf.dataset.rel_mode}' is not supported")

        logger.info(self.model)
        logger.info(f"no. of params in model: {sum(p.numel() for p in self.model.parameters())}")

        if self.wandb_enabled and self.conf.training.wandb_watch:
            wandb.watch(self.model, log_freq=1)

        self.es = EarlyStoppingModule(
            self.conf.base_dir,
            run_id=self.conf.task.name,
            patience=self.conf.early_stopping.patience,
            delta=self.conf.early_stopping.tolerance,
            best_val_name=self.conf.early_stopping.best_val_name,
            logger=logger,
            full_run_id=getattr(self.conf.task, 'full_name', self.conf.task.name),
        )

        logger.info(f"This uses the {self.conf.model.classPath}.{self.conf.model.name} model")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        conf.actual_max_node_set_size = max(
            self.train_dataset.max_node_set_size,
            self.val_dataset.max_node_set_size,
            self.test_dataset.max_node_set_size,
        )
        conf.actual_max_edge_set_size = max(
            self.train_dataset.max_edge_set_size,
            self.val_dataset.max_edge_set_size,
            self.test_dataset.max_edge_set_size,
        )

    def resume(self):
        banner = pyfiglet.figlet_format('Resuming training', font="slant", justify="center")
        logger.info(banner)

        checkpoint = self.es.load_latest_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.run = checkpoint['epoch'] + 1

        use_simple = (self.conf.training.loss == 'celoss' or not self.conf.training.stratified_batching)
        batch_desc = 'simple' if use_simple else 'stratified'
        for _ in tqdm.tqdm(range(self.run), desc=f'Discarding {batch_desc} batches (reproducibility)'):
            if use_simple:
                self.train_dataset.create_simple_batches()
            else:
                self.train_dataset.create_stratified_batches()

        torch.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['random_state'])

        try:
            self.best_val_ap = float(checkpoint['best_val_ap'])
            self.best_val_map = float(checkpoint['best_val_map'])
        except KeyError as e:
            logger.info(f"Could not load best_val_ap, best_val_map from checkpoint; Setting to 0: {e}")

        self.run = checkpoint['epoch'] + 1
        logger.info(f"Resuming training from epoch {self.run} with best val map: {self.best_val_map:.6f} and patience: {self.es.patience}")

    def train(self):
        if not self.conf.training.resume:
            self.es.save_initial_state({
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'np_rng_state': np.random.get_state(),
                'random_state': random.getstate(),
                'patience': self.es.patience,
                'best_scores': self.es.best_scores,
                'num_bad_epochs': self.es.num_bad_epochs,
                'should_stop_now': self.es.should_stop_now,
            })

        self.best_val_ap = 0
        self.best_val_map = 0
        self.run = 0
        self.global_step = 0

        if self.conf.training.resume:
            self.resume()

        while self.conf.training.run_till_early_stopping and self.run < self.conf.training.num_epochs:
            self.train_iter()
            self.validate_iter()

            if self.es.should_stop_now:
                break
            self.run += 1

        self.post_train()

    @abstractmethod
    def train_iter(self): ...

    def train_scalable(self):
        if not self.conf.training.resume:
            self.es.save_initial_state({
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'np_rng_state': np.random.get_state(),
                'random_state': random.getstate(),
                'patience': self.es.patience,
                'best_scores': self.es.best_scores,
                'num_bad_epochs': self.es.num_bad_epochs,
                'should_stop_now': self.es.should_stop_now,
            })

        self.best_val_ap = 0
        self.best_val_map = 0
        self.run = 0
        self.global_step = 0

        if self.conf.training.resume:
            self.resume()

        while self.run < self.conf.training.num_epochs:
            self.train_iter()

            if self.conf.training.run_till_early_stopping and self.es.should_stop_now:
                break

        self.post_train()

    @abstractmethod
    def _validate(self, which): ...

    @abstractmethod
    def validate_iter(self): ...

    @abstractmethod
    def post_train(self): ...
