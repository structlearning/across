import random
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from loguru import logger
from torch.optim import Optimizer

from ..utils.dataset_loader import SubgraphIsomorphismDataset
from ..utils.training_utils import evaluate_model_faster, pairwise_ranking_loss
from .base import Trainer


class GNNTrainer(Trainer):
    def __init__(self, conf, model, optimizer):
        super().__init__(conf, model, optimizer)

    def train_iter(self):
        self.model.train()

        use_celoss = self.conf.training.loss == 'celoss'
        if use_celoss or not self.conf.training.stratified_batching:
            n_batches = self.train_dataset.create_simple_batches()
        else:
            n_batches = self.train_dataset.create_stratified_batches()
        sub_factor = self.conf.early_stopping.patience // 2
        num_sub_batches = n_batches // sub_factor

        ce_criterion = nn.BCEWithLogitsLoss() if use_celoss else None
        validate_every_n_steps = self.conf.early_stopping.validate_every_n_steps

        # For legacy scalable mode (sub-epoch validation derived from n_batches/patience)
        if self.conf.training.scalable and validate_every_n_steps is None:
            sub_factor = self.conf.early_stopping.patience // 2
            num_sub_batches = max(1, n_batches // sub_factor)

        self.epoch_loss = 0
        training_start_time = time.time()

        for batch_idx in tqdm.tqdm(range(n_batches)):
            batch_graphs, batch_graph_node_sizes, batch_graph_edge_sizes, labels = \
                self.train_dataset.fetch_batch_by_id(batch_idx)

            self.optimizer.zero_grad()
            prediction = self.model(batch_graphs, batch_graph_node_sizes, batch_graph_edge_sizes)

            if use_celoss:
                losses = ce_criterion(prediction.view(-1), labels)
            else:
                predictions_for_positives = prediction[labels > 0.5]
                predictions_for_negatives = prediction[labels < 0.5]
                losses = pairwise_ranking_loss(
                    predictions_for_positives.unsqueeze(1),
                    predictions_for_negatives.unsqueeze(1),
                    self.conf.training.margin
                )
            losses.backward()
            self.optimizer.step()
            self.epoch_loss += losses.item()
            self.global_step += 1

            if validate_every_n_steps is not None and self.global_step % validate_every_n_steps == 0:
                self.epoch_training_time = time.time() - training_start_time
                logger.info(f"Step: {self.global_step} train loss: {self.epoch_loss} Time: {self.epoch_training_time}")
                self.validate_iter()
                self.model.train()
                if self.es.should_stop_now:
                    break
                self.run += 1
                self.epoch_loss = 0
                training_start_time = time.time()

            elif self.conf.training.scalable and validate_every_n_steps is None and \
                    ((batch_idx + 1) % num_sub_batches == 0 or (batch_idx + 1) == n_batches):
                self.epoch_training_time = time.time() - training_start_time
                logger.info(f"Run: {self.run} train loss: {self.epoch_loss} Time: {self.epoch_training_time}")
                self.validate_iter()
                self.model.train()
                if self.es.should_stop_now:
                    break
                self.run += 1
                self.epoch_loss = 0
                training_start_time = time.time()

        if not self.conf.training.scalable and validate_every_n_steps is None:
            self.epoch_training_time = time.time() - training_start_time
            logger.info(f"Run: {self.run} train loss: {self.epoch_loss} Time: {self.epoch_training_time}")

    def _validate(self, which="val"):
        if which == "val":
            dataset = self.val_dataset
        elif which == "test":
            dataset = self.test_dataset

        ap_score, map_score = evaluate_model_faster(self.conf, self.model, dataset)
        return ap_score, map_score

    def validate_iter(self):
        self.model.eval()

        logger.info("Validating")
        ap_score, map_score = self._validate(which="val")
        logger.info(f"Run: {self.run} VAL ap_score: {ap_score} map_score: {map_score}")

        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "epoch": self.run,
            "best_val_ap": self.best_val_ap,
            "best_val_map": self.best_val_map,
            "val_ap_score": ap_score,
            "val_map_score": map_score,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'random_state': random.getstate(),
            'patience': self.es.patience,
            'best_scores': self.es.best_scores,
            'num_bad_epochs': self.es.num_bad_epochs,
            'should_stop_now': self.es.should_stop_now,
        }

        if self.conf.early_stopping.metric == 'ap_score':
            metric_score = ap_score
        elif self.conf.early_stopping.metric == 'map_score':
            metric_score = map_score
        else:
            raise ValueError(f"Unsupported early stopping metric: {self.conf.early_stopping.metric}")

        state_dict = self.es.check([metric_score], state_dict)
        self.best_val_ap = state_dict["best_val_ap"]
        self.best_val_map = state_dict["best_val_map"]
        logger.info(f"Run: {self.run} best_val_ap: {self.best_val_ap} best_val_map: {self.best_val_map}")

        if self.wandb_enabled:
            wandb.log({
                "train_loss": self.epoch_loss,
                "train_time": self.epoch_training_time,
                "val_ap_score": ap_score,
                "val_map_score": map_score,
                "best_val_ap": self.best_val_ap,
                "best_val_map": self.best_val_map,
            })

    def post_train(self):
        ckpt = self.es.load_best_model()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        logger.info("Testing")
        test_ap_score, test_map_score = self._validate(which="test")
        logger.info(f"Run: run TEST ap_score: {test_ap_score} map_score: {test_map_score}")

        if self.wandb_enabled:
            wandb.log({
                "test_ap_score": test_ap_score,
                "test_map_score": test_map_score,
            })
