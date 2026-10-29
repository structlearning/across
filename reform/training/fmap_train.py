import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from loguru import logger
from torch.func import vmap
from sklearn.metrics import average_precision_score

import wandb
from reform.fmap_embedder import FMapEmbedder
from reform.utils.model_utils import gen_map_from_embeds

from .base import Trainer


class FMapTrainer(Trainer):
    def __init__(
        self, conf, model, optimizer, gmn_config, graph_model, graph_model_best, datasets=None,
    ):
        super().__init__(conf, model, optimizer, datasets=datasets)

        self.gmn_config = gmn_config
        self.graph_model = graph_model
        self.graph_model_best = graph_model_best
        self.fmap_embedder = FMapEmbedder(conf).to(conf.training.device)

    def train_iter(self):
        self.model.train()

        n_batches = self.train_dataset.create_stratified_batches()
        sub_factor = self.conf.early_stopping.patience // 2
        num_sub_batches = n_batches // sub_factor

        d = self.conf.gmn.filters_3
        d_1 = self.conf.gmn.filters_3
        mlp_out_dim = self.conf.fmap_embedder.output_dim
        lamb_m = self.conf.fmap_embedder.lambda_times
        u_n = self.conf.fmap_embedder.u_times

        theta_0 = self.graph_model.last_linear_layer_params
        criterion = nn.BCELoss(reduction="none")

        self.epoch_loss = 0
        training_start_time = time.time()

        for batch_idx in tqdm.tqdm(range(n_batches)):
            self.optimizer.zero_grad()

            with torch.no_grad():
                batch_graphs, batch_graph_node_sizes, batch_graph_edge_sizes, labels = \
                    self.train_dataset.fetch_batch_by_id(batch_idx)
                query_emb, corpus_emb = self.graph_model(
                    batch_graphs, batch_graph_node_sizes, batch_graph_edge_sizes,
                    fetch_split=True, pre_last_layer=True)

            B = query_emb.size(0)
            q_shape = c_shape = (B, mlp_out_dim) if self.conf.fmap_embedder.flat_mlp else (B, d, mlp_out_dim)

            if self.conf.training.loss == "ce-avg":
                lambda_tensor = torch.rand(lamb_m, device=self.device)
                u_tensor = F.normalize(
                    torch.randn((u_n, d * d_1), device=self.device), p=2, dim=-1
                )
                theta_scale = theta_0.norm(p=2)

                def loss_for_pair(lamb, u, theta_0=theta_0, theta_scale=theta_scale,
                                  query_emb=query_emb, corpus_emb=corpus_emb, labels=labels):
                    theta_curr = theta_0 + lamb * u * theta_scale
                    return self.get_loss(
                        corpus_emb, criterion, d, d_1, labels, query_emb, theta_curr, q_shape, c_shape
                    )

                all_losses = vmap(
                    lambda lamb, u_tensor=u_tensor: vmap(lambda u: loss_for_pair(lamb, u))(u_tensor)
                )(lambda_tensor)

                loss = all_losses.mean()

            elif self.conf.training.loss == "ce-adv-max":
                epsilon = self.conf.fmap_embedder.pgd_epsilon
                delta = (
                    F.normalize(torch.randn_like(theta_0), dim=-1)
                    * epsilon
                    * torch.rand(1, device=theta_0.device)
                ).requires_grad_(True)

                self.model.requires_grad_(False)
                for _ in range(self.conf.fmap_embedder.num_pgd_steps):
                    delta.grad = None
                    theta_curr = theta_0 + delta
                    loss = self.get_loss(
                        corpus_emb, criterion, d, d_1, labels, query_emb, theta_curr, q_shape, c_shape
                    )
                    loss.backward()

                    with torch.no_grad():
                        delta_new = delta + self.conf.fmap_embedder.pgd_step_size * delta.grad
                        delta_norm = delta_new.norm(p=2)
                        if delta_norm > epsilon:
                            delta_new = epsilon * delta_new / delta_norm
                        delta = delta_new.detach().requires_grad_(True)
                self.model.requires_grad_(True)

                theta_curr = (theta_0 + delta).detach()
                loss = self.get_loss(
                    corpus_emb, criterion, d, d_1, labels, query_emb, theta_curr, q_shape, c_shape
                )
            else:
                msg = f"Loss {self.conf.training.loss} not implemented"
                raise NotImplementedError(msg)

            loss.backward()
            self.optimizer.step()
            self.epoch_loss += loss.item()

            if self.wandb_enabled:
                wandb.log({"train_loss_batch": loss.item()})

            if self.conf.training.scalable and \
                    ((batch_idx + 1) % num_sub_batches == 0 or (batch_idx + 1) == n_batches):
                self.epoch_training_time = time.time() - training_start_time
                logger.info(
                    f"Run: {self.run} train loss: {self.epoch_loss} Time: {self.epoch_training_time}"
                )
                self.validate_iter()
                self.model.train()
                if self.es.should_stop_now:
                    break
                self.run += 1
                self.epoch_loss = 0
                training_start_time = time.time()

        if not self.conf.training.scalable:
            self.epoch_training_time = time.time() - training_start_time
            logger.info(
                f"Run: {self.run} train loss: {self.epoch_loss} Time: {self.epoch_training_time}"
            )

    def get_hat_embs(self, corpus_emb, d, d_1, query_emb, theta_curr, q_shape, c_shape):
        q_hat_batch, x_hat_batch = self.fmap_embedder.get_embs(
            query_emb, corpus_emb, theta_curr.reshape(d, d_1)
        )
        if self.conf.fmap_embedder.flat_mlp:
            q_hat_batch = q_hat_batch.reshape(q_hat_batch.size(0), -1)
            x_hat_batch = x_hat_batch.reshape(x_hat_batch.size(0), -1)
        q_hat_batch_mapped, x_hat_batch_mapped = self.model(q_hat_batch, x_hat_batch)
        assert q_hat_batch_mapped.shape == q_shape
        assert x_hat_batch_mapped.shape == c_shape
        return q_hat_batch_mapped, x_hat_batch_mapped

    def get_loss(self, corpus_emb, criterion, d, d_1, labels, query_emb, theta_curr, q_shape, c_shape):
        q_hat, c_hat = self.get_hat_embs(
            corpus_emb, d, d_1, query_emb, theta_curr, q_shape, c_shape
        )
        cos_sim = (1.0 + F.cosine_similarity(q_hat, c_hat, dim=-1)) / 2.0
        if self.conf.fmap_embedder.flat_mlp:
            return criterion(cos_sim, labels).mean()
        labels_exp = labels[:, None].expand_as(cos_sim)
        loss = criterion(cos_sim, labels_exp)
        return loss.sum(dim=-1).mean()

    def _validate_common(self, which="val", preserve_dim=False):
        if which == "val":
            dataset = self.val_dataset
        elif which == "test":
            dataset = self.test_dataset
        elif which == "train":
            dataset = self.train_dataset

        d = self.conf.gmn.filters_3
        d_1 = self.conf.gmn.filters_3

        self.graph_model_best.fetch_embed = True
        q_embeds = self.graph_model_best(
            dataset._pack_batch_1d(dataset.query_graphs),
            dataset.query_graph_node_sizes, dataset.query_graph_edge_sizes, pre_last_layer=True
        )
        c_embeds = self.graph_model_best(
            dataset._pack_batch_1d(dataset.corpus_graphs),
            dataset.corpus_graph_node_sizes, dataset.corpus_graph_edge_sizes, pre_last_layer=True
        )
        self.graph_model_best.fetch_embed = False

        W_t = self.graph_model_best.last_linear_layer_params.reshape(d, d_1)

        out_dim = self.conf.fmap_embedder.output_dim
        if self.conf.fmap_embedder.flat_mlp:
            q_shape = (q_embeds.size(0), out_dim)
            c_shape = (c_embeds.size(0), out_dim)
        else:
            q_shape = (q_embeds.size(0), d, out_dim)
            c_shape = (c_embeds.size(0), d, out_dim)
        q_hat, c_hat = self.get_hat_embs(
            c_embeds, d, d_1, query_emb=q_embeds, theta_curr=W_t, q_shape=q_shape, c_shape=c_shape
        )

        if preserve_dim:
            return q_hat, c_hat, dataset.all_gt

        if not self.conf.fmap_embedder.flat_mlp:
            q_hat = q_hat.sum(dim=1)
            c_hat = c_hat.sum(dim=1)
        return q_hat, c_hat, dataset.all_gt

    def _validate(self, which="val"):
        q_hat, c_hat, all_gt = self._validate_common(which)
        map_score, all_scores = gen_map_from_embeds(self.conf, q_hat, c_hat, all_gt)
        ap_score = average_precision_score(all_gt, all_scores)
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

        state_dict = self.es.check([map_score], state_dict)
        self.best_val_ap = state_dict["best_val_ap"]
        self.best_val_map = state_dict["best_val_map"]
        logger.info(
            f"Run: {self.run} best_val_ap: {self.best_val_ap} best_val_map: {self.best_val_map}"
        )

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
