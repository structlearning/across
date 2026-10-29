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
from reform.aug_embedder import AugmentedEmbedder
from reform.utils.model_utils import gen_map_from_embeds

from .base import Trainer


class AugEmbTrainer(Trainer):
    def __init__(
        self, conf, model, optimizer, gmn_config, graph_model, graph_model_best, datasets=None,
        q_embeds_0=None, c_embeds_0=None
    ):
        super().__init__(conf, model, optimizer, datasets=datasets)

        self.gmn_config = gmn_config
        self.graph_model = graph_model
        self.graph_model_best = graph_model_best
        self.aug_embedder = AugmentedEmbedder(conf)

        self.q_embeds_0 = q_embeds_0
        self.c_embeds_0 = c_embeds_0

    def train_iter(self):
        self.model.train()

        n_batches = self.train_dataset.create_stratified_batches()
        sub_factor = self.conf.early_stopping.patience // 2
        num_sub_batches = n_batches // sub_factor

        # TODO: Verify that filters_3 is the right dim
        d = self.conf.gmn.filters_3
        d_1 = self.conf.gmn.filters_3
        mlp_out_dim = self.conf.aug_embedder.output_dim
        lamb_m = self.conf.aug_embedder.lambda_times
        u_n = self.conf.aug_embedder.u_times

        # Get the params of the final linear layer on graph_model.
        # theta_0 dim: d.d_1
        theta_0 = self.graph_model.last_linear_layer_params
        criterion = nn.BCELoss(reduction="none")

        self.epoch_loss = 0
        training_start_time = time.time()

        for batch_idx in tqdm.tqdm(range(n_batches)):
            self.optimizer.zero_grad()

            if self.conf.aug_embedder.use_pretrained_embs:
                q_idxs, c_idxs, labels = \
                    self.train_dataset.fetch_batch_by_id(batch_idx)
                query_emb = self.q_embeds_0["train"][q_idxs].to(self.device)
                corpus_emb = self.c_embeds_0["train"][c_idxs].to(self.device)

            else:
                with torch.no_grad():
                    batch_graphs, batch_graph_node_sizes, batch_graph_edge_sizes, labels = \
                        self.train_dataset.fetch_batch_by_id(batch_idx)
                    query_emb, corpus_emb = self.graph_model(
                        batch_graphs, batch_graph_node_sizes, batch_graph_edge_sizes,
                        fetch_split=True, pre_last_layer=True)

            B = query_emb.size(0)
            q_shape = c_shape = (B, d, mlp_out_dim)

            if self.conf.training.loss == "ce-avg":
                lambda_tensor = torch.rand(lamb_m, device=self.device)                     # (L,)
                u_tensor = F.normalize(
                    torch.randn((u_n, d * d_1), device=self.device), p=2, dim=-1
                )                                                                          # (U, P)

                theta_scale = theta_0.norm(p=2)

                def loss_for_pair(lamb, u, theta_0=theta_0, theta_scale=theta_scale,
                                  query_emb=query_emb, corpus_emb=corpus_emb,
                                  labels=labels):
                    theta_curr = theta_0 + lamb * u * theta_scale
                    return self.get_loss(
                        corpus_emb,
                        criterion,
                        d,
                        d_1,
                        labels,
                        query_emb,
                        theta_0,
                        theta_curr,
                        q_shape, c_shape
                    )

                # First vmap over u for a fixed lambda, then over lambda
                # XXX: Be careful about vmap + lambda function, there can be lots of gotchas if not
                # careful, esp around variable binding.
                all_losses = vmap(
                    lambda lamb, u_tensor=u_tensor: vmap(lambda u: loss_for_pair(lamb, u))(u_tensor)
                )(lambda_tensor)                                                                  # (L, U)

                loss = all_losses.mean()

            elif self.conf.training.loss == "ce-adv-max":
                epsilon = self.conf.aug_embedder.pgd_epsilon
                delta = (
                    F.normalize(torch.randn_like(theta_0), dim=-1)
                    * epsilon
                    * torch.rand(1, device=theta_0.device)
                ).requires_grad_(True)

                self.model.requires_grad_(False)
                for _ in range(self.conf.aug_embedder.num_pgd_steps):
                    delta.grad = None
                    theta_curr = theta_0 + delta
                    loss = self.get_loss(
                        corpus_emb,
                        criterion,
                        d,
                        d_1,
                        labels,
                        query_emb,
                        theta_0,
                        theta_curr,
                        q_shape, c_shape
                    )
                    loss.backward()

                    with torch.no_grad():
                        delta_new = delta + self.conf.aug_embedder.pgd_step_size * delta.grad
                        delta_norm = delta_new.norm(p=2)
                        if delta_norm > epsilon:
                            delta_new = epsilon * delta_new / delta_norm
                        delta = delta_new.detach().requires_grad_(True)
                self.model.requires_grad_(True)

                theta_curr = (theta_0 + delta).detach()
                loss = self.get_loss(
                    corpus_emb,
                    criterion,
                    d,
                    d_1,
                    labels,
                    query_emb,
                    theta_0,
                    theta_curr,
                    q_shape, c_shape
                )
            else:
                msg = f"Loss {self.conf.training.loss} not implemented"
                raise NotImplementedError(msg)

            loss.backward()
            self.optimizer.step()
            self.epoch_loss += loss.item()

            if self.wandb_enabled:
                wandb.log({
                    "train_loss_batch": loss.item(),
                })

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

    def get_hat_embs(self, corpus_emb, d, d_1, query_emb, theta_0, theta_curr, q_shape, c_shape,
                     theta_0_q=None, pre_augnet=False):
        q_hat_batch, x_hat_batch = self.aug_embedder.get_embs(
            query_emb, corpus_emb, theta_curr.reshape(d, d_1), theta_0.reshape(d, d_1),
            init_params_q=theta_0_q.reshape(d, d_1) if theta_0_q is not None else None,
        )
        if self.conf.aug_embedder.push_minus:
            q_hat_batch = -q_hat_batch
        if pre_augnet:
            return q_hat_batch, x_hat_batch
        q_hat_batch_mapped, x_hat_batch_mapped = self.model(q_hat_batch, x_hat_batch)
        assert q_hat_batch_mapped.shape == q_shape
        assert x_hat_batch_mapped.shape == c_shape
        return q_hat_batch_mapped, x_hat_batch_mapped

    def get_loss(
        self,
        corpus_emb,
        criterion,
        d,
        d_1,
        labels,
        query_emb,
        theta_0,
        theta_curr,
        q_shape, c_shape
    ):
        q_hat, c_hat = self.get_hat_embs(corpus_emb, d, d_1, query_emb, theta_0, theta_curr, q_shape, c_shape)
        cos_sim = (1.0 + F.cosine_similarity(q_hat, c_hat, dim=-1)) / 2.0
        labels_exp = labels[:, None].expand_as(cos_sim)
        loss = criterion(cos_sim, labels_exp)

        return loss.sum(dim=-1).mean()

    def _validate_common(self, which="val", preserve_dim=False, query_graph_model=None,
                         corpus_graph_model=None, override_W0_q=None, pre_augnet=False):
        if which == "val":
            dataset = self.val_dataset
        elif which == "test":
            dataset = self.test_dataset
        elif which == "train":
            dataset = self.train_dataset

        query_gm = query_graph_model if query_graph_model is not None else self.graph_model_best
        corpus_gm = corpus_graph_model if corpus_graph_model is not None else self.graph_model

        query_gm.fetch_embed = True
        corpus_gm.fetch_embed = True
        if query_gm is not self.graph_model_best:
            self.graph_model_best.fetch_embed = True  # needed for corpus when aug_corpus_model == "current"
        d = self.conf.gmn.filters_3
        d_1 = self.conf.gmn.filters_3
        q_embeds = query_gm(dataset._pack_batch_1d(dataset.query_graphs),
            dataset.query_graph_node_sizes, dataset.query_graph_edge_sizes, pre_last_layer=True)

        if self.conf.aug_embedder.aug_corpus_model == "init":
            c_embeds = corpus_gm(dataset._pack_batch_1d(dataset.corpus_graphs),
                dataset.corpus_graph_node_sizes, dataset.corpus_graph_edge_sizes, pre_last_layer=True)
        elif self.conf.aug_embedder.aug_corpus_model == "current":  # XXX: UNUSED
            c_embeds = self.graph_model_best(dataset._pack_batch_1d(dataset.corpus_graphs),
                dataset.corpus_graph_node_sizes, dataset.corpus_graph_edge_sizes, pre_last_layer=True)

        query_gm.fetch_embed = False
        corpus_gm.fetch_embed = False
        if query_gm is not self.graph_model_best:
            self.graph_model_best.fetch_embed = False

        W_t = query_gm.last_linear_layer_params.reshape(d, d_1)
        W_0 = self.graph_model.last_linear_layer_params.reshape(d, d_1)

        q_shape = (q_embeds.size(0), d, self.conf.aug_embedder.output_dim)
        c_shape = (c_embeds.size(0), d, self.conf.aug_embedder.output_dim)
        q_hat, c_hat = self.get_hat_embs(c_embeds, d, d_1, query_emb=q_embeds, theta_0=W_0,
                                         theta_curr=W_t, q_shape=q_shape, c_shape=c_shape,
                                         theta_0_q=override_W0_q, pre_augnet=pre_augnet)
        if preserve_dim or pre_augnet:
            return q_hat, c_hat, dataset.all_gt

        # B x d x output_dim -> B x output_dim
        q_hat = q_hat.sum(dim=1)
        c_hat = c_hat.sum(dim=1)

        return q_hat, c_hat, dataset.all_gt

    def _validate(self, which="val"):
        q_hat, c_hat, all_gt = self._validate_common(which)
        map_score, all_scores = gen_map_from_embeds(self.conf, q_hat, c_hat, all_gt)
        ap_score = average_precision_score(all_gt, all_scores)

        return ap_score, map_score

    # XXX: validate_iter and post_train are the same as in GNNTrainer, can be moved to a mixin
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
