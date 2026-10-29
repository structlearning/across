import operator
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import Generator
from torch import Tensor
from torch.optim import Optimizer

from reform.aug_embedder import AugmentedEmbedder
from reform.lsh import LSH
from reform.models import GEN, AugEmbeddingMap

from .gnn import GNNTrainer


class ActiveSelectionTrainer(GNNTrainer):
    def __init__(
        self,
        conf,
        model: nn.Module,
        optimizer: Optimizer,
        lsh: LSH,
        graph_model: GEN,
        q_embd: Tensor,
        c_embd: Tensor,
        q_embd_pre: Tensor,
        c_embd_pre: Tensor,
        aug_embedder: AugmentedEmbedder,
        aug_emb_model: AugEmbeddingMap,
        rng: Generator,
    ):
        super().__init__(conf, model, optimizer)
        self.lsh = lsh
        self.graph_model = graph_model
        self.q_embd = q_embd
        self.c_embd = c_embd
        self.q_embd_pre = q_embd_pre
        self.c_embd_pre = c_embd_pre
        self.aug_embedder = aug_embedder
        self.aug_emb_model = aug_emb_model
        self.rng = rng
        self.c_pre_normed = F.normalize(self.c_embd, dim=-1)
        # TODO: May change with more baselines, will have to turn this into a mode variable
        self.selection_method = (
            self.lsh_select
            if not self.conf.active_selection.random_selection
            else self.random_select
        )

    def lsh_select(self):
        use_oracle = self.conf.active_selection.oracle
        if use_oracle:
            idxs, q_graphs, gt = self.train_dataset.get_unkept_query_graphs_and_gt(
                self.conf.active_selection.num_query_samples, self.rng
            )
            d = self.conf.gmn.filters_3
            d_1 = self.conf.gmn.filters_3
            query_graph_node_sizes = torch.tensor([qg.number_of_nodes() for qg in q_graphs])
            query_graph_edge_sizes = torch.tensor([qg.number_of_edges() for qg in q_graphs])

            self.model.fetch_embed = True
            self.model.eval()
            q_embds = self.model(
                self.train_dataset._pack_batch_1d(q_graphs),
                query_graph_node_sizes,
                query_graph_edge_sizes,
            )
            q_embds_pre = self.model(
                self.train_dataset._pack_batch_1d(q_graphs),
                query_graph_node_sizes,
                query_graph_edge_sizes,
                pre_last_layer=True
            )
            self.model.fetch_embed = False

            foo = self.aug_embedder.get_embs(
                q_embds_pre,
                torch.zeros_like(q_embds_pre),  # dummy
                self.model.last_linear_layer_params.reshape(d, d_1),
                self.graph_model.last_linear_layer_params.reshape(d, d_1),
            )
            z_q, _ = self.aug_emb_model(*foo)

            cosine_sims = F.normalize(q_embds, dim=-1) @ self.c_pre_normed.T
            cosine_sims = cosine_sims.cpu()

            success = False
            to_keep = []
            for q_idx, query, scores, query_gt in zip(idxs, z_q, cosine_sims, gt, strict=True):
                c_ids = self.lsh.retrieve(query.cpu())
                sorted_c_ids = sorted(c_ids)
                top_k_idxs = torch.argsort(scores[sorted_c_ids], descending=True)[
                    : self.conf.active_selection.K
                ]
                top_k_c_ids = [sorted_c_ids[i] for i in top_k_idxs.tolist()]
                pos_c_ids = [cid for cid in top_k_c_ids if query_gt[cid]]
                neg_c_ids = [cid for cid in top_k_c_ids if not query_gt[cid]]
                if len(pos_c_ids) > 0:
                    success = True
                    to_keep.append((q_idx, pos_c_ids, neg_c_ids))

            for q_idx, pos_c_ids, neg_c_ids in sorted(
                to_keep, key=operator.itemgetter(0), reverse=True
            ):
                self.train_dataset.keep_query_graph(q_idx, pos_c_ids, neg_c_ids)
            return success
        raise NotImplementedError()

    def random_select(self):
        use_oracle = self.conf.active_selection.oracle
        if use_oracle:
            idxs, _, gt = self.train_dataset.get_unkept_query_graphs_and_gt(
                self.conf.active_selection.num_query_samples, self.rng
            )

            num_candidates = len(self.c_embd)

            success = False
            to_keep = []
            for q_idx, query_gt in zip(idxs, gt, strict=True):
                random_c_ids = self.rng.choice(
                    num_candidates, size=self.conf.active_selection.K, replace=False
                ).tolist()
                pos_c_ids = [cid for cid in random_c_ids if query_gt[cid]]
                neg_c_ids = [cid for cid in random_c_ids if not query_gt[cid]]
                if len(pos_c_ids) > 0:
                    success = True
                    to_keep.append((q_idx, pos_c_ids, neg_c_ids))

            for q_idx, pos_c_ids, neg_c_ids in sorted(
                to_keep, key=operator.itemgetter(0), reverse=True
            ):
                self.train_dataset.keep_query_graph(q_idx, pos_c_ids, neg_c_ids)
            return success
        raise NotImplementedError()

    def train_scalable(self):
        raise NotImplementedError()

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
            # msg = "will require saving queries added to dataset during active selection"
            # raise NotImplementedError(msg)

        self.best_val_ap = 0
        self.best_val_map = 0
        self.run = 0

        if self.conf.training.resume:
            self.resume()

        num_bad_selections = 0
        while num_bad_selections < self.conf.active_selection.patience:
            if not self.selection_method():
                num_bad_selections += 1
            else:
                num_bad_selections = 0
            for _ in range(self.conf.active_selection.num_iters):
                self.train_iter()
                self.validate_iter()
                self.run += 1

        self.post_train()
