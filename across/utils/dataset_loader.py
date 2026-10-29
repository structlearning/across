import collections
import json
import math
import os
import pickle
import random

import networkx as nx
import numpy as np
import torch
from loguru import logger
from networkx import Graph
from numpy.random import Generator
from numpy.typing import NDArray
from torch import Tensor
from torch_geometric.data import Data

TRAIN_MODE = "train"
VAL_MODE = "val"
TEST_MODE = "test"

GraphCollection = collections.namedtuple(
    'GraphCollection',
    ['from_idx', 'to_idx', 'node_features', 'edge_features', 'graph_idx', 'num_graphs']
)


class DatasetBase:
    def __init__(self, conf, mode):
        assert mode in [TRAIN_MODE, VAL_MODE, TEST_MODE]
        self.mode = mode
        # self.conf = conf
        self.dataset_name = conf.dataset.name
        self.init_dataset_stats(conf)
        self.data_type = conf.dataset.data_type

        self.batch_size = conf.training.batch_size
        self.device = conf.training.device

        self.load_graphs(conf)
        # self.build_adjacency_info()
        self.preprocess_subgraphs_to_pyG_data()
        self.print_dataset_stats()

        self.pos_to_neg_ratio: float | None = None
        self.pos_pairs: list[tuple[int, int]]
        self.neg_pairs: list[tuple[int, int]]
        self.query_graphs: list[Graph]
        self.corpus_graphs: list[Graph]
        self.unkept_query_graphs: list[Graph]
        self.relationships: dict[int, dict[str, list[int]]]
        self.unkept_relationships: list[dict[str, list[int]]]
        self.all_gt: Tensor
        self.unkept_gt: list[Tensor]

    def init_dataset_stats(self, conf):
        raise NotImplementedError()

    def load_graphs(self, conf):
        raise NotImplementedError()

    def keep_query_graph(self, idx: int, pos_c_ids: list[int], neg_c_ids: list[int]):
        """Makes the query graph a candidate for being sampled in batches in the future.

        When not using the oracle for getting the true labels (i.e., training
        using pseudo labels), pass new_gt to modify the ground truth for the
        given query graph.
        """
        # XXX: UNUSED
        new_idx = len(self.query_graphs)

        graph = self.unkept_query_graphs.pop(idx)
        self.query_graphs.append(graph)

        rel = self.unkept_relationships.pop(idx)
        self.relationships[new_idx] = rel

        for c_id in pos_c_ids:
            self.pos_pairs.append((new_idx, c_id))
        for c_id in neg_c_ids:
            self.neg_pairs.append((new_idx, c_id))

        self.query_graph_node_sizes = torch.cat((
            self.query_graph_node_sizes,
            torch.tensor([self.query_graphs[new_idx].number_of_nodes()], device=self.query_graph_node_sizes.device),
        ))
        self.query_graph_edge_sizes = torch.cat((
            self.query_graph_edge_sizes,
            torch.tensor([self.query_graphs[new_idx].number_of_edges()], device=self.query_graph_edge_sizes.device),
        ))

        gt = self.unkept_gt.pop(idx)
        self.all_gt = torch.cat((self.all_gt, gt))

    def get_unkept_query_graph_idxs(self, K: int, rng: Generator) -> NDArray:
        # XXX: UNUSED
        num_unkept_graphs = len(self.unkept_query_graphs)
        return rng.choice(num_unkept_graphs, min(num_unkept_graphs, K), replace=False)

    def get_unkept_query_graphs(self, K: int, rng: Generator) -> GraphCollection:
        """Returns K unkept query graphs, sampled uniformly at random from all
        unkept graphs.

        If there are less than K unkept query graphs, all unkept query graphs
        are returned.
        """
        # XXX: UNUSED
        return self._pack_batch_1d([
            self.unkept_query_graphs[i] for i in self.get_unkept_query_graph_idxs(K, rng)
        ])

    def get_unkept_query_graphs_and_gt(
        self, K: int, rng: Generator
    ) -> tuple[NDArray, list[Graph], list[Tensor]]:
        """Returns K unkept query graphs and their ground truths. The graphs are
        sampled uniformly at random from all unkept graphs.

        If there are less than K unkept query graphs, all unkept query graphs
        and their ground truths are returned.
        """
        # XXX: UNUSED
        idxs = self.get_unkept_query_graph_idxs(K, rng)
        return (
            idxs,
            [self.unkept_query_graphs[i] for i in idxs],
            [self.unkept_gt[i] for i in idxs],
        )

    def create_pyG_object(self, graph):
        num_nodes = graph.number_of_nodes()
        features = torch.ones(num_nodes, 1, dtype=torch.float, device=self.device)

        edges = list(graph.edges)
        doubled_edges = [[x, y] for (x, y) in edges]  # + [[y, x] for (x, y) in edges]
        edge_index = torch.tensor(np.array(doubled_edges).T, dtype=torch.int64, device=self.device)
        return Data(x=features, edge_index=edge_index), num_nodes

    def preprocess_subgraphs_to_pyG_data(self):
        self.query_graph_data, self.query_graph_sizes = zip(
            *[self.create_pyG_object(query_graph) for query_graph in self.query_graphs]
        )
        self.corpus_graph_data, self.corpus_graph_sizes = zip(
            *[self.create_pyG_object(corpus_graph) for corpus_graph in self.corpus_graphs]
        )
        self.query_graph_data = list(self.query_graph_data)
        self.corpus_graph_data = list(self.corpus_graph_data)

    # def build_adjacency_info(self):
    #     def adj_list_from_graph_list(graphs):
    #         adj_list = []
    #         for graph in graphs:
    #             unpadded_adj = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float, device=self.device)
    #             assert unpadded_adj.shape[0] == unpadded_adj.shape[1]
    #             num_nodes = len(unpadded_adj)
    #             padded_adj = F.pad(unpadded_adj, pad = (0, self.max_node_set_size - num_nodes, 0, self.max_node_set_size - num_nodes))
    #             adj_list.append(padded_adj)
    #         return adj_list

    #     self.query_adj_list = adj_list_from_graph_list(self.query_graphs)
    #     self.corpus_adj_list = adj_list_from_graph_list(self.corpus_graphs)

    def _pack_batch_1d(self, graphs):
        from_idx = []
        to_idx = []
        graph_idx = []
        # all_graphs = [individual_graph for graph_tuple in graphs for individual_graph in graph_tuple]
        all_graphs = graphs

        total_nodes, total_edges = 0, 0
        for idx, graph in enumerate(all_graphs):
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edges = np.array(graph.edges(), dtype=np.int32)

            from_idx.append(edges[:, 0] + total_nodes)
            to_idx.append(edges[:, 1] + total_nodes)
            graph_idx.append(np.ones(num_nodes, dtype=np.int32) * idx)

            total_nodes += num_nodes
            total_edges += num_edges

        return GraphCollection(
            from_idx=torch.tensor(np.concatenate(from_idx, axis=0), dtype=torch.int64, device=self.device),
            to_idx=torch.tensor(np.concatenate(to_idx, axis=0), dtype=torch.int64, device=self.device),
            graph_idx=torch.tensor(np.concatenate(graph_idx, axis=0), dtype=torch.int64, device=self.device),
            num_graphs=len(all_graphs),
            node_features=torch.ones(total_nodes, 1, dtype=torch.float, device=self.device),
            edge_features=torch.ones(total_edges, 1, dtype=torch.float, device=self.device)
        )

    def _pack_batch(self, graphs):
        from_idx = []
        to_idx = []
        graph_idx = []
        all_graphs = [individual_graph for graph_tuple in graphs for individual_graph in graph_tuple]

        total_nodes, total_edges = 0, 0
        for idx, graph in enumerate(all_graphs):
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edges = np.array(graph.edges(), dtype=np.int32)

            from_idx.append(edges[:, 0] + total_nodes)
            to_idx.append(edges[:, 1] + total_nodes)
            graph_idx.append(np.ones(num_nodes, dtype=np.int32) * idx)

            total_nodes += num_nodes
            total_edges += num_edges

        return GraphCollection(
            from_idx=torch.tensor(np.concatenate(from_idx, axis=0), dtype=torch.int64, device=self.device),
            to_idx=torch.tensor(np.concatenate(to_idx, axis=0), dtype=torch.int64, device=self.device),
            graph_idx=torch.tensor(np.concatenate(graph_idx, axis=0), dtype=torch.int64, device=self.device),
            num_graphs=len(all_graphs),
            node_features=torch.ones(total_nodes, 1, dtype=torch.float, device=self.device),
            edge_features=torch.ones(total_edges, 1, dtype=torch.float, device=self.device)
        )

    def create_stratified_batches(self, query_graph_idx=None):
        self.batch_setting = 'pair_batches'

        if query_graph_idx is None:
            pos_pairs = self.pos_pairs
            neg_pairs = self.neg_pairs
        else:
            pos_pairs = [(query_graph_idx, corpus_graph_idx) for corpus_graph_idx in self.relationships[query_graph_idx]['pos']]
            neg_pairs = [(query_graph_idx, corpus_graph_idx) for corpus_graph_idx in self.relationships[query_graph_idx]['neg']]

        random.shuffle(pos_pairs), random.shuffle(neg_pairs)
        if self.pos_to_neg_ratio is None:
            # This is set once, in the first call to create_stratified_batches.
            #
            # After that, the same ratio is used, even when the
            # pos_pairs:neg_pairs ratio changes due to active sampling adding
            # more positive (negative) pairs than negative (positive) pairs
            self.pos_to_neg_ratio = len(pos_pairs) / len(neg_pairs)
        pos_to_neg_ratio = self.pos_to_neg_ratio

        num_pos_per_batch = math.ceil(pos_to_neg_ratio / (1 + pos_to_neg_ratio) * self.batch_size)
        num_neg_per_batch = self.batch_size - num_pos_per_batch

        batches_pos, batches_neg = [], []
        labels_pos, labels_neg = [], []
        for idx in range(0, len(pos_pairs), num_pos_per_batch):
            elements_remaining = len(pos_pairs) - idx
            elements_chosen = min(num_pos_per_batch, elements_remaining)
            batches_pos.append(pos_pairs[idx : idx + elements_chosen])
            labels_pos.append([1.0] * elements_chosen)
        for idx in range(0, len(neg_pairs), num_neg_per_batch):
            elements_remaining = len(neg_pairs) - idx
            elements_chosen = min(num_neg_per_batch, elements_remaining)
            batches_neg.append(neg_pairs[idx : idx + elements_chosen])
            labels_neg.append([0.0] * elements_chosen)

        if query_graph_idx is None:
            self.num_batches = min(len(batches_pos), len(batches_neg))
            self.batches = [pos + neg for (pos, neg) in zip(batches_pos[:self.num_batches], batches_neg[:self.num_batches])]
            self.labels = [pos + neg for (pos, neg) in zip(labels_pos[:self.num_batches], labels_neg[:self.num_batches])]

        else:
            # Accumulate batches and labels for all query graphs in this case.
            self.num_batches = self.num_batches + min(len(batches_pos), len(batches_neg))
            self.batches.extend([pos + neg for (pos, neg) in zip(batches_pos[:self.num_batches], batches_neg[:self.num_batches])])
            self.labels.extend([pos + neg for (pos, neg) in zip(labels_pos[:self.num_batches], labels_neg[:self.num_batches])])

        return self.num_batches

    def create_simple_batches(self):
        """Non-stratified batching: shuffle all pairs together and slice into batches of batch_size.
        Each pair appears exactly once per epoch. No pos-neg ratio enforcement."""
        self.batch_setting = 'pair_batches'

        all_pairs = self.pos_pairs + self.neg_pairs
        all_labels = [1.0] * len(self.pos_pairs) + [0.0] * len(self.neg_pairs)

        combined = list(zip(all_pairs, all_labels))
        random.shuffle(combined)
        all_pairs_shuf, all_labels_shuf = zip(*combined)

        self.batches = []
        self.labels = []
        for idx in range(0, len(all_pairs_shuf), self.batch_size):
            self.batches.append(list(all_pairs_shuf[idx:idx + self.batch_size]))
            self.labels.append(list(all_labels_shuf[idx:idx + self.batch_size]))

        self.num_batches = len(self.batches)
        return self.num_batches

    def create_eval_batches(self, pair_list):
        self.batch_setting = 'eval'
        self.eval_batches = []
        # for idx in range(0, len(pair_list), self.batch_size):
            # self.batches.append(pair_list[idx : idx + self.batch_size])
        # bsz = self.num_corpus_graphs//10
        bsz = self.num_corpus_graphs
        for idx in range(0, len(pair_list), bsz):
            self.eval_batches.append(pair_list[idx : idx + bsz])

        self.num_eval_batches = len(self.eval_batches)
        return self.num_eval_batches

    def fetch_batch_by_id(self, idx):
        if self.batch_setting == 'pair_batches':
            assert idx < self.num_batches
            batch = self.batches[idx]
        elif self.batch_setting == 'eval':
            assert idx < self.num_eval_batches
            batch = self.eval_batches[idx]

        query_graph_idxs, corpus_graph_idxs = zip(*batch)

        if self.data_type == "gmn":
            query_graphs = [self.query_graphs[idx] for idx in query_graph_idxs]
            corpus_graphs = [self.corpus_graphs[idx] for idx in corpus_graph_idxs]
            all_graphs = self._pack_batch(zip(query_graphs, corpus_graphs))
        elif self.data_type == "pyg":
            query_graphs = [self.query_graph_data[idx] for idx in query_graph_idxs]
            corpus_graphs = [self.corpus_graph_data[idx] for idx in corpus_graph_idxs]
            all_graphs = query_graphs + corpus_graphs  # list(zip(query_graphs, corpus_graphs))
        elif self.data_type == "idx":
            if self.batch_setting == 'pair_batches':
                target = torch.tensor(np.array(self.labels[idx]), dtype=torch.float, device=self.device)
            elif self.batch_setting == 'eval':
                target = None

            return list(query_graph_idxs), list(corpus_graph_idxs), target

        elif self.data_type.startswith("gmn_1d"):
            if self.batch_setting == 'pair_batches':
                target = torch.tensor(np.array(self.labels[idx]), dtype=torch.float, device=self.device)
            elif self.batch_setting == 'eval':
                target = None

            query_graphs = [self.query_graphs[idx] for idx in query_graph_idxs]
            corpus_graphs = [self.corpus_graphs[idx] for idx in corpus_graph_idxs]

            query_graphs_pkd = self._pack_batch_1d(query_graphs)
            corpus_graphs_pkd = self._pack_batch_1d(corpus_graphs)

            query_graph_node_sizes = self.query_graph_node_sizes[list(query_graph_idxs)]
            corpus_graph_node_sizes = self.corpus_graph_node_sizes[list(corpus_graph_idxs)]

            # model.fetch_embed = True
            # q_embeds = model(sampler._pack_batch_1d(sampler.query_graphs),sampler.query_graph_node_sizes,sampler.query_graph_edge_sizes)
            # c_embeds = model(sampler._pack_batch_1d(sampler.corpus_graphs),sampler.corpus_graph_node_sizes,sampler.corpus_graph_edge_sizes)
            # model.fetch_embed = False

            if self.data_type.endswith("_with_idx"):
                return query_graphs_pkd, corpus_graphs_pkd, query_graph_node_sizes, corpus_graph_node_sizes,\
                    list(query_graph_idxs), list(corpus_graph_idxs), target
            return query_graphs_pkd, corpus_graphs_pkd, query_graph_node_sizes, corpus_graph_node_sizes, target

        # query_graph_sizes = [self.query_graph_sizes[idx] for idx in query_graph_idxs]
        # corpus_graph_sizes = [self.corpus_graph_sizes[idx] for idx in corpus_graph_idxs]
        # all_graph_sizes = list(zip(query_graph_sizes, corpus_graph_sizes))

        query_graph_node_sizes = self.query_graph_node_sizes[list(query_graph_idxs)]
        corpus_graph_node_sizes = self.corpus_graph_node_sizes[list(corpus_graph_idxs)]
        all_graph_node_sizes = torch.dstack((query_graph_node_sizes, corpus_graph_node_sizes)).flatten()
        query_graph_edge_sizes = self.query_graph_edge_sizes[list(query_graph_idxs)]
        corpus_graph_edge_sizes = self.corpus_graph_edge_sizes[list(corpus_graph_idxs)]
        all_graph_edge_sizes = torch.dstack((query_graph_edge_sizes, corpus_graph_edge_sizes)).flatten()

        # query_graph_adjs = [self.query_adj_list[idx] for idx in query_graph_idxs]
        # corpus_graph_adjs = [self.corpus_adj_list[idx] for idx in corpus_graph_idxs]
        # all_graph_adjs = list(zip(query_graph_adjs, corpus_graph_adjs))

        if self.batch_setting == 'pair_batches':
            target = torch.tensor(np.array(self.labels[idx]), dtype=torch.float, device=self.device)
            return all_graphs, all_graph_node_sizes, all_graph_edge_sizes, target  # , all_graph_adjs
        if self.batch_setting == 'eval':
            return all_graphs, all_graph_node_sizes, all_graph_edge_sizes, None  # , all_graph_adjs
        raise NotImplementedError

    def print_dataset_stats(self):
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Number of query graphs: {len(self.query_graphs)}")
        logger.info(f"Number of corpus graphs: {len(self.corpus_graphs)}")
        # logger.info(f"Number of positive pairs: {len(self.pos_pairs)}")
        # logger.info(f"Number of negative pairs: {len(self.neg_pairs)}")
        logger.info(f"Maximum node set size: {self.max_node_set_size}")
        logger.info(f"Maximum edge set size: {self.max_edge_set_size}")
        r_list = []
        for k, v in self.relationships.items():
            r_list.append(len(v['pos']) / len(v['neg']))
        logger.info(f"Per Query Average pos to neg ratio: {np.mean(r_list)}")
        logger.info(f"Macro Average pos to neg ratio: {len(self.pos_pairs) / len(self.neg_pairs)}")


class SubgraphIsomorphismDataset(DatasetBase):
    def __init__(self, conf, mode):
        super().__init__(conf, mode)

    def init_dataset_stats(self, conf):
        # NOTE: this loads all graphs (tr+te+val). Deliberately not storing the graphs. Only size stats are stored.
        directory = f"{conf.base_dir}data/{conf.dataset.name}/preprocessed"
        q_fname = f"{directory}/relabeled_queries_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_relabeled.pkl"
        c_fname = f"{directory}/{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_relabeled.pkl"
        all_query_graphs = pickle.load(open(q_fname, 'rb'))
        all_corpus_graphs = pickle.load(open(c_fname, 'rb'))
        self.max_edge_set_size = max(
            max([graph.number_of_edges() for graph in all_query_graphs]),
            max([graph.number_of_edges() for graph in all_corpus_graphs])
        )
        self.max_node_set_size = max(
            max([graph.number_of_nodes() for graph in all_query_graphs]),
            max([graph.number_of_nodes() for graph in all_corpus_graphs])
        )

    def _filter_query_graphs(self, keep_mask, track_unkept: bool = False, skip_relationships: bool = False):
        """Filter query_graphs, relationships, and all_gt to only the kept queries.

        Args:
            keep_mask: Iterable of truthy/falsy values, one per query graph.
            track_unkept: If True, also populate self.unkept_query_graphs,
                          self.unkept_relationships, and self.unkept_gt.
            skip_relationships: If True, skip building self.relationships and
                                 self.all_gt (caller will set these themselves).
        """
        query_graphs: list[Graph] = []
        if not skip_relationships:
            all_gt = []
            relationships: dict[int, dict[str, list[int]]] = {}

        if track_unkept:
            unkept_query_graphs: list[Graph] = []
            unkept_relationships: list[dict[str, list[int]]] = []
            unkept_gt = []

        kept_idx = 0
        for g, keep, rel in zip(
            self.query_graphs, keep_mask, self.relationships.values(), strict=True
        ):
            if keep:
                query_graphs.append(g)
                if not skip_relationships:
                    gt = torch.zeros(self.num_corpus_graphs)
                    gt[rel["pos"]] = 1
                    relationships[kept_idx] = rel
                    all_gt.append(gt)
                kept_idx += 1
            elif track_unkept:
                gt = torch.zeros(self.num_corpus_graphs)
                gt[rel["pos"]] = 1
                unkept_query_graphs.append(g)
                unkept_relationships.append(rel)
                unkept_gt.append(gt)

        self.query_graphs = query_graphs
        if not skip_relationships:
            self.all_gt = torch.cat(all_gt)
            self.relationships = relationships

        if track_unkept:
            self.unkept_query_graphs = unkept_query_graphs
            self.unkept_relationships = unkept_relationships
            self.unkept_gt = unkept_gt

    def load_graphs(self, conf):
        directory = f"{conf.base_dir}{conf.dataset.path}/{conf.dataset.name}/preprocessed/splits"

        if self.mode == "train" and conf.training.oracle_dataset:
            logger.info(f"Using oracle dataset with oracle mode {conf.training.oracle_dataset} for training, ignoring original train-test-val splits and query ratio range settings")
            logger.info(f"Loading oracle dataset queries from {directory}train/oracle_50q, corpus from original")

            oracle_mode = conf.training.oracle_dataset  # "lsh" or "random"

            oracle_dir = os.path.join(directory, "train", "oracle_50q")
            c_fname = f"{directory}/../{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_relabeled.pkl"
            if oracle_mode.endswith("100"):
                q_fname = os.path.join(oracle_dir, "queries_100.pkl")
            else:
                q_fname = os.path.join(oracle_dir, "queries.pkl")

            if "abs" in oracle_mode:
                parts = oracle_mode.split("_")
                omode = parts[0]
                Kneg = f"_{parts[-2]}" if parts[-2] == "Kneg" else ""

                if parts[-1] == "100":
                    rel_fname = os.path.join(oracle_dir, f"{omode}_rel{Kneg}_100.pkl")
                else:
                    rel_fname = os.path.join(oracle_dir, f"{omode}_rel{Kneg}.pkl")
                logger.info(f"Using absolute oracle with rel file {rel_fname}")
            else:
                parts = oracle_mode.split("_")
                omode = parts[0]
                pct = f"{parts[1]}pct" if parts[1].isdigit() else "pct"

                if parts[-1] == "100":
                    rel_fname = os.path.join(oracle_dir, f"{omode}_rel{Kneg}_{pct}_100.pkl")
                else:
                    rel_fname = os.path.join(oracle_dir, f"{oracle_mode}_rel{Kneg}_{pct}.pkl")
                logger.info(f"Using percentage oracle with rel file {rel_fname}")

            self.query_graphs = pickle.load(open(q_fname, "rb"))
            self.corpus_graphs = pickle.load(open(c_fname, "rb"))
            self.num_corpus_graphs = len(self.corpus_graphs)
            self.relationships = pickle.load(open(rel_fname, "rb"))
            all_gt = []
            num_query_graphs = len(self.query_graphs)
            for qidx in range(num_query_graphs):
                gt = torch.zeros(self.num_corpus_graphs)
                gt[self.relationships[qidx]["pos"]] = 1
                all_gt.append(gt)
            self.all_gt = torch.cat(all_gt)
            self.query_graph_node_sizes = torch.tensor(
                [g.number_of_nodes() for g in self.query_graphs], device=self.device)
            self.corpus_graph_node_sizes = torch.tensor(
                [g.number_of_nodes() for g in self.corpus_graphs], device=self.device)
            self.query_graph_edge_sizes = torch.tensor(
                [g.number_of_edges() for g in self.query_graphs], device=self.device)
            self.corpus_graph_edge_sizes = torch.tensor(
                [g.number_of_edges() for g in self.corpus_graphs], device=self.device)
            self.query_graph_edges = [
                torch.tensor(np.array(g.edges()), dtype=torch.int64, device=self.device)
                for g in self.query_graphs
            ]
            self.corpus_graph_edges = [
                torch.tensor(np.array(g.edges()), dtype=torch.int64, device=self.device)
                for g in self.corpus_graphs
            ]
            self.pos_pairs, self.neg_pairs = [], []
            for query_idx in range(num_query_graphs):
                for corpus_idx in self.relationships[query_idx]["pos"]:
                    self.pos_pairs.append((query_idx, corpus_idx))
                for corpus_idx in self.relationships[query_idx]["neg"]:
                    self.neg_pairs.append((query_idx, corpus_idx))
            logger.info(f"Loaded oracle-{oracle_mode} dataset: {num_query_graphs} queries, {self.num_corpus_graphs} corpus graphs from {oracle_dir}")
            return

        # XXX: use_labeled is old code for the older active learning setup. We use it only to check and reproduce earlier experiments now.
        # use_clusters is the current code for the "active data collection" setup. In this case the val set is the same across clusters.
        use_labeled = self.mode == "train" and conf.training.labeled_subset
        load_mode = "test" if self.mode == "val" and conf.get("val_use_test_set", False) else self.mode
        use_clusters = load_mode in ["train", "test"] and conf.use_cluster is not None

        if self.mode == "train":
            qpart_fname = os.path.join(directory, self.mode, f"{self.mode}_qpart_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_relabeled_range_{conf.dataset.MinR}_to_{conf.dataset.MaxR}.pkl")
        else:
            qpart_fname = None

        if not os.path.exists(directory):
            os.makedirs(directory)
        q_fname = f"{directory}/{load_mode}/{load_mode}_queries_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_relabeled_range_{conf.dataset.MinR}_to_{conf.dataset.MaxR}.pkl"
        if conf.training.silver_top_K:
            # Postings list training where each query's ground truths are decided by the top-K silver (hinge distance) scores.
            rel_fname = f"{directory}/{load_mode}/{load_mode}_silver_{conf.training.silver_top_K}pct_rel_dict_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_range_{conf.dataset.MinR}_to_{conf.dataset.MaxR}.pkl"
        else:
            rel_fname = f"{directory}/{load_mode}/{load_mode}_rel_dict_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_range_{conf.dataset.MinR}_to_{conf.dataset.MaxR}.pkl"
        c_fname = f"{directory}/../{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_relabeled.pkl"

        if not os.path.exists(c_fname):
            raise FileNotFoundError(f"Corpus graphs not found at {c_fname}")

        if not os.path.exists(rel_fname) and conf.training.silver_top_K:
            raise FileNotFoundError(f"Silver relationships file for top-{conf.training.silver_top_K}% relationships not found at {rel_fname}")

        if use_labeled and not os.path.exists(qpart_fname):
            raise FileNotFoundError(f"Query partition file for labeled subset not found at {qpart_fname}")

        if not os.path.exists(q_fname) or not os.path.exists(rel_fname):
            # Generate splits for the specified range
            all_q_fname = f"{directory}/../relabeled_queries_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_relabeled.pkl"
            all_rel_fname = f"{directory}/../rel_dict_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}.pkl"
            logger.info(f"Loaded all queries from {all_q_fname} for the purpose of creating splits")
            logger.info(f"Loaded all relationships from {all_rel_fname}  for the purpose of creating splits")

            all_queries = pickle.load(open(all_q_fname, 'rb'))
            all_rels = pickle.load(open(all_rel_fname, 'rb'))

            logger.info(f"Filtering queries for required query ratio range in {conf.dataset.MinR} to {conf.dataset.MaxR}")
            relevant_queries = []
            relevant_rels = {}
            new_qidx = 0
            for idx, rel in all_rels.items():
                if len(rel['neg']) > 0 and conf.dataset.MinR <= len(rel['pos']) / len(rel['neg']) <= conf.dataset.MaxR:
                    relevant_queries.append(all_queries[idx])
                    relevant_rels[new_qidx] = rel
                    new_qidx += 1

            logger.info(f"Found {len(relevant_queries)} within specified range in {conf.dataset.MinR} to {conf.dataset.MaxR}")
            logger.info("Splitting into tr-test-val - 60:20:20")
            all_qidx = list(range(len(relevant_queries)))
            random.shuffle(all_qidx)
            tr_idx = all_qidx[:int(0.6 * len(all_qidx))]
            test_idx = all_qidx[int(0.6 * len(all_qidx)):int(0.8 * len(all_qidx))]
            val_idx = all_qidx[int(0.8 * len(all_qidx)):]
            tr_queries = [relevant_queries[idx] for idx in tr_idx]
            test_queries = [relevant_queries[idx] for idx in test_idx]
            val_queries = [relevant_queries[idx] for idx in val_idx]
            tr_rels = {new_idx: relevant_rels[idx] for new_idx, idx in enumerate(tr_idx)}
            test_rels = {new_idx: relevant_rels[idx] for new_idx, idx in enumerate(test_idx)}
            val_rels = {new_idx: relevant_rels[idx] for new_idx, idx in enumerate(val_idx)}

            logger.info(f"Saving tr-test-val queries to {directory}")
            os.makedirs(f"{directory}/train", exist_ok=True)
            os.makedirs(f"{directory}/test", exist_ok=True)
            os.makedirs(f"{directory}/val", exist_ok=True)

            tr_q_fname = f"{directory}/train/train_queries_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_relabeled_range_{conf.dataset.MinR}_to_{conf.dataset.MaxR}.pkl"
            test_q_fname = f"{directory}/test/test_queries_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_relabeled_range_{conf.dataset.MinR}_to_{conf.dataset.MaxR}.pkl"
            val_q_fname = f"{directory}/val/val_queries_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_relabeled_range_{conf.dataset.MinR}_to_{conf.dataset.MaxR}.pkl"
            pickle.dump(tr_queries, open(tr_q_fname, 'wb'))
            pickle.dump(test_queries, open(test_q_fname, 'wb'))
            pickle.dump(val_queries, open(val_q_fname, 'wb'))
            tr_rel_fname = f"{directory}/train/train_rel_dict_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_range_{conf.dataset.MinR}_to_{conf.dataset.MaxR}.pkl"
            test_rel_fname = f"{directory}/test/test_rel_dict_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_range_{conf.dataset.MinR}_to_{conf.dataset.MaxR}.pkl"
            val_rel_fname = f"{directory}/val/val_rel_dict_for_{conf.dataset.aug_num_cgraphs}_corpus_subgraphs_min{conf.dataset.min_cgraph_nodes}_max{conf.dataset.max_cgraph_nodes}_range_{conf.dataset.MinR}_to_{conf.dataset.MaxR}.pkl"
            pickle.dump(tr_rels, open(tr_rel_fname, 'wb'))
            pickle.dump(test_rels, open(test_rel_fname, 'wb'))
            pickle.dump(val_rels, open(val_rel_fname, 'wb'))

        self.query_graphs = pickle.load(open(q_fname, 'rb'))
        self.corpus_graphs = pickle.load(open(c_fname, 'rb'))

        self.num_corpus_graphs = len(self.corpus_graphs)
        assert self.num_corpus_graphs == conf.dataset.aug_num_cgraphs, f"{self.num_corpus_graphs} and {conf.dataset.aug_num_cgraphs} mismatch"

        self.relationships = pickle.load(open(rel_fname, 'rb'))
        if use_labeled:
            with open(qpart_fname, 'rb') as f:
                qpart = pickle.load(f)

            logger.info(f"There are {sum(qpart)} query graphs marked for keeping in the labeled subset out of {len(qpart)} total query graphs")
            assert len(qpart) == len(self.query_graphs)
            self._filter_query_graphs(qpart, track_unkept=True)

            assert len(self.query_graphs) == len(self.relationships), (
                f"{len(self.query_graphs)} and {len(self.relationships)} mismatch after filtering for labeled subset"
            )
            assert len(self.query_graphs) == sum(qpart), (
                f"{len(self.query_graphs)} and {sum(qpart)} mismatch after filtering for labeled subset"
            )
        elif use_clusters:
            assignment_path = os.path.join(directory, load_mode, "cluster_assignments.json")
            with open(assignment_path, "r") as f:
                cluster_assignments = json.load(f)

            keep_mask = [a == conf.use_cluster for a in cluster_assignments]
            load_cluster_rels = (
                self.mode == "train"
                and conf.use_cluster not in (None, 0)
                and conf.training.relationship_load != 'base'
            )
            self._filter_query_graphs(keep_mask, skip_relationships=load_cluster_rels)

            if load_cluster_rels:
                cluster_dir = os.path.join(directory, "train", "clusters", str(conf.use_cluster))
                if conf.cluster_rel_subdir is not None:
                    cluster_dir = os.path.join(cluster_dir, conf.cluster_rel_subdir)
                suffix = ""
                if conf.lsh.subtract_mean:
                    suffix += "_sub_mean"

                if conf.cluster_suffix is not None:
                    suffix += f"_{conf.cluster_suffix}"

                cluster_rel_fname = os.path.join(cluster_dir, f"{conf.training.relationship_load}_relationships{suffix}.pkl")
                self.relationships = pickle.load(open(cluster_rel_fname, 'rb'))
                all_gt = []
                for qidx in range(len(self.query_graphs)):
                    gt = torch.zeros(self.num_corpus_graphs)
                    gt[self.relationships[qidx]["pos"]] = 1
                    all_gt.append(gt)
                self.all_gt = torch.cat(all_gt)
                logger.info(f"Loaded cluster {conf.use_cluster} {conf.training.relationship_load} relationships from {cluster_rel_fname}")

            assert len(self.query_graphs) == len(self.relationships), (
                f"{len(self.query_graphs)} and {len(self.relationships)} mismatch after filtering for cluster {conf.use_cluster}"
            )

        else:
            all_gt = []  # for evaluation
            # FIXME: all_pairs is not updated in self.keep_query_graph()
            # self.all_pairs = []  # for evaluation
            for qidx in range(len(self.query_graphs)):
                gt = torch.zeros(self.num_corpus_graphs)
                gt[self.relationships[qidx]["pos"]] = 1
                all_gt.append(gt)
                # FIXME: all_pairs is not updated in self.keep_query_graph()
                # self.all_pairs.extend(list(zip([qidx] * self.num_corpus_graphs, range(self.num_corpus_graphs))))
            self.all_gt = torch.cat(all_gt)

        self.query_graph_node_sizes = torch.tensor(
            list(map(lambda g: g.number_of_nodes(), self.query_graphs)), device=self.device)
        self.corpus_graph_node_sizes = torch.tensor(
            list(map(lambda g: g.number_of_nodes(), self.corpus_graphs)), device=self.device)

        self.query_graph_edge_sizes = torch.tensor(
            list(map(lambda g: g.number_of_edges(), self.query_graphs)), device=self.device)
        self.corpus_graph_edge_sizes = torch.tensor(
            list(map(lambda g: g.number_of_edges(), self.corpus_graphs)), device=self.device)

        self.query_graph_edges = list(map(lambda g: torch.tensor(np.array(g.edges()), dtype=torch.int64, device=self.device), self.query_graphs))
        self.corpus_graph_edges = list(map(lambda g: torch.tensor(np.array(g.edges()), dtype=torch.int64, device=self.device), self.corpus_graphs))

        num_query_graphs = len(self.query_graphs)
        assert sorted(list(self.relationships.keys())) == list(range(num_query_graphs)), f"{sorted(list(self.relationships.keys()))}, and {list(range(num_query_graphs))}"

        if use_labeled:
            logger.info(f"Loaded {len(self.query_graphs)} query graphs from {q_fname} after filtering for labeled subset")
            logger.info(f"Loaded relationships from {rel_fname} and then filtered for the purpose of creating splits")
        elif use_clusters:
            logger.info(f"Loaded {len(self.query_graphs)} query graphs from {q_fname} after filtering for cluster {conf.use_cluster} (load_mode={load_mode})")
            logger.info(f"Loaded relationships from {rel_fname} and then filtered for cluster {conf.use_cluster}")
        else:
            logger.info(f"Loaded {len(self.query_graphs)} query graphs from {q_fname}")
            logger.info(f"Loaded relationships from {rel_fname} for the purpose of creating splits")
        logger.info(f"Loaded {len(self.corpus_graphs)} corpus graphs from {c_fname}")

        self.pos_pairs, self.neg_pairs = [], []

        for query_idx in range(num_query_graphs):
            for corpus_idx in self.relationships[query_idx]['pos']:
                self.pos_pairs.append((query_idx, corpus_idx))
            for corpus_idx in self.relationships[query_idx]['neg']:
                self.neg_pairs.append((query_idx, corpus_idx))

