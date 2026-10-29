import itertools
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch_geometric as pyg
from torch import Tensor

import GMN.graphembeddingnetwork as gmngen
import GMN.utils as gmnutils

from .utils import model_utils


class GEN(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        super().__init__()
        self.config = gmn_config
        self.build_layers()

        self.diagnostic_mode = False
        self.fetch_embed = False
        self.device = conf.training.device
        assert conf.model.scoring_function in model_utils.scoring_functions.keys()
        self.scoring_function = model_utils.scoring_functions[conf.model.scoring_function]
        self.use_sig = False
        if conf.model.scoring_function == "sighinge":
            self.use_sig = True
            self.sigmoid_a = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
            self.sigmoid_b = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))

    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config["encoder"])
        prop_config = self.config["graph_embedding_net"].copy()
        prop_config.pop("n_prop_layers", None)
        prop_config.pop("share_prop_params", None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        self.aggregator = gmngen.GraphAggregator(**self.config["aggregator"])

    def forward(self, batch_data, batch_data_node_sizes, batch_data_edge_sizes, fetch_split=False,
                pre_last_layer=False):
        """
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(batch_data)

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config["graph_embedding_net"]["n_prop_layers"]):
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)

        # 2*B x D
        graph_vectors = self.aggregator(node_features_enc, graph_idx, len(batch_data_node_sizes),
                                        pre_last_layer=pre_last_layer)

        if self.fetch_embed:
            return graph_vectors

        # Both x and y: B x D
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)

        if fetch_split:
            return x, y
        if self.use_sig:
            scores = self.scoring_function(self.sigmoid_a, self.sigmoid_b, x, y)
        else:
            scores = self.scoring_function(x, y)
        return scores

    @property
    def last_linear_layer_params(self):
        """
        Get the parameters of the last linear layer on top of graph representations.
        Subject to change if the model architecture changes.
        """
        return torch.flatten(self.aggregator.MLP2[-1].weight)


class NANL(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        super().__init__()
        self.config = gmn_config
        self.diagnostic_mode = False
        self.fetch_embed = False
        self.device = conf.training.device
        self.max_node_set_size = conf.actual_max_node_set_size
        self.build_layers()

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=self.max_node_set_size, lateral_dim=self.max_node_set_size, device=self.device
        )
        self.sinkhorn_temp = conf.training.sinkhorn_temp
        self.scoring_layer = conf.dataset.rel_mode

    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config["encoder"])
        prop_config = self.config["graph_embedding_net"].copy()
        prop_config.pop("n_prop_layers", None)
        prop_config.pop("share_prop_params", None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        self.aggregator = gmngen.GraphAggregator(**self.config["aggregator"])
        self.node_sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(prop_config["node_state_dim"], self.max_node_set_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.max_node_set_size, self.max_node_set_size)
        )

    def forward(self, graphs, graph_node_sizes, graph_edge_sizes):
        query_sizes = graph_node_sizes[0::2]
        corpus_sizes = graph_node_sizes[1::2]

        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        # Propagation to compute node embeddings
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for _ in range(self.config["graph_embedding_net"]["n_prop_layers"]):
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)

        stacked_node_features_query, stacked_node_features_corpus = model_utils.split_and_stack_pairs(
            node_features_enc, graph_node_sizes.tolist(), self.max_node_set_size
        )

        # Computation of node transport plan
        transformed_features_query = self.node_sinkhorn_feature_layers(stacked_node_features_query)
        transformed_features_corpus = self.node_sinkhorn_feature_layers(stacked_node_features_corpus)

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features
        masked_features_query = mask_graphs(transformed_features_query, query_sizes)
        masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

        node_sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
        node_transport_plan = model_utils.pytorch_sinkhorn_iters(
            log_alpha=node_sinkhorn_input, device=self.device, temperature=self.sinkhorn_temp
        )

        if self.diagnostic_mode:
            return node_features_enc, node_transport_plan, stacked_node_features_query, stacked_node_features_corpus, node_transport_plan
        if self.fetch_embed:
            return model_utils.split_and_stack_singles(node_features_enc, graph_node_sizes.tolist(), self.max_node_set_size)
        if self.scoring_layer == "sub_iso":
            return model_utils.subiso_feature_alignment_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan)
        if self.scoring_layer == "ged":
            return model_utils.ged_feature_alignment_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan)
        if self.scoring_layer == "uneq_ged":
            return model_utils.uneq_ged_feature_alignment_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan)
        if self.scoring_layer == "mcs":
            # return model_utils.mcs_feature_alignment_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan)
            raise NotImplementedError("MCS not implemented")
        raise NotImplementedError(f"Scoring layer {self.scoring_layer} not implemented")


class NANLSIG(NANL):
    def __init__(self, conf, gmn_config):
        super().__init__(conf, gmn_config)
        self.sigmoid_a = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.sigmoid_b = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))

    def forward(self, graphs, graph_node_sizes, graph_edge_sizes):
        query_sizes = graph_node_sizes[0::2]
        corpus_sizes = graph_node_sizes[1::2]

        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        # Propagation to compute node embeddings
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for _ in range(self.config["graph_embedding_net"]["n_prop_layers"]):
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)

        stacked_node_features_query, stacked_node_features_corpus = model_utils.split_and_stack_pairs(
            node_features_enc, graph_node_sizes.tolist(), self.max_node_set_size
        )

        # Computation of node transport plan
        transformed_features_query = self.node_sinkhorn_feature_layers(stacked_node_features_query)
        transformed_features_corpus = self.node_sinkhorn_feature_layers(stacked_node_features_corpus)

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features
        masked_features_query = mask_graphs(transformed_features_query, query_sizes)
        masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

        node_sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
        node_transport_plan = model_utils.pytorch_sinkhorn_iters(
            log_alpha=node_sinkhorn_input, device=self.device, temperature=self.sinkhorn_temp
        )

        if self.diagnostic_mode:
            return node_features_enc, node_transport_plan, stacked_node_features_query, stacked_node_features_corpus, node_transport_plan
        if self.fetch_embed:
            return model_utils.split_and_stack_singles(node_features_enc, graph_node_sizes.tolist(), self.max_node_set_size)
        if self.scoring_layer == "sub_iso":
            return model_utils.subiso_feature_alignment_sig_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan, self.sigmoid_a, self.sigmoid_b)
        if self.scoring_layer == "ged":
            return model_utils.ged_feature_alignment_sig_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan, self.sigmoid_a, self.sigmoid_b)
        if self.scoring_layer == "uneq_ged":
            raise NotImplementedError("Unequal GED not implemented for sigmoid")
        if self.scoring_layer == "mcs":
            # return model_utils.mcs_feature_alignment_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan)
            raise NotImplementedError("MCS not implemented")
        raise NotImplementedError(f"Scoring layer {self.scoring_layer} not implemented")


class ISONET(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        super().__init__()
        self.config = gmn_config
        self.diagnostic_mode = False
        self.fetch_embed = False
        self.device = conf.training.device
        self.max_edge_set_size = conf.actual_max_edge_set_size
        self.build_layers()

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=self.max_edge_set_size, lateral_dim=self.max_edge_set_size, device=self.device
        )
        self.sinkhorn_temp = conf.training.sinkhorn_temp

    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config["encoder"])
        prop_config = self.config["graph_embedding_net"].copy()
        prop_config.pop("n_prop_layers", None)
        prop_config.pop("share_prop_params", None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        self.aggregator = gmngen.GraphAggregator(**self.config["aggregator"])
        self.edge_sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(
                prop_config["edge_hidden_sizes"][-1], self.max_edge_set_size
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(self.max_edge_set_size, self.max_edge_set_size),
        )

    def forward(self, graphs, graph_node_sizes, graph_edge_sizes):
        query_sizes = graph_edge_sizes[0::2]
        corpus_sizes = graph_edge_sizes[1::2]

        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(graphs)

        # Propagation to compute node embeddings
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for _ in range(self.config["graph_embedding_net"]["n_prop_layers"]):
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)

        edge_features_enc = model_utils.propagation_messages(
            propagation_layer=self.prop_layer,
            node_features=node_features_enc,
            edge_features=edge_features_enc,
            from_idx=from_idx,
            to_idx=to_idx
        )
        # paired_edge_counts = model_utils.get_paired_edge_counts(
        #     from_idx=from_idx, to_idx=to_idx,
        #     graph_idx=graph_idx, num_graphs=len(graph_node_sizes)
        # )
        # assert  (torch.tensor([gidx for gidx_tuple in paired_edge_counts for gidx in gidx_tuple], device=graph_edge_sizes.device)  == graph_edge_sizes).all(), print(f"paired_edge_counts: {paired_edge_counts}, graph_edge_sizes: {graph_edge_sizes}")

        (
            stacked_edge_features_query,
            stacked_edge_features_corpus,
        ) = model_utils.split_and_stack_pairs(
            edge_features_enc, graph_edge_sizes.tolist(), self.max_edge_set_size
        )
        # Computation of edge transport plan
        transformed_features_query = self.edge_sinkhorn_feature_layers(
            stacked_edge_features_query
        )
        transformed_features_corpus = self.edge_sinkhorn_feature_layers(
            stacked_edge_features_corpus
        )

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features
        masked_features_query = mask_graphs(transformed_features_query, query_sizes)
        masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

        edge_sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
        edge_transport_plan = model_utils.pytorch_sinkhorn_iters(
            log_alpha=edge_sinkhorn_input, device=self.device, temperature=self.sinkhorn_temp
        )

        if self.diagnostic_mode:
            return node_features_enc, edge_transport_plan, stacked_edge_features_query, stacked_edge_features_corpus, edge_transport_plan
        # return model_utils.feature_alignment_score(stacked_edge_features_query, stacked_edge_features_corpus, edge_transport_plan)
        raise NotImplementedError("TODO: Function does not exist")


class NANL2(torch.nn.Module):
    def __init__(self, conf, gmn_config):
        super().__init__()
        self.config = gmn_config
        self.diagnostic_mode = False
        self.fetch_embed = False
        self.device = conf.training.device
        self.max_node_set_size = conf.actual_max_node_set_size
        self.build_layers()

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=self.max_node_set_size, lateral_dim=self.max_node_set_size, device=self.device
        )
        self.sinkhorn_temp = conf.training.sinkhorn_temp

    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config["encoder"])
        prop_config = self.config["graph_embedding_net"].copy()
        prop_config.pop("n_prop_layers", None)
        prop_config.pop("share_prop_params", None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        self.aggregator = gmngen.GraphAggregator(**self.config["aggregator"])
        self.node_sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(prop_config["node_state_dim"], self.max_node_set_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.max_node_set_size, self.max_node_set_size)
        )

    def forward(self, graphs, graph_node_sizes, graph_edge_sizes):
        query_sizes = graph_node_sizes[0::2]
        corpus_sizes = graph_node_sizes[1::2]
        batch_graphs = pyg.data.Batch.from_data_list(graphs)
        edge_features = torch.ones((batch_graphs.edge_index.shape[1], 1), device=batch_graphs.edge_index.device)
        # node_features, edge_features, from_idx, to_idx, graph_idx= model_utils.get_graph_features(graphs)

        # Propagation to compute node embeddings
        node_features_enc, edge_features_enc = self.encoder(batch_graphs.x, edge_features)
        for _ in range(self.config["graph_embedding_net"]["n_prop_layers"]):
            node_features_enc = self.prop_layer(node_features_enc, batch_graphs.edge_index[0, :], batch_graphs.edge_index[1, :], edge_features_enc)
        stacked_node_features_query, stacked_node_features_corpus = model_utils.split_and_stack_pairs_contiguous(
            node_features_enc, query_sizes.tolist() + corpus_sizes.tolist(), self.max_node_set_size, len(query_sizes)
                )

        # Computation of node transport plan
        transformed_features_query = self.node_sinkhorn_feature_layers(stacked_node_features_query)
        transformed_features_corpus = self.node_sinkhorn_feature_layers(stacked_node_features_corpus)

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features
        masked_features_query = mask_graphs(transformed_features_query, query_sizes)
        masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

        node_sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
        node_transport_plan = model_utils.pytorch_sinkhorn_iters(
            log_alpha=node_sinkhorn_input, device=self.device, temperature=self.sinkhorn_temp
        )

        if self.diagnostic_mode:
            return node_features_enc, node_transport_plan, stacked_node_features_query, stacked_node_features_corpus, node_transport_plan
        return model_utils.feature_alignment_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        *,
        hidden_act: type[nn.Module] = nn.ReLU,
        final_act: type[nn.Module] | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dims = (in_dim, *tuple(hidden_dims))
        for d_in, d_out in itertools.pairwise(dims):
            layers.extend([nn.Linear(d_in, d_out), hidden_act()])
        layers.append(nn.Linear(dims[-1], out_dim))
        if final_act is not None:
            layers.append(final_act())
        self.net = nn.Sequential(*layers)
        self.reset_params()

    def reset_params(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.kaiming_uniform_(last.weight, nonlinearity="relu")
            nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)


class HashModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, gen_out_dim: int,
                 linear_every_dim: bool = False):
        super().__init__()
        self.linear_every_dim = linear_every_dim

        if linear_every_dim:
            self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(gen_out_dim)])
        else:
            self.linears = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.linear_every_dim:
            x = x.permute(1, 0, 2)  # D x B x d
            x = torch.stack([self.linears[d](x[d]) for d in range(x.shape[0])], dim=1)  # D x B x output_dim -> B x D x output_dim
        else:
            x = self.linears(x)
        return x


class AugEmbeddingMap(torch.nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, *, symmetric: bool,
        aug_emb_model: str = "mlp", model_every_dim: bool = False, gen_out_dim: int = None
    ):
        super().__init__()
        self.model_every_dim = model_every_dim

        if aug_emb_model == "mlp":
            ModelClass = MLP
            model_args = (input_dim, hidden_dims, output_dim)
        elif aug_emb_model == "linear":
            # Only one layer linear should be sufficient
            ModelClass = nn.Linear
            model_args = (input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported aug_emb_model: {aug_emb_model}")

        if model_every_dim:
            self.model_query = nn.ModuleList([ModelClass(*model_args) for _ in range(gen_out_dim)])
        else:
            self.model_query = ModelClass(*model_args)

        if symmetric:
            self.model_corpus = self.model_query
        else:
            if model_every_dim:
                self.model_corpus = nn.ModuleList([ModelClass(*model_args) for _ in range(gen_out_dim)])
            else:
                self.model_corpus = ModelClass(*model_args)

    def forward(self, q_hat: Tensor, x_hat: Tensor) -> tuple[Tensor, Tensor]:
        if self.model_every_dim:
            # q_hat and x_hat: Q/C x D x (R*2*(1+d1)) or Q/C x D x (R*2*(2+d1)) depending on aug_form
            # Apply the corresponding model to each D dimension separately
            Q, D, _ = q_hat.shape
            C, D, _ = x_hat.shape

            assert len(self.model_query) == D,\
                f"Dimension D of input ({D}) must match number of models ({len(self.model_query)})"
            assert len(self.model_corpus) == D,\
                f"Dimension D of input ({D}) must match number of models ({len(self.model_corpus)})"

            q_hat = q_hat.permute(1, 0, 2)  # D x Q x (R*2*(1+d1))
            x_hat = x_hat.permute(1, 0, 2)  # D x C x (R*2*(1+d1))

            q_hat = torch.stack([self.model_query[d](q_hat[d]) for d in range(D)], dim=1)  # D x Q x output_dim -> Q x D x output_dim
            x_hat = torch.stack([self.model_corpus[d](x_hat[d]) for d in range(D)], dim=1)  # D x C x output_dim -> C x D x output_dim

        else:
            q_hat = self.model_query(q_hat)
            x_hat = self.model_corpus(x_hat)
        return q_hat, x_hat
