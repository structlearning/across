import torch
from GMN.segment import unsorted_segment_sum
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import numpy as np


@torch.inference_mode()
def compute_gnn_embeds(model, dataset, pre_last_layer=True):
    model.eval()
    model.fetch_embed = True
    with torch.no_grad():
        q_embeds = model(
            dataset._pack_batch_1d(dataset.query_graphs),
            dataset.query_graph_node_sizes,
            dataset.query_graph_edge_sizes,
            pre_last_layer=pre_last_layer,
        )
        c_embeds = model(
            dataset._pack_batch_1d(dataset.corpus_graphs),
            dataset.corpus_graph_node_sizes,
            dataset.corpus_graph_edge_sizes,
            pre_last_layer=pre_last_layer,
        )
    model.fetch_embed = False
    return q_embeds, c_embeds

def dot_product_score(x,y):
    return (x*y).sum(-1)

def cosine_score(x,y):
    return torch.nn.functional.cosine_similarity(x,y)

def euclidean_distance_score(x, y):
    """This is the squared Euclidean distance based score.
    Returning negative of the distance as score"""
    return -torch.sum((x - y) ** 2, dim=-1)

def hinge_distance_score(x, y):
    """ Hinge distance based score.
    """
    return -torch.sum(torch.nn.ReLU()(x-y),dim=-1)

def min_score(x, y):
    """ Min score between two vectors.
    """
    return torch.min(x,y).sum(-1)

def l1_score(x,y):
    return -torch.sum(torch.abs(x-y),dim=-1)


def sigmoid_hinge_sim(sigmoid_a, sigmoid_b, x, y):
    sig_in= torch.sum(torch.nn.ReLU()(x-y),dim=-1)

    return torch.nn.Sigmoid()(sigmoid_a*sig_in + sigmoid_b)

#create a function dictionary for scoring layers. map names to functions.
scoring_functions = {
    "dot": dot_product_score,
    "cos": cosine_score,
    "euc": euclidean_distance_score,   
    "hinge": hinge_distance_score,
    "sighinge": sigmoid_hinge_sim,
    "min": min_score,
    "l1": l1_score,
}


def get_default_gmn_config(conf):
    """The default configs."""
    model_type = "matching"
    # Set to `embedding` to use the graph embedding net.
    node_state_dim = 32
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here. gru
        node_update_type="gru",
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=False,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
        # set to `embedding` to use the graph embedding net.
        prop_type="embedding",
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"  # other: euclidean, cosine
    graph_matching_net_config["prop_type"] = "matching"  # other: euclidean, cosine
    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            node_feature_dim=1,
            edge_hidden_sizes=None,
        ),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type="sum",
        ),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        model_type=model_type,
        data=dict(
            problem="graph_edit_distance",
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
                n_nodes_range=[20, 20],
                p_edge_range=[0.2, 0.2],
                n_changes_positive=1,
                n_changes_negative=2,
                validation_dataset_size=1000,
            ),
        ),
        training=dict(
            batch_size=20,
            learning_rate=1e-4,
            mode="pair",
            loss="margin",  # other: hamming
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            n_training_steps=500000,
            # Print training information every this many training steps.
            print_after=100,
            # Evaluate on validation set every `eval_after * print_after` steps.
            eval_after=10,
        ),
        evaluation=dict(batch_size=20),
        seed=conf.training.seed,
    )


def modify_gmn_main_config(gmn_config, conf, logger):
    gmn_config["encoder"]["node_hidden_sizes"] = [conf.gmn.filters_3]  # [10]
    gmn_config["encoder"]["node_feature_dim"] = 1
    gmn_config["encoder"]["edge_feature_dim"] = 1
    gmn_config["aggregator"]["node_hidden_sizes"] = [conf.gmn.filters_3]  # [10]
    gmn_config["aggregator"]["graph_transform_sizes"] = [conf.gmn.filters_3]  # [10]
    gmn_config["aggregator"]["input_size"] = [conf.gmn.filters_3]  # [10]
    gmn_config["graph_matching_net"]["node_state_dim"] = conf.gmn.filters_3  # 10
    # gmn_config['graph_matching_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
    gmn_config["graph_matching_net"]["edge_hidden_sizes"] = [
        2 * conf.gmn.filters_3
    ]  # [20]
    gmn_config["graph_matching_net"]["node_hidden_sizes"] = [conf.gmn.filters_3]  # [10]
    gmn_config["graph_matching_net"]["n_prop_layers"] = 5
    gmn_config["graph_embedding_net"]["node_state_dim"] = conf.gmn.filters_3  # 10
    # gmn_config['graph_embedding_net'] ['n_prop_layers'] = av.GMN_NPROPLAYERS
    gmn_config["graph_embedding_net"]["edge_hidden_sizes"] = [
        2 * conf.gmn.filters_3
    ]  # [20]
    gmn_config["graph_embedding_net"]["node_hidden_sizes"] = [
        conf.gmn.filters_3
    ]  # [10]
    gmn_config["graph_embedding_net"]["n_prop_layers"] = 5

    # logger.info("av gmn_prop_param")
    # logger.info(av.GMN_NPROPLAYERS)
    # logger.info("gmn_config param")
    # logger.info(gmn_config['graph_embedding_net'] ['n_prop_layers'] )
    gmn_config["graph_embedding_net"]["n_prop_layers"] = conf.gmn.GMN_NPROPLAYERS
    gmn_config["graph_matching_net"]["n_prop_layers"] = conf.gmn.GMN_NPROPLAYERS
    # logger.info("gmn_config param")
    # logger.info(gmn_config['graph_embedding_net'] ['n_prop_layers'] )

    gmn_config["training"]["batch_size"] = conf.training.batch_size
    # gmn_config['training']['margin']  = av.MARGIN
    gmn_config["evaluation"]["batch_size"] = conf.training.batch_size
    gmn_config["model_type"] = "embedding"
    gmn_config["graphsim"] = {}
    gmn_config["graphsim"]["conv_kernel_size"] = [10, 4, 2]
    gmn_config["graphsim"]["linear_size"] = [24, 16]
    gmn_config["graphsim"]["gcn_size"] = [10, 10, 10]
    gmn_config["graphsim"]["conv_pool_size"] = [3, 3, 2]
    gmn_config["graphsim"]["conv_out_channels"] = [2, 4, 8]
    gmn_config["graphsim"]["dropout"] = conf.training.dropout
    logger.info("Modified GMN config:")
    for k, v in gmn_config.items():
        logger.info("%s= %s" % (k, v))
    return gmn_config



def get_graph_features(graphs):
    return graphs.node_features, graphs.edge_features, graphs.from_idx, graphs.to_idx, graphs.graph_idx    

def pytorch_sample_gumbel(shape, device, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape, device=device, dtype=torch.float)
    return -torch.log(eps - torch.log(U + eps))

def pytorch_sinkhorn_iters(log_alpha, device, temperature=0.1, noise_factor=0, num_iters=20):
    batch_size, num_objs, _ = log_alpha.shape
    noise = pytorch_sample_gumbel([batch_size, num_objs, num_objs], device) * noise_factor
    log_alpha = log_alpha + noise
    log_alpha = torch.div(log_alpha, temperature)
    for _ in range(num_iters):
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, num_objs, 1)
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, num_objs)
    return torch.exp(log_alpha)

def pytorch_sinkhorn_iters_mask(log_alpha, mask, device, temperature=0.1, noise_factor=0, num_iters=20):
    batch_size, num_objs, _ = log_alpha.shape
    noise = pytorch_sample_gumbel([batch_size, num_objs, num_objs], device) * noise_factor
    log_alpha = log_alpha + noise
    log_alpha = torch.div(log_alpha, temperature)
    for _ in range(num_iters):
       log_alpha = (log_alpha - (torch.logsumexp(log_alpha.masked_fill_(mask, float("-inf")), dim=2, keepdim=True)).view(-1, num_objs, 1)).masked_fill_(mask, float("-inf"))
       log_alpha = (log_alpha - (torch.logsumexp(log_alpha.masked_fill_(mask, float("-inf")), dim=1, keepdim=True)).view(-1, 1, num_objs)).masked_fill_(mask, float("-inf"))
 
    return torch.exp(log_alpha)

def graph_size_to_mask_map(max_set_size, lateral_dim, device=None):
    return [torch.cat((
        torch.tensor([1], device=device, dtype=torch.float).repeat(x, 1).repeat(1, lateral_dim),
        torch.tensor([0], device=device, dtype=torch.float).repeat(max_set_size - x, 1).repeat(1, lateral_dim)
    )) for x in range(0, max_set_size + 1)]
    
def set_size_to_mask_map(max_set_size, device=None):
    # Mask pattern sets top left (k)*(k) square to 1 inside arrays of size n*n. Rest elements are 0
    return [torch.cat(
            (
                torch.repeat_interleave(torch.tensor([1, 0], device=device, dtype=torch.float), torch.tensor([x, max_set_size - x], device=device)).repeat(x, 1),
                torch.repeat_interleave(torch.tensor([1, 0], device=device, dtype=torch.float), torch.tensor([0,max_set_size], device=device)).repeat(max_set_size - x, 1),
            )
        ) for x in range(0, max_set_size + 1)]


def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def get_padded_indices(paired_sizes, max_set_size, device):
    num_pairs = len(paired_sizes)
    max_set_size_arange = torch.arange(max_set_size, dtype=torch.long, device=device).reshape(1, -1).repeat(num_pairs * 2, 1)
    flattened_sizes = torch.tensor(flatten_list_of_lists(paired_sizes), device=device)
    presence_mask = max_set_size_arange < flattened_sizes.unsqueeze(1)

    cumulative_set_sizes = torch.cumsum(torch.tensor(
        max_set_size, dtype=torch.long, device=device
    ).repeat(len(flattened_sizes)), dim=0)
    max_set_size_arange[1:, :] += cumulative_set_sizes[:-1].unsqueeze(1)
    return max_set_size_arange[presence_mask]

def split_to_query_and_corpus(features, graph_sizes):
    # [(8, 12), (10, 13), (10, 14)] -> [8, 12, 10, 13, 10, 14]
    # flattened_graph_sizes  = flatten_list_of_lists(graph_sizes)
    # features_split = torch.split(features, flattened_graph_sizes, dim=0)
    features_split = torch.split(features, graph_sizes, dim=0)
    features_query = features_split[0::2]
    features_corpus = features_split[1::2]
    return features_query, features_corpus

def split_and_stack_pairs(features, graph_sizes, max_set_size):
    features_query, features_corpus = split_to_query_and_corpus(features, graph_sizes)
    
    stack_features = lambda features_array: torch.stack([
        F.pad(features, pad=(0, 0, 0, max_set_size - features.shape[0])) for features in features_array
    ])
    return stack_features(features_query), stack_features(features_corpus)


def split_to_query_and_corpus_contiguous(features, graph_sizes, b_sz):
    features_split = torch.split(features, graph_sizes, dim=0)
    
    features_query = features_split[:b_sz]
    features_corpus = features_split[b_sz:]
    return features_query, features_corpus

def split_and_stack_pairs_contiguous(features, graph_sizes, max_set_size, b_sz):
    features_query, features_corpus = split_to_query_and_corpus_contiguous(features, graph_sizes, b_sz)
    
    stack_features = lambda features_array: torch.stack([
        F.pad(features, pad=(0, 0, 0, max_set_size - features.shape[0])) for features in features_array
    ])
    return stack_features(features_query), stack_features(features_corpus)

def split_and_stack_singles(features, graph_sizes, max_set_size):
    features_split = torch.split(features, graph_sizes, dim=0)
    
    stack_features = lambda features_array: torch.stack([
        F.pad(features, pad=(0, 0, 0, max_set_size - features.shape[0])) for features in features_array
    ])
    return stack_features(features_split)

def subiso_feature_alignment_sig_score(query_features, corpus_features, transport_plan, sigmoid_a, sigmoid_b):
    sig_in =  torch.maximum(
        query_features - transport_plan @ corpus_features,
        torch.tensor([0], device=query_features.device)
    ).sum(dim=(-1, -2))
    return torch.nn.Sigmoid()(sigmoid_a*sig_in + sigmoid_b)


def ged_feature_alignment_sig_score(query_features, corpus_features, transport_plan, sigmoid_a, sigmoid_b):
    sig_in = torch.abs(query_features - transport_plan @ corpus_features).sum(dim=(-1, -2))
    return torch.nn.Sigmoid()(sigmoid_a*sig_in + sigmoid_b)


def subiso_feature_alignment_score(query_features, corpus_features, transport_plan):
    return - torch.maximum(
        query_features - transport_plan @ corpus_features,
        torch.tensor([0], device=query_features.device)
    ).sum(dim=(-1, -2))

def subiso_discrete_alignment_distance(query_features, corpus_features, transport_plan):
    # \sum_{(u, v): i} H_q[u, i] * (1 - H_c[v, i]) * P[u, v]
    return (query_features * (
        transport_plan @ (1 - corpus_features))).sum((-1, -2))

def subiso_discrete_alignment_score(query_features, corpus_features, transport_plan):
    return - subiso_discrete_alignment_distance(
        query_features, corpus_features,
        transport_plan)

def ged_feature_alignment_score(query_features, corpus_features, transport_plan):
    return - torch.abs(query_features - transport_plan @ corpus_features).sum(dim=(-1, -2))

def ged_feature_alignment_distance(query_features, corpus_features, transport_plan):
    return torch.abs(query_features - transport_plan @ corpus_features).sum(dim=(-1, -2))

def uneq_ged_feature_alignment_distance(query_features, corpus_features, transport_plan):
    add_cost=1 #NOTE: Hardcoded in this repo. Can be general cost
    del_cost=2 #NOTE: Hardcoded in this repo. Can be general cost
    
    del_component = del_cost * torch.sum(torch.nn.ReLU()(query_features - transport_plan @ corpus_features), dim=(-1, -2))
    ins_component = add_cost * torch.sum(torch.nn.ReLU()(transport_plan @ corpus_features - query_features), dim=(-1, -2))
    
    return del_component + ins_component

def uneq_ged_feature_alignment_score(query_features, corpus_features, transport_plan):
    add_cost=1 #NOTE: Hardcoded in this repo. Can be general cost
    del_cost=2 #NOTE: Hardcoded in this repo. Can be general cost
    
    del_component = del_cost * torch.sum(torch.nn.ReLU()(query_features - transport_plan @ corpus_features), dim=(-1, -2))
    ins_component = add_cost * torch.sum(torch.nn.ReLU()(transport_plan @ corpus_features - query_features), dim=(-1, -2))
    
    return -(del_component + ins_component)

def get_paired_edge_counts(from_idx, to_idx, graph_idx, num_graphs):
    edges_per_src_node = unsorted_segment_sum(torch.ones_like(from_idx, dtype=torch.float), from_idx, len(graph_idx))
    edges_per_graph_from = unsorted_segment_sum(edges_per_src_node, graph_idx, num_graphs)

    edges_per_dest_node = unsorted_segment_sum(torch.ones_like(to_idx, dtype=torch.float), to_idx, len(graph_idx))
    edges_per_graph_to = unsorted_segment_sum(edges_per_dest_node, graph_idx, num_graphs)

    assert (edges_per_graph_from == edges_per_graph_to).all()

    return edges_per_graph_to.reshape(-1, 2).int().tolist()

def propagation_messages(propagation_layer, node_features, edge_features, from_idx, to_idx):
    edge_src_features = node_features[from_idx]
    edge_dest_features = node_features[to_idx]

    forward_edge_msg = propagation_layer._message_net(torch.cat([
        edge_src_features, edge_dest_features, edge_features
    ], dim=-1))
    backward_edge_msg = propagation_layer._reverse_message_net(torch.cat([
        edge_dest_features, edge_src_features, edge_features
    ], dim=-1))
    return forward_edge_msg + backward_edge_msg



def gen_map_from_embeds(conf,q_embeds, c_embeds, all_gt, sigmoid_a=None, sigmoid_b=None):
    all_ap = []
    all_preds = []
    num_cgraphs = c_embeds.shape[0]
    for query_idx in range(q_embeds.shape[0]):
        if conf.model.scoring_function == "sighinge":
            preds = scoring_functions[conf.model.scoring_function](sigmoid_a, sigmoid_b, q_embeds[query_idx][None,:],c_embeds)
        else:
            preds = scoring_functions[conf.model.scoring_function](q_embeds[query_idx][None,:],c_embeds)
        all_labels = all_gt[query_idx*num_cgraphs:(query_idx+1)*num_cgraphs]
        average_precision = average_precision_score(all_labels, preds.detach().cpu())
        all_ap.append(average_precision)
        all_preds.extend(preds.detach().cpu().tolist())
    return np.mean(all_ap), all_preds

def gen_map_from_embeds_using_scoring_fn(scoring_fn,q_embeds, c_embeds, all_gt, sigmoid_a=None, sigmoid_b=None):
    all_ap = []
    all_preds = []
    num_cgraphs = c_embeds.shape[0]
    for query_idx in range(q_embeds.shape[0]):
        if scoring_fn == "sighinge":
            preds = scoring_functions[scoring_fn](sigmoid_a, sigmoid_b, q_embeds[query_idx][None,:],c_embeds)
        else:
            preds = scoring_functions[scoring_fn](q_embeds[query_idx][None,:],c_embeds)
        all_labels = all_gt[query_idx*num_cgraphs:(query_idx+1)*num_cgraphs]
        average_precision = average_precision_score(all_labels, preds.detach().cpu())
        all_ap.append(average_precision)
        all_preds.extend(preds.detach().cpu().tolist())
    return np.mean(all_ap), all_preds
    
    
def gen_map_from_scores(all_scores, all_gt):
    all_ap = []
    num_cgraphs = all_scores.shape[1]
    for query_idx in range(all_scores.shape[0]):
        all_preds = all_scores[query_idx]
        all_labels = all_gt[query_idx*num_cgraphs:(query_idx+1)*num_cgraphs]
        average_precision = average_precision_score(all_labels, all_preds.detach().cpu())
        all_ap.append(average_precision)
    return np.mean(all_ap)

def pytorch_sinkhorn_iters_inplace(log_alpha, temperature=0.1, num_iters=20):
    batch_size, num_objs, _ = log_alpha.shape
    log_alpha = torch.div(log_alpha, temperature)
    for _ in range(num_iters):
        log_alpha -= (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, num_objs, 1)
        log_alpha -=  (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, num_objs)
    return torch.exp(log_alpha)

def mask_graphs(model,graph_sizes):
    mask = torch.stack([model.graph_size_to_mask_map[i] for i in graph_sizes])
    return mask 



def nanl_fast_inference(conf, model, sampler, use_sig=False):
    ## Note that this is optmized for speed, under the assumption that the 
    ## pre sinkhorn transformation is an LRL with weights w1,b1,w2,b2
    
    #Run an assert to check assumption 
    assert len(model.node_sinkhorn_feature_layers)==3
    model.fetch_embed = True
    q_embeds = model(sampler._pack_batch_1d(sampler.query_graphs),sampler.query_graph_node_sizes,sampler.query_graph_edge_sizes)
    c_embeds = model(sampler._pack_batch_1d(sampler.corpus_graphs),sampler.corpus_graph_node_sizes,sampler.corpus_graph_edge_sizes)
    model.fetch_embed = False
    
    w1 = model.node_sinkhorn_feature_layers[0].weight
    b1 = model.node_sinkhorn_feature_layers[0].bias
    w2 = model.node_sinkhorn_feature_layers[2].weight    
    b2 = model.node_sinkhorn_feature_layers[2].bias
    transformed_features_query = (torch.nn.ReLU()(q_embeds@w1.T+b1) )@w2.T+b2
    transformed_features_corpus = (torch.nn.ReLU()(c_embeds@w1.T+b1))@w2.T+b2
    query_mask = mask_graphs(model,sampler.query_graph_node_sizes)
    corpus_mask = mask_graphs(model,sampler.corpus_graph_node_sizes)
    masked_features_query = query_mask * transformed_features_query
    masked_features_corpus = corpus_mask * transformed_features_corpus

    chunk_size = 30
    all_scores = []
    for start_idx in range(0,q_embeds.shape[0],chunk_size):
        end_idx = min(start_idx+chunk_size, q_embeds.shape[0])
        node_sinkhorn_input =torch.matmul(masked_features_query[start_idx:end_idx,None,:,:],\
                masked_features_corpus.permute(0,2,1)[None,:,:,:])
        b1,b2,n1,n2 = node_sinkhorn_input.shape
        node_transport_plan = pytorch_sinkhorn_iters_inplace(
            node_sinkhorn_input.reshape(b1*b2, n1,n2), temperature=conf.training.sinkhorn_temp, num_iters=20
        ).reshape(b1,b2,n1,n2)
        q_emb = q_embeds[start_idx:end_idx,None,:,:]

        if conf.dataset.rel_mode ==  "sub_iso":
            if use_sig:
                scores = subiso_feature_alignment_sig_score(q_emb, c_embeds, node_transport_plan, model.sigmoid_a, model.sigmoid_b)
            else:
                scores = subiso_feature_alignment_score(q_emb, c_embeds, node_transport_plan)
        elif conf.dataset.rel_mode == "ged":
            if use_sig:
                scores = ged_feature_alignment_sig_score(q_emb, c_embeds, node_transport_plan, model.sigmoid_a, model.sigmoid_b)
            else:
                scores = ged_feature_alignment_score(q_emb, c_embeds, node_transport_plan)
        elif conf.dataset.rel_mode == "uneq_ged":
            if use_sig:
                raise NotImplementedError()
            else: 
                scores = uneq_ged_feature_alignment_score(q_emb, c_embeds, node_transport_plan)
        # if conf.dataset.rel_mode == "sub_iso":
        #     scores = -torch.nn.ReLU()(q_emb -  (node_transport_plan@c_embeds)).sum((-1,-2))
        # elif conf.dataset.rel_mode == "ged":
        #     scores = torch.abs(q_emb -  (node_transport_plan@c_embeds)).sum((-1,-2))
        else: 
            raise NotImplementedError()
        all_scores.append(scores)
    return torch.vstack(all_scores)


def nanl_fast_inference_without_model(q_embeds, c_embeds, masked_features_query, masked_features_corpus, sinkhorn_temp, rel_mode, use_sig=False, sigmoid_a=None, sigmoid_b=None):
    chunk_size = 30
    all_scores = []
    for start_idx in range(0,q_embeds.shape[0],chunk_size):
        end_idx = min(start_idx+chunk_size, q_embeds.shape[0])
        node_sinkhorn_input =torch.matmul(masked_features_query[start_idx:end_idx,None,:,:],\
                masked_features_corpus.permute(0,2,1)[None,:,:,:])
        b1,b2,n1,n2 = node_sinkhorn_input.shape
        node_transport_plan = pytorch_sinkhorn_iters_inplace(
            node_sinkhorn_input.reshape(b1*b2, n1,n2), temperature=sinkhorn_temp, num_iters=20
        ).reshape(b1,b2,n1,n2)
        if rel_mode ==  "sub_iso":
            if use_sig:
                scores = subiso_feature_alignment_sig_score(q_embeds[start_idx:end_idx,None,:,:], c_embeds, node_transport_plan, sigmoid_a, sigmoid_b)
            else:
                # print(q_embeds[start_idx:end_idx,None,:,:].shape, c_embeds.shape, node_transport_plan.shape)
                scores = subiso_feature_alignment_score(q_embeds[start_idx:end_idx,None,:,:], c_embeds, node_transport_plan)
        elif rel_mode == "ged":
            if use_sig:
                scores = ged_feature_alignment_sig_score(q_embeds[start_idx:end_idx,None,:,:], c_embeds, node_transport_plan, sigmoid_a, sigmoid_b)
            else:
                scores = ged_feature_alignment_score(q_embeds[start_idx:end_idx,None,:,:], c_embeds, node_transport_plan)
        elif rel_mode == "uneq_ged":
            if use_sig:
                raise NotImplementedError()
            else:
                scores = uneq_ged_feature_alignment_score(q_embeds[start_idx:end_idx,None,:,:], c_embeds, node_transport_plan)
                
        all_scores.append(scores)
    
    return torch.vstack(all_scores)
    

    
    
