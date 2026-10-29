import time

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import average_precision_score

from .model_utils import gen_map_from_embeds, gen_map_from_scores, nanl_fast_inference


from sklearn.metrics import ndcg_score


def precision_at_k(y_true, y_scores, k=None):
    """
    Computes Precision@K for a single query/list.
    """
    if k is None:
        # Compute set precision
        # NOTE: The condition below only works if our scores are cosine-similarity.
        # They will not work on hinge scores for example.
        k = y_scores[y_scores > -2].shape[0]
    # Rank items by score and get the indices of top K
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    # Get the true labels for the top K items
    top_k_true = np.asarray(y_true)[top_k_indices]
    # Count relevant items in top K (True Positives)
    relevant_in_top_k = np.sum(top_k_true)
    # Precision@K is relevant items / K
    return relevant_in_top_k / k if k > 0 else 0


def recall_at_k(y_true, y_scores, k=None):
    """
    Computes Recall@K for a single query/list.
    """
    if k is None:
        # Set recall
        k = y_scores[y_scores > -2].shape[0]
    # Rank items by score and get the indices of top K
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    # Get the true labels for the top K items
    top_k_true = np.asarray(y_true)[top_k_indices]
    # Count relevant items in top K (True Positives)
    relevant_in_top_k = np.sum(top_k_true)
    # Total relevant items in the whole dataset
    total_relevant_items = np.sum(y_true)
    # Recall@K is relevant items in top K / total relevant items
    return relevant_in_top_k / total_relevant_items if total_relevant_items > 0 else 0


def compute_mrr(y_true, y_scores):
    """
    Computes Mean Reciprocal Rank (MRR) for a single query/list.
    """
    # Rank items by score and get the indices
    ranked_indices = np.argsort(y_scores)[::-1]
    # Get the true labels for the ranked items
    ranked_true = np.asarray(y_true)[ranked_indices]
    # Find the rank of the first relevant item
    relevant_ranks = np.where(ranked_true == 1)[0]
    if len(relevant_ranks) == 0:
        return 0.0  # No relevant items, MRR is 0
    else:
        return 1.0 / (relevant_ranks[0] + 1)  # +1 for 1-based rank


def compute_ndcg(ground_truth, pred_cids, pred_scores, K):
    x = np.zeros(100000) #NOTE: hardcoded to max corpus size. Okay as long las max(pred_cids) < 100000
    x[ground_truth]=1
    # handling edge cases
    if len(pred_cids)==0:
        return 0
    elif len(pred_cids)==1:
        return x[pred_cids][0]  # if only one item, return corresponding binary label as  ndcg score
    #another edge case
    # valid_pos = ~np.isnan(pred_scores)
    valid_pos = np.isfinite(pred_scores)
    pred_cids = np.array(pred_cids)[valid_pos]
    pred_scores = np.array(pred_scores)[valid_pos]
    return ndcg_score([x[pred_cids].tolist()], [pred_scores], k=K)


def custom_ap(ground_truth, pred_cids, pred_scores, K, len_gt = None):
    """
        ground_truth : set of relevant corpus ids
        pred_cids : list of predicted corpus ids
        pred_scores: list of predicted scores for the pred_cids
        K : required top K items (only needed to check and throw exception)
    """
    if K>=0:
        try:
            assert len(pred_cids)==K
        except Exception as e:
            logger.exception(e)
            logger.info(f"# ground truth={len(ground_truth)}, # preds={len(pred_cids)}")

    sorted_pred_scores = sorted(((e, i) for i, e in enumerate(pred_scores)), reverse=True)
    sum_precision= 0
    positive_count = 0
    position_count = 0
    for sc, idx in sorted_pred_scores:
        position_count += 1
        #check if label=1
        if pred_cids[idx] in ground_truth:
            positive_count += 1
            sum_precision += (positive_count / position_count)
    if len_gt is not None:
        average_precision = sum_precision / len_gt
    else:
        average_precision = sum_precision / len(ground_truth)
    return average_precision


def custom_ap_at_K(ground_truth, pred_cids, pred_scores, K, len_gt=None):
    """
        ground_truth : set of relevant corpus ids
        pred_cids : list of predicted corpus ids
        pred_scores: list of predicted scores for the pred_cids
        K : required top K items (only needed to check and throw exception)
    """
#     if K>=0:
#         try:
#             assert len(pred_cids)==K
#         except Exception as e:
#             logger.exception(e)
#             logger.info(f"# ground truth={len(ground_truth)}, # preds={len(pred_cids)}")
    assert (K > 0)
    sorted_pred_scores = sorted(((e, i) for i, e in enumerate(pred_scores)), reverse=True)[:K]
    sum_precision = 0
    positive_count = 0
    position_count = 0
    for sc, idx in sorted_pred_scores:
        position_count += 1
        #check if label=1
        if pred_cids[idx] in ground_truth:
            positive_count += 1
            sum_precision += (positive_count / position_count)
    if positive_count == 0:
        average_precision = 0
    else:
        average_precision = sum_precision / positive_count
#     if len_gt is not None: 
#         average_precision = sum_precision/len_gt
#     else:
#         average_precision = sum_precision/len(ground_truth)
    return average_precision


def pairwise_ranking_loss(pred_pos, pred_neg, margin):
    expanded_pred_pos = pred_pos.unsqueeze(1)
    expanded_pred_neg = pred_neg.unsqueeze(0)
    relu = torch.nn.ReLU()
    loss = relu(margin + expanded_pred_neg - expanded_pred_pos)
    mean_loss = torch.mean(loss, dim=(0, 1))

    return mean_loss


def evaluate_model_faster(conf, model, dataset):
    model.eval()
    with torch.no_grad():
        model.fetch_embed = True

        if conf.model.name == "NANL":
            all_scores = nanl_fast_inference(conf, model, dataset)
            map_score = gen_map_from_scores(all_scores, dataset.all_gt)
            ap_score = average_precision_score( dataset.all_gt, all_scores.reshape(-1).detach().cpu())
        elif conf.model.name == "NANLSIG":
            all_scores = nanl_fast_inference(conf, model, dataset, use_sig=True)
            map_score = gen_map_from_scores(all_scores, dataset.all_gt)
            ap_score = average_precision_score( dataset.all_gt, all_scores.reshape(-1).detach().cpu())
        else:
            q_embeds = model(dataset._pack_batch_1d(dataset.query_graphs),dataset.query_graph_node_sizes,dataset.query_graph_edge_sizes)
            c_embeds = model(dataset._pack_batch_1d(dataset.corpus_graphs),dataset.corpus_graph_node_sizes,dataset.corpus_graph_edge_sizes)
            if conf.model.scoring_function == "sighinge":
                map_score, all_preds = gen_map_from_embeds(conf, q_embeds, c_embeds, dataset.all_gt, model.sigmoid_a, model.sigmoid_b)  
            else:
                map_score, all_preds = gen_map_from_embeds(conf, q_embeds, c_embeds, dataset.all_gt)
            ap_score = average_precision_score(dataset.all_gt, all_preds)
            
        model.fetch_embed = False
    
    return ap_score, map_score
         

def compute_average_precision(
    model,
    pos_pairs,
    neg_pairs,
    dataset,
    return_pred_and_labels=False,
    return_running_time=False,
):
    assert not(return_running_time and return_pred_and_labels)
    all_pairs = pos_pairs + neg_pairs
    num_pos_pairs, num_neg_pairs = len(pos_pairs), len(neg_pairs)

    if return_running_time:
        total_running_time = 0
        total_batches = 0
    predictions = []
    num_batches = dataset.create_eval_batches(all_pairs)
    for batch_idx in range(num_batches):
        batch_graphs, batch_graph_node_sizes, batch_graph_edge_sizes, _ = dataset.fetch_batch_by_id(batch_idx)

        if return_running_time:
            start_time = time.time()

        model_output = model(batch_graphs, batch_graph_node_sizes, batch_graph_edge_sizes)

        if return_running_time:
            end_time = time.time()
            total_running_time += end_time - start_time
            total_batches += 1

        predictions.append(model_output.data)
    all_predictions = torch.cat(predictions, dim=0)
    all_labels = torch.cat([torch.ones(num_pos_pairs), torch.zeros(num_neg_pairs)])

    average_precision = average_precision_score(all_labels, all_predictions.cpu())
    if return_pred_and_labels:
        return average_precision, all_labels, all_predictions
    elif return_running_time:
        return average_precision, total_running_time, total_batches
    else:
        return average_precision


@torch.no_grad()
def compute_ap_map_scores(conf, q_logits, c_logits, all_gt):
    """
    q_logits: [Q, H]  pre-computed model outputs
    c_logits: [C, H]  pre-computed model outputs
    all_gt: [Q, C] binary relevance matrix (float, on device)
    """
    Q, C = q_logits.shape[0], c_logits.shape[0]
    all_gt = all_gt.reshape(Q, C).to(q_logits.device).float()

    # Raw scores
    raw_scores = q_logits @ c_logits.T   # [Q, C]

    # Sign-code scores (inference-style)
    q_sign = torch.sign(q_logits)
    c_sign = torch.sign(c_logits)
    q_sign[q_sign == 0] = 1
    c_sign[c_sign == 0] = 1
    sign_scores = q_sign @ c_sign.T

    # Tanh-code scores (smooth surrogate)
    temp = conf.lsh.tanh_temp
    q_tanh = torch.tanh(temp * q_logits)
    c_tanh = torch.tanh(temp * c_logits)
    tanh_scores = q_tanh @ c_tanh.T

    labels_flat = all_gt.reshape(-1).detach().cpu().numpy()
    raw_flat = raw_scores.reshape(-1).detach().cpu().numpy()
    sign_flat = sign_scores.reshape(-1).detach().cpu().numpy()
    tanh_flat = tanh_scores.reshape(-1).detach().cpu().numpy()

    raw_ap = average_precision_score(labels_flat, raw_flat)
    sign_ap = average_precision_score(labels_flat, sign_flat)
    tanh_ap = average_precision_score(labels_flat, tanh_flat)

    raw_all_ap, sign_all_ap, tanh_all_ap = [], [], []
    for i in range(Q):
        q_labels = all_gt[i].detach().cpu().numpy()
        if q_labels.sum() == 0 or q_labels.sum() == len(q_labels):
            continue
        q_raw = raw_scores[i].detach().cpu().numpy()
        q_sign = sign_scores[i].detach().cpu().numpy()
        q_tanh = tanh_scores[i].detach().cpu().numpy()
        raw_all_ap.append(average_precision_score(q_labels, q_raw))
        sign_all_ap.append(average_precision_score(q_labels, q_sign))
        tanh_all_ap.append(average_precision_score(q_labels, q_tanh))

    raw_map = float(np.mean(raw_all_ap)) if len(raw_all_ap) > 0 else 0.0
    sign_map = float(np.mean(sign_all_ap)) if len(sign_all_ap) > 0 else 0.0
    tanh_map = float(np.mean(tanh_all_ap)) if len(tanh_all_ap) > 0 else 0.0

    return {
        "raw_ap": float(raw_ap),
        "raw_map": raw_map,
        "raw_all_ap": raw_all_ap,
        "sign_ap": float(sign_ap),
        "sign_map": sign_map,
        "sign_all_ap": sign_all_ap,
        "tanh_ap": float(tanh_ap),
        "tanh_map": tanh_map,
        "tanh_all_ap": tanh_all_ap,
        "raw_scores": raw_scores,
        "sign_scores": sign_scores,
        "tanh_scores": tanh_scores,
    }