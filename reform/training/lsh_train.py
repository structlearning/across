import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from loguru import logger

import wandb
from reform.training.base import Trainer
# from reform.utils.training_utils import compute_ap_map_scores


def reg_losses(all_hcodes):
    """
    all_hcodes: D x N x d
    Returns (fence_sitting, bit_balance), both D-dim.
    """
    N, d = all_hcodes.shape[1], all_hcodes.shape[2]
    fence_sitting = torch.norm(all_hcodes.abs() - 1, p=1, dim=[1, 2]) / (N * d)
    bit_balance = torch.sum(torch.abs(torch.sum(all_hcodes, dim=1)), dim=-1) / (N * d)
    return fence_sitting, bit_balance


def pairwise_ranking_loss_similarity(predPos, predNeg, margin):
    """
    predPos: (n_pos, D)  — positive scores for each of D trials
    predNeg: (n_neg, D)  — negative scores for each of D trials
    Returns D-dim loss vector (normalised by n_pos * n_neg).
    """
    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, -1)  # (n_pos, n_neg, D)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, -1)  # (n_pos, n_neg, D)
    ell = margin + expanded_2 - expanded_1
    sum_loss = torch.sum(F.relu(ell), dim=[0, 1])            # D-dim
    return sum_loss / (n_1 * n_2)


class LSHTrainer(Trainer):
    def __init__(self, conf, model, optimizer, q_hats, x_hats):
        super().__init__(conf, model, optimizer)

        # Q x D x d, C x D x d
        self.q_hats = q_hats
        self.x_hats = x_hats

    def train_iter(self):
        self.model.train()

        n_batches = self.train_dataset.create_stratified_batches()
        sub_factor = self.conf.early_stopping.patience // 2
        num_sub_batches = n_batches // sub_factor

        self.epoch_loss = 0
        training_start_time = time.time()

        for batch_idx in tqdm.tqdm(range(n_batches)):
            q_idxs, c_idxs, labels = \
                self.train_dataset.fetch_batch_by_id(batch_idx)

            self.optimizer.zero_grad()
            q_hat = self.q_hats["train"][q_idxs]
            c_hat = self.x_hats["train"][c_idxs]

            # Compute the losses here with conf.lsh.lambda_fence_sit, conf.lsh.lambda_coll_min,
            # conf.lsh.lambda_bit_bal
            loss, bit_balance_loss, third_loss, fence_sitting_loss =\
                self.compute_loss(q_hat, c_hat, labels)

            loss.backward()
            self.optimizer.step()
            self.epoch_loss += loss.item()

            if self.wandb_enabled:
                wandb.log({
                    "train_loss_batch": loss.item(),
                    "bit_balance_loss_batch": bit_balance_loss.sum().item(),
                    "third_loss_batch": third_loss.sum().item(),
                    "fence_sitting_loss_batch": fence_sitting_loss.sum().item()
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

    def compute_loss(self, q_hat, c_hat, labels):
        # q_hat, c_hat: (Q, D, d), (C, D, d)
        # During training, Q = C = B (batch size)
        q_hcodes = self.model(q_hat)
        c_hcodes = self.model(c_hat)

        q_hcodes = torch.tanh(self.conf.lsh.tanh_temp * q_hcodes)
        c_hcodes = torch.tanh(self.conf.lsh.tanh_temp * c_hcodes)

        all_hcodes = torch.cat([q_hcodes, c_hcodes], dim=0)
        all_hcodes = all_hcodes.permute(1, 0, 2)  # D x (Q + C) x d
        fence_sitting_loss, bit_balance_loss = reg_losses(all_hcodes)

        # B x D x d @ B x D x d -> B x D
        preds = (q_hcodes * c_hcodes).sum(dim=-1)

        pos_mask = labels > 0.5
        neg_mask = labels < 0.5
        pred_pos = preds[pos_mask]  # (n_pos, D)
        pred_neg = preds[neg_mask]  # (n_neg, D)

        third_loss_type = self.conf.lsh.third_loss_type
        if third_loss_type == 'ad_decorr':
            # Asymmetric decorrelation: sg(z_q) · z_c for positives, normalized by n_pos
            preds_ad = (q_hcodes.detach() * c_hcodes).sum(dim=-1)  # (B, D)
            pred_pos_ad = preds_ad[pos_mask]  # (n_pos, D)
            third_loss = -pred_pos_ad.mean(dim=0) if pred_pos_ad.shape[0] > 0 \
                else preds.new_zeros(preds.shape[1])
        else:
            # D-dim, chunked over pred_pos to bound (chunk x n_neg x D) expansion
            chunk_size = getattr(self.conf.lsh, 'chunk_size', 50)
            total_ranking_loss = preds.new_zeros(preds.shape[1])
            total_num_pairs = 0

            for chunk_start in range(0, len(pred_pos), chunk_size):
                pos_chunk = pred_pos[chunk_start:chunk_start + chunk_size]  # (cs, D)
                n_pairs = pos_chunk.shape[0] * pred_neg.shape[0]
                total_ranking_loss = total_ranking_loss + pairwise_ranking_loss_similarity(
                    pos_chunk, pred_neg, self.conf.lsh.coll_min_margin
                ) * n_pairs
                total_num_pairs += n_pairs

            ranking_loss = (total_ranking_loss / total_num_pairs) if total_num_pairs > 0 \
                else preds.new_zeros(preds.shape[1])
            third_loss = ranking_loss

        loss = (
            self.conf.lsh.lambda_fence_sit * fence_sitting_loss
            + self.conf.lsh.lambda_bit_bal * bit_balance_loss
            + self.conf.lsh.lambda_coll_min * third_loss
        )
        loss = loss.sum(dim=-1)

        return loss, bit_balance_loss, third_loss, fence_sitting_loss

    def _validate_common(self, which="val"):
        if which == "val":
            all_gt = self.val_dataset.all_gt
        elif which == "test":
            all_gt = self.test_dataset.all_gt

        q_hat = self.q_hats[which]
        c_hat = self.x_hats[which]
        Q = q_hat.shape[0]
        C = c_hat.shape[0]

        all_gt = all_gt.reshape(Q, C).to(q_hat.device).float()

        return q_hat, c_hat, all_gt

    @torch.no_grad()
    def _validate(self, which="val"):
        q_hat, c_hat, all_gt = self._validate_common(which=which)

        logger.info(f"Validation: q_hat shape: {q_hat.shape}, c_hat shape: {c_hat.shape}, all_gt shape: {all_gt.shape}")

        # Hash network outputs: [Q/C, D, H]
        q_logits = self.model(q_hat)
        c_logits = self.model(c_hat)

        temp = self.conf.lsh.tanh_temp
        q_hcodes = torch.tanh(temp * q_logits)
        c_hcodes = torch.tanh(temp * c_logits)

        # Global fence-sitting and bit-balance over all codes
        all_hcodes = torch.cat([q_hcodes, c_hcodes], dim=0)       # (Q+C) x D x H
        fence_sitting_loss, bit_balance_loss = reg_losses(
            all_hcodes.permute(1, 0, 2)                            # D x (Q+C) x H
        )
        fence_sitting_loss = fence_sitting_loss.sum()
        bit_balance_loss = bit_balance_loss.sum()

        # Ranking loss: chunked over queries to avoid Q x D x C allocation
        Q = q_hcodes.shape[0]
        chunk_size = getattr(self.conf.lsh, 'chunk_size', 50)
        third_loss_type = self.conf.lsh.third_loss_type

        if third_loss_type == 'ad_decorr':
            pos_q, pos_c = (all_gt > 0.5).nonzero(as_tuple=True)  # each (n_pos_total,)
            if pos_q.shape[0] > 0:
                scores = (q_hcodes[pos_q].detach() * c_hcodes[pos_c]).sum(dim=-1)  # (n_pos_total, D)
                third_loss = -scores.mean(dim=0).sum()
            else:
                third_loss = q_hcodes.new_zeros(())
        else:
            total_ranking_loss = q_hcodes.new_zeros(q_hcodes.shape[1])  # D-dim
            total_num_pairs = 0

            for chunk_start in range(0, Q, chunk_size):
                q_chunk = q_hcodes[chunk_start:chunk_start + chunk_size]   # (cs, D, H)
                gt_chunk = all_gt[chunk_start:chunk_start + chunk_size]    # (cs, C)
                # (cs, D, C) — only this chunk is live at a time
                scores_chunk = torch.einsum('qdi,cdi->qdc', q_chunk, c_hcodes)

                for i in range(scores_chunk.shape[0]):
                    pos = scores_chunk[i, :, gt_chunk[i] > 0.5]   # [D, P_i]
                    neg = scores_chunk[i, :, gt_chunk[i] < 0.5]   # [D, N_i]

                    if pos.numel() == 0 or neg.numel() == 0:
                        continue

                    # Chunk over negatives: bounds (P_i x neg_chunk x D) expansion
                    for neg_start in range(0, neg.shape[1], chunk_size):
                        neg_c = neg[:, neg_start:neg_start + chunk_size]  # [D, cs]
                        n_pairs = pos.shape[1] * neg_c.shape[1]
                        total_ranking_loss = total_ranking_loss + pairwise_ranking_loss_similarity(
                            pos.T, neg_c.T, self.conf.lsh.coll_min_margin
                        ) * n_pairs
                        total_num_pairs += n_pairs

            ranking_loss = (total_ranking_loss / total_num_pairs).sum() if total_num_pairs > 0 \
                else q_hcodes.new_zeros(())
            third_loss = ranking_loss

        loss = (
            self.conf.lsh.lambda_fence_sit * fence_sitting_loss
            + self.conf.lsh.lambda_bit_bal * bit_balance_loss
            + self.conf.lsh.lambda_coll_min * third_loss
        )

        return {
            f"{which}_loss": loss.item(),
            f"{which}_third_loss": third_loss.item(),
            f"{which}_fence_sitting_loss": fence_sitting_loss.item(),
            f"{which}_bit_balance_loss": bit_balance_loss.item(),
        }

    def _eval_split(self, which: str):
        self.model.eval()
        loss_dict = self._validate(which=which)

        # metric_dict = compute_ap_map_scores(
        #     self.conf, loss_dict["q_logits"], loss_dict["c_logits"], loss_dict["all_gt"]
        # )
        logger.info(
            f"{which.upper()} "
            f"loss: {loss_dict[f'{which}_loss']:.6f} "
            f"third: {loss_dict[f'{which}_third_loss']:.6f} "
            f"fence: {loss_dict[f'{which}_fence_sitting_loss']:.6f} "
            f"bitbal: {loss_dict[f'{which}_bit_balance_loss']:.6f} "
            # f"raw_ap: {metric_dict['raw_ap']:.6f} raw_map: {metric_dict['raw_map']:.6f} "
            # f"sign_ap: {metric_dict['sign_ap']:.6f} sign_map: {metric_dict['sign_map']:.6f} "
            # f"tanh_ap: {metric_dict['tanh_ap']:.6f} tanh_map: {metric_dict['tanh_map']:.6f}"
        )
        return loss_dict

    def validate_iter(self):
        logger.info(f"Validating (run {self.run})")
        loss_dict = self._eval_split("val")

        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "epoch": self.run,

            "val_loss": loss_dict["val_loss"],

            "rng_state": torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'random_state': random.getstate(),
            'patience': self.es.patience,
            'best_scores': self.es.best_scores,
            'num_bad_epochs': self.es.num_bad_epochs,
            'should_stop_now': self.es.should_stop_now,
        }

        if self.conf.lsh.es_metric == "val_loss":
            score_for_es = -loss_dict["val_loss"]
        else:
            raise ValueError(f"Unsupported early stopping metric: {self.conf.lsh.es_metric}")

        state_dict = self.es.check([score_for_es], state_dict)

        if self.wandb_enabled:
            wandb.log({
                "train_loss": self.epoch_loss,
                "train_time": self.epoch_training_time,

                "val_loss": loss_dict["val_loss"],
                "val_third_loss": loss_dict["val_third_loss"],
                "val_fence_sitting_loss": loss_dict["val_fence_sitting_loss"],
                "val_bit_balance_loss": loss_dict["val_bit_balance_loss"],

                # "val_raw_ap": metric_dict["raw_ap"],
                # "val_raw_map": metric_dict["raw_map"],
                # "val_sign_ap": metric_dict["sign_ap"],
                # "val_sign_map": metric_dict["sign_map"],
                # "val_tanh_ap": metric_dict["tanh_ap"],
                # "val_tanh_map": metric_dict["tanh_map"],
            })

    def post_train(self):
        ckpt = self.es.load_best_model()
        self.model.load_state_dict(ckpt["model_state_dict"])

        logger.info("Testing")
        loss_dict = self._eval_split("test")

        if self.wandb_enabled:
            wandb.log({
                "test_loss": loss_dict["test_loss"],
                "test_third_loss": loss_dict["test_third_loss"],
                "test_fence_sitting_loss": loss_dict["test_fence_sitting_loss"],
                "test_bit_balance_loss": loss_dict["test_bit_balance_loss"],

                # "test_raw_ap": metric_dict["raw_ap"],
                # "test_raw_map": metric_dict["raw_map"],
                # "test_sign_ap": metric_dict["sign_ap"],
                # "test_sign_map": metric_dict["sign_map"],
                # "test_tanh_ap": metric_dict["tanh_ap"],
                # "test_tanh_map": metric_dict["tanh_map"],
            })