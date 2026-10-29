import math
from pathlib import Path

import torch
from torch import Tensor


def augment(x: Tensor, hyperplanes: Tensor) -> Tensor:
    B, d, _ = x.shape
    R, L = hyperplanes.shape
    x = x.repeat_interleave(R, dim=1).repeat(1, 1, 2)
    x[:, :, L:] *= torch.sign(
        torch.sum(hyperplanes.repeat(d, 1) * x[:, :, L:], dim=-1, keepdim=True)
    )
    x /= math.sqrt(2)
    return x.view(B, d, R * 2 * L)


def get_long_augmented_embs(
    q: Tensor, c: Tensor, wt: Tensor, w0: Tensor, hyperplanes: Tensor, w0_q: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """
    q is phi(q) in the paper and c is phi(c)

    q, c: (_, d1)
    wt, w0: (d, d1)
    w0_q: (d, d1) — override for w0 used only in q_hat; defaults to w0 if None
    hyperplanes: (R, 2 + d1)
    """
    d, _ = wt.shape
    if w0_q is None:
        w0_q = w0

    B, d1 = q.shape
    q_hat = augment(
        torch.cat(
            (
                wt @ q.view(B, d1, 1),
                torch.full((B, d, 1), -1, device=q.device),
                -(wt - w0_q).expand(B, d, d1),
            ),
            dim=-1,
        ),
        hyperplanes,
    )

    B, _ = c.shape
    c_hat = augment(
        torch.cat(
            (
                torch.ones(B, d, 1, device=q.device),
                w0 @ c.view(B, d1, 1),
                c.view(B, 1, d1).expand(B, d, d1),
            ),
            dim=-1,
        ),
        hyperplanes
    )

    return q_hat, c_hat


def get_long_augmented_embs_slow(
    q: Tensor, c: Tensor, wt: Tensor, w0: Tensor, hyperplanes: Tensor
) -> tuple[Tensor, Tensor]:
    B, d1 = q.shape
    d, _ = wt.shape
    R, _ = hyperplanes.shape

    q_embs, c_embs = [], []
    for i in range(d):
        q_hat_i = torch.cat(
            (
                (wt[i].view(1, d1) @ q.view(B, d1, 1)).view(B, 1),
                torch.full((B, 1), -1, device=q.device),
                -(wt[i] - w0[i]).expand(B, d1),
            ),
            dim=-1,
        )
        assert q_hat_i.shape == (B, 2 + d1)

        c_hat_i = torch.cat(
            (
                torch.ones(B, 1, device=q.device),
                (w0[i].view(1, d1) @ c.view(B, d1, 1)).view(B, 1),
                c,
            ),
            dim=-1,
        )
        assert c_hat_i.shape == (B, 2 + d1)

        def augment(x: Tensor) -> Tensor:
            return torch.cat(
                [
                    torch.cat(
                        (
                            x,
                            torch.sign(hyperplanes[r].view(1, 2 + d1) @ x.view(B, 2 + d1, 1)).view(
                                B, 1
                            )
                            * x,
                        ),
                        dim=-1,
                    )
                    for r in range(R)
                ],
                dim=-1,
            ).reshape(B, R * 2 * (2 + d1)) / math.sqrt(2)

        q_embs.append(augment(q_hat_i))
        c_embs.append(augment(c_hat_i))

        assert q_embs[-1].shape == (B, R * 2 * (2 + d1))
        assert c_embs[-1].shape == (B, R * 2 * (2 + d1))

    return torch.stack(q_embs, dim=1), torch.stack(c_embs, dim=1)


def get_short_augmented_embs(
    q: Tensor, c: Tensor, wt: Tensor, hyperplanes: Tensor
) -> tuple[Tensor, Tensor]:
    """
    q is phi(q) in the paper and c is phi(c)

    q, c: (_, d1)
    wt, w0: (d, d1)
    hyperplanes: (R, 1 + d1)
    """
    d, _ = wt.shape

    B, d1 = q.shape
    q_hat = augment(torch.cat((wt @ q.view(B, d1, 1), -wt.expand(B, d, d1)), dim=-1), hyperplanes)

    B, _ = c.shape
    c_hat = augment(
        torch.cat(
            (torch.ones(B, d, 1, device=q.device), c.view(B, 1, d1).expand(B, d, d1)), dim=-1
        ),
        hyperplanes,
    )

    return q_hat, c_hat


def get_short_augmented_embs_slow(
    q: Tensor, c: Tensor, wt: Tensor, hyperplanes: Tensor
) -> tuple[Tensor, Tensor]:
    B, d1 = q.shape
    d, _ = wt.shape
    R, _ = hyperplanes.shape

    q_embs, c_embs = [], []
    for i in range(d):
        q_hat_i = torch.cat(
            ((wt[i].view(1, d1) @ q.view(B, d1, 1)).view(B, 1), -wt[i].expand(B, d1)), dim=-1
        )
        assert q_hat_i.shape == (B, 1 + d1)

        c_hat_i = torch.cat((torch.ones(B, 1, device=q.device), c), dim=-1)
        assert c_hat_i.shape == (B, 1 + d1)

        def augment(x: Tensor) -> Tensor:
            return torch.cat(
                [
                    torch.cat(
                        (
                            x,
                            torch.sign(hyperplanes[r].view(1, 1 + d1) @ x.view(B, 1 + d1, 1)).view(
                                B, 1
                            )
                            * x,
                        ),
                        dim=-1,
                    )
                    for r in range(R)
                ],
                dim=-1,
            ).reshape(B, R * 2 * (1 + d1)) / math.sqrt(2)

        q_embs.append(augment(q_hat_i))
        c_embs.append(augment(c_hat_i))

        assert q_embs[-1].shape == (B, R * 2 * (1 + d1))
        assert c_embs[-1].shape == (B, R * 2 * (1 + d1))

    return torch.stack(q_embs, dim=1), torch.stack(c_embs, dim=1)


class AugmentedEmbedder:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_type = config.aug_embedder.aug_form
        self.hyperplane_path = (
            Path(config.base_dir)
            / "allPklDumps"
            / f"greform_hyperplanes_{self.embedding_type}_{config.aug_embedder.num_hyperplanes}.pt"
        )
        self.hyperplanes = torch.load(self.hyperplane_path, weights_only=False).to(self.device)

    def get_embs(
        self, query_embs: Tensor, corpus_embs: Tensor, curr_params: Tensor, init_params: Tensor,
        init_params_q: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        q, c: (_, d1) - pre-last-layer outputs from the graph model
        curr_params, init_params: (d, d1)
        init_params_q: (d, d1) — override for init_params used only in q_hat (long form only)
        """
        match self.embedding_type:
            case "long":
                return get_long_augmented_embs(
                    query_embs, corpus_embs, curr_params, init_params, self.hyperplanes,
                    w0_q=init_params_q,
                )
            case "short":
                return get_short_augmented_embs(
                    query_embs, corpus_embs, curr_params, self.hyperplanes
                )
            case _:
                msg = "Invalid augmentation type"
                raise RuntimeError(msg)
