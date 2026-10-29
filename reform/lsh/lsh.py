import typing
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from numpy.random import Generator
from numpy.typing import NDArray
from torch import Tensor

# Ancient Python 3.10 does not have the `type` keyword
IntArray = NDArray[np.int32]
FloatArray = NDArray[np.float32]


def create_bit_indices(
    hashcode_dim: int,
    subset_size: int,
    gen_embd_dim: int,
    rng: Generator,
    *,
    same_idxs_across_dims: bool = False,
) -> IntArray:
    assert hashcode_dim >= subset_size

    def sample_idxs():
        return rng.choice(hashcode_dim, subset_size, replace=False)

    if same_idxs_across_dims:
        idxs = sample_idxs()
        return np.stack([idxs for _ in range(gen_embd_dim)], axis=0)

    return np.stack([sample_idxs() for _ in range(gen_embd_dim)], axis=0)


class LSH:
    def __init__(
        self, hp_weights: Tensor, hp_biases: Tensor, bit_indices: IntArray, corpus: Tensor
    ):
        """
        Args:
            hp_weights:
                (hashcode_dim, mlp_output_dim) or (gen_embd_dim, hashcode_dim, mlp_output_dim)
            hp_biases: (hashcode_dim,) or (gen_embd_dim, hashcode_dim)
            bit_indices: (gen_embd_dim, log_2(bucket_count))
            corpus: (gen_embd_dim, |C|, mlp_output_dim)
        """
        self.hp_weights = hp_weights
        self.hp_biases = hp_biases
        self.bit_indices = bit_indices
        gen_embd_dim, subset_size = bit_indices.shape
        self.gen_embd_dim = gen_embd_dim
        self.powers_of_two = 1 << np.arange(subset_size - 1, -1, -1)
        assert hp_weights.dim() == 2 or hp_weights.size(0) == gen_embd_dim, (
            f"{hp_weights.size()=}, {gen_embd_dim=}"
        )
        assert hp_weights.size(-2) >= subset_size, f"{hp_weights.size()}, {subset_size=}"
        assert hp_weights.size(-1) == corpus.size(2), f"{hp_weights.size()}, {corpus.size()}"
        assert hp_biases.dim() == 1 or hp_biases.size(0) == gen_embd_dim, (
            f"{hp_biases.size()=}, {gen_embd_dim=}"
        )
        assert hp_biases.size(-1) == hp_weights.size(-2), (
            f"{hp_biases.size()=}, {hp_weights.size()=}"
        )
        self.hash_tables: list[defaultdict[int, list[int]]] = [
            defaultdict(list) for _ in range(self.gen_embd_dim)
        ]
        self.build_index(corpus)

    @classmethod
    def from_random(
        cls,
        hashcode_dim: int,
        embd_dim: int,
        bit_indices: IntArray,
        corpus: Tensor,
        rng: Generator,
        *,
        linear_every_dim: bool = False,
    ) -> "LSH":
        """Random hyperplane LSH: sample Gaussian normal vectors as hyperplanes."""
        gen_embd_dim = bit_indices.shape[0]
        if linear_every_dim:
            hp_weights = torch.tensor(
                rng.standard_normal((gen_embd_dim, hashcode_dim, embd_dim)), dtype=torch.float32
            )
            hp_biases = torch.zeros(gen_embd_dim, hashcode_dim)
        else:
            hp_weights = torch.tensor(
                rng.standard_normal((hashcode_dim, embd_dim)), dtype=torch.float32
            )
            hp_biases = torch.zeros(hashcode_dim)
        return cls(hp_weights, hp_biases, bit_indices, corpus)

    @classmethod
    def from_model(
        cls, model: nn.ModuleList | nn.Linear, bit_indices: IntArray, corpus: Tensor
    ) -> "LSH":  # no typing.Self in ancient Python 3.10
        hp_weights: Tensor
        hp_biases: Tensor
        if isinstance(model, nn.ModuleList):
            hp_weights = torch.stack([
                typing.cast(nn.Linear, layer).weight for layer in model
            ])
            hp_biases = torch.stack([
                typing.cast(nn.Linear, layer).bias for layer in model
            ])
        else:
            hp_weights = model.weight
            hp_biases = model.bias

        return cls(hp_weights.detach(), hp_biases.detach(), bit_indices, corpus)

    def hashcodes(self, embeddings: Tensor) -> IntArray:
        """
        Args:
            embeddings: (gen_embd_dim, N, mlp_output_dim)
        Returns:
            {0, 1}^(gen_embd_dim x N x hashcode_dim)
        """
        projections = embeddings @ self.hp_weights.transpose(-1, -2) + self.hp_biases.unsqueeze(-2)
        codes = torch.empty(projections.shape, dtype=torch.int32)
        torch.gt(projections, 0, out=codes)
        return codes.numpy()

    def bucket_ids(self, hashcodes: IntArray) -> IntArray:
        """
        Args:
            hashcodes: {0, 1}^(gen_embd_dim x N x hashcode_dim)
        Returns:
            (gen_embd_dim, N)
        """
        N = hashcodes.shape[1]
        bucket_ids = np.empty((self.gen_embd_dim, N), dtype=np.int32)
        for t, indices in enumerate(self.bit_indices):
            bucket_ids[t] = hashcodes[t][:, indices] @ self.powers_of_two
        return bucket_ids

    def build_index(self, corpus: Tensor):
        """corpus: (gen_embd_dim, |C|, mlp_output_dim)"""
        for hash_table, bucket_ids in zip(
            self.hash_tables, self.bucket_ids(self.hashcodes(corpus)), strict=True
        ):
            for corpus_id, bucket_id in enumerate(bucket_ids):
                hash_table[bucket_id].append(corpus_id)

    def retrieve(self, query: Tensor) -> set[int]:
        """
        Args:
            query: (gen_embd_dim, mlp_output_dim)
        """
        q_bucket_ids = self.bucket_ids(self.hashcodes(query.unsqueeze(1)))  # (gen_embd_dim, 1)
        candidates: set[int] = set()
        for hash_table, bucket_ids in zip(self.hash_tables, q_bucket_ids, strict=True):
            candidates.update(hash_table[bucket_ids[0]])
        return candidates
