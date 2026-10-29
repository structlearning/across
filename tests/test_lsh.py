import unittest

import numpy as np
import torch
from numpy.random import Generator
from torch import Tensor

from reform.lsh import LSH, create_bit_indices


def create_lsh(
    hashcode_dim: int,
    subset_size: int,
    corpus: Tensor,
    rng: Generator,
) -> LSH:
    gen_embd_dim, _, mlp_output_dim = corpus.size()
    bit_indices = create_bit_indices(hashcode_dim, subset_size, gen_embd_dim, rng)
    return LSH(
        torch.randn(gen_embd_dim, hashcode_dim, mlp_output_dim),
        torch.randn(gen_embd_dim, hashcode_dim),
        bit_indices,
        corpus,
    )


class TestLSH(unittest.TestCase):
    def test_one(self):
        for _ in range(5):
            rng = np.random.default_rng()

            C = 10_000
            gen_embd_dim = 10
            mlp_output_dim = 12
            corpus = torch.randn(gen_embd_dim, C, mlp_output_dim)
            hashcode_dim = 32
            subset_size = 11

            lsh = create_lsh(hashcode_dim, subset_size, corpus, rng)

            for _ in range(10):
                c_id = rng.integers(C)
                assert c_id in lsh.retrieve(corpus[:, c_id])

    def test_one_bucket(self):
        for _ in range(5):
            rng = np.random.default_rng()

            C = 10_000
            gen_embd_dim = 10
            mlp_output_dim = 12
            corpus = torch.randn(gen_embd_dim, C, mlp_output_dim)
            hashcode_dim = 32
            subset_size = 0

            lsh = create_lsh(hashcode_dim, subset_size, corpus, rng)

            for _ in range(10):
                c_id = rng.integers(C)
                assert len(lsh.retrieve(corpus[:, c_id])) == C

    def test_many(self):
        for _ in range(100):
            rng = np.random.default_rng()

            C = 10_000
            gen_embd_dim = 10
            mlp_output_dim = 12
            corpus = torch.randn(gen_embd_dim, C, mlp_output_dim)
            hashcode_dim = 32
            subset_size = rng.integers(hashcode_dim + 1)

            K = rng.integers(1, 100)
            c_ids = rng.choice(C, size=(K,), replace=False)
            embd_to_copy = corpus[:, c_ids[rng.integers(K)]].clone()
            corpus[:, c_ids] = embd_to_copy.unsqueeze(1)

            lsh = create_lsh(hashcode_dim, subset_size, corpus, rng)

            for _ in range(10):
                assert set(c_ids).issubset(lsh.retrieve(corpus[:, c_ids[rng.integers(K)]]))


if __name__ == "__main__":
    unittest.main()
