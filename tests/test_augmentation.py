import unittest
from collections.abc import Callable
from typing import ParamSpec

import pytest
import torch
from torch import Tensor

from reform.aug_embedder import (
    get_long_augmented_embs,
    get_long_augmented_embs_slow,
    get_short_augmented_embs,
    get_short_augmented_embs_slow,
)

CASES = [
    (2, 2, 3, 4),
    (8, 10, 10, 4),
    (8, 32, 64, 10),
]


def make_inputs(
    B: int, d: int, d1: int, device: torch.device | None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.randn(B, d1, device=device),
        torch.randn(B, d1, device=device),
        torch.randn(d, d1, device=device),
        torch.randn(d, d1, device=device),
    )


def assert_stuff(q1: Tensor, c1: Tensor, q2: Tensor, c2: Tensor, shape: tuple):
    assert q1.shape == shape
    assert c1.shape == shape
    assert q2.shape == shape
    assert c2.shape == shape
    assert torch.allclose(q1, q2, atol=1e-5)
    assert torch.allclose(c1, c2, atol=1e-5)


class TestAugmentation(unittest.TestCase):
    @torch.inference_mode()
    def run_long_case(self, B: int, d: int, d1: int, R: int):
        q, c, wt, w0 = make_inputs(B, d, d1)
        hyperplanes = torch.randn(R, 2 + d1)
        q1, c1 = get_long_augmented_embs(q, c, wt, w0, hyperplanes)
        q2, c2 = get_long_augmented_embs_slow(q, c, wt, w0, hyperplanes)
        assert_stuff(q1, c1, q2, c2, (B, d, R * 2 * (2 + d1)))

    @torch.inference_mode()
    def run_short_case(self, B: int, d: int, d1: int, R: int):
        q, c, wt, _ = make_inputs(B, d, d1)
        hyperplanes = torch.randn(R, 1 + d1)
        q1, c1 = get_short_augmented_embs(q, c, wt, hyperplanes)
        q2, c2 = get_short_augmented_embs_slow(q, c, wt, hyperplanes)
        assert_stuff(q1, c1, q2, c2, (B, d, R * 2 * (1 + d1)))

    def test_long_vectorization(self):
        for B, d, d1, R in CASES:
            with self.subTest(B=B, d=d, d1=d1, R=R):
                for _ in range(5):
                    self.run_long_case(B, d, d1, R)

    def test_short_vectorization(self):
        for B, d, d1, R in CASES:
            with self.subTest(B=B, d=d, d1=d1, R=R):
                for _ in range(5):
                    self.run_short_case(B, d, d1, R)

    def test_long_correctness(self):
        pytest.fail("TODO: Not implemented")

    def test_short_correctness(self):
        pytest.fail("TODO: Not implemented")


LONG_FNS = [
    pytest.param(get_long_augmented_embs_slow, id="for-loop"),
    pytest.param(torch.compile(get_long_augmented_embs_slow), id="for-loop-compiled"),
    pytest.param(get_long_augmented_embs, id="tensorized"),
    pytest.param(torch.compile(get_long_augmented_embs), id="tensorized-compiled"),
]
SHORT_FNS = [
    pytest.param(get_short_augmented_embs_slow, id="for-loop"),
    pytest.param(torch.compile(get_short_augmented_embs_slow), id="for-loop-compiled"),
    pytest.param(get_short_augmented_embs, id="tensorized"),
    pytest.param(torch.compile(get_short_augmented_embs), id="tensorized-compiled"),
]

# This code was written for ancient Python which does not support the [**P] syntax
P = ParamSpec("P")


def benchmark_fn(
    benchmark,
    device: torch.device,
    fn: Callable[P, tuple[Tensor, Tensor]],
    *args: P.args,
    **kwargs: P.kwargs,
):
    def run():
        result = fn(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return result

    benchmark.warmup_iterations = 3
    benchmark(run)


BENCHMARK_SHAPE = (8, 10, 10, 4)


@pytest.mark.parametrize("fn", LONG_FNS)
@torch.inference_mode()
def test_benchmark_long(
    benchmark,
    device: torch.device,
    fn: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
):
    B, d, d1, R = BENCHMARK_SHAPE
    q, c, wt, w0 = make_inputs(B, d, d1, device=device)
    hyperplanes = torch.randn(R, 2 + d1, device=device)
    benchmark.group = f"long_B{B}_d{d}_d1{d1}_R{R}"
    benchmark_fn(benchmark, device, fn, q, c, wt, w0, hyperplanes)


@pytest.mark.parametrize("fn", SHORT_FNS)
@torch.inference_mode()
def test_benchmark_short(
    benchmark,
    device: torch.device,
    fn: Callable[[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
):
    B, d, d1, R = BENCHMARK_SHAPE
    q, c, wt, _ = make_inputs(B, d, d1, device=device)
    hyperplanes = torch.randn(R, 1 + d1, device=device)
    benchmark.group = f"short_B{B}_d{d}_d1{d1}_R{R}"
    benchmark_fn(benchmark, device, fn, q, c, wt, hyperplanes)


if __name__ == "__main__":
    unittest.main()
