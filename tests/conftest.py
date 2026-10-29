import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--device", default="cpu", help="device for benchmarks")


@pytest.fixture(scope="session")
def device(request) -> torch.device:
    return torch.device(request.config.getoption("--device"))
