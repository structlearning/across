import random
import numpy as np
import torch

from pathlib import Path


def binary(x, bits):
    mask = 2**torch.arange(bits -1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def get_class(kls):
    parts = kls.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def set_seed(seed, conf=None):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.backends.cudnn.deterministic = True
    # Pretraining was done with benchmark set to True, but going forward, we keep it False for reproducibility.
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def pretty_print_matrix(m, name='Mat', num=4):
    if isinstance(m, torch.Tensor):
        matrix = m.detach().cpu().numpy()
    else:
        matrix = m
    print('\u0332'.join(name + ' '))
    print('\n'.join(['\t'.join([str(round(cell.item(),num)) for cell in row]) for row in matrix]))


def zero_mean_unit_var(embeds):
    mean = embeds.mean(dim=-1, keepdim=True)
    std = embeds.std(dim=-1, keepdim=True)

    return (embeds - mean) / std


def get_full_lsh_dump_dir(conf, which, dump_suffix="_dump_aug_embs"):
    task_name = conf.task.name
    if not task_name.endswith(dump_suffix):
        task_name = task_name + dump_suffix

    return Path(conf.lsh.dump_dir) / conf.dataset.name / task_name / which


def get_full_fmap_dump_dir(conf, which):
    return get_full_lsh_dump_dir(conf, which, dump_suffix="_dump_fmap_embs")