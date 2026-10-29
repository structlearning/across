# ruff: noqa: E402
from dotenv import load_dotenv

# Load environment variables before importing torch
load_dotenv()

import sys
from pathlib import Path

import torch
from loguru import logger

from reform.models import HashModel
from reform.runner.aug_emb_run import AugEmbRunner
from reform.runner.base import RunnerBase
from reform.runner.fmap_run import FMapRunner
from reform.training.lsh_train import LSHTrainer
from reform.utils.utils import get_full_lsh_dump_dir, get_full_fmap_dump_dir, set_seed


def _build_conf_only(runner_cls, overrides: list[str]):
    """Instantiate runner_cls config + task name without loading embeddings or wandb.

    Useful for resolving checkpoint paths in sweep scripts without paying
    the cost of loading train/val/test embeddings.
    """
    inst = object.__new__(runner_cls)
    inst.load_config(overrides)
    if issubclass(runner_cls, FMapRunner):
        FMapRunner.set_task_name(inst)
    else:
        AugEmbRunner.set_task_name(inst)
    runner_cls.set_task_name(inst)
    RunnerBase._finalize_task_name(inst.conf)
    return inst.conf


class LSHRunner(AugEmbRunner):
    def __init__(self, overrides: list[str] | None = None):
        # XXX: Order of operations kept same as base runner.
        self.load_config(overrides=overrides)
        set_seed(self.conf.training.seed)
        # XXX: ugly
        super().set_task_name()

        # Load query and corpus embeddings for all splits from dump_dir
        self.q_hats = {}
        self.x_hats = {}

        _dump_sfx = (
            "_dump_random_init_embs" if self.conf.aug_embedder.random_init_dumps
            else "_dump_pre_augnet_embs" if self.conf.aug_embedder.pre_augnet_dumps
            else "_dump_aug_embs"
        )
        for which in ["train", "val", "test"]:
            self.q_hats[which] = torch.load(
                get_full_lsh_dump_dir(self.conf, which, dump_suffix=_dump_sfx) / "q_hat.pt",
                map_location=self.conf.training.device)
            self.x_hats[which] = torch.load(
                get_full_lsh_dump_dir(self.conf, which, dump_suffix=_dump_sfx) / "c_hat.pt",
                map_location=self.conf.training.device)

        self.set_task_name()
        RunnerBase._finalize_task_name(self.conf)
        self.init_wandb()

    def load_config(self, overrides: list[str] | None = None):
        super().load_config(overrides=overrides)

        self.conf.dataset.data_type = 'idx' # load pairs of query, corpus index lists

    def set_task_name(self):
        conf = self.conf

        if conf.dataset.rel_mode == "sub_iso":
            if conf.aug_embedder.random_init_dumps:
                conf.task.name += "_random_init"  # before _lsh_ so _aug_readable_name captures it
            elif conf.aug_embedder.pre_augnet_replicas:
                conf.task.name += "_pre_augnet_replicas"  # before _lsh_ so _aug_readable_name captures it
            elif conf.aug_embedder.pre_augnet_dumps:
                conf.task.name += "_pre_augnet"  # before _lsh_ so _aug_readable_name captures it
            conf.task.name += (
                f"_lsh_{conf.lsh.model}_outdim{conf.lsh.output_dim}"
                f"_FS{conf.lsh.lambda_fence_sit}_CM{conf.lsh.lambda_coll_min}_BB{conf.lsh.lambda_bit_bal}"
                f"_collMargin{conf.lsh.coll_min_margin}"
                f"_tTemp{conf.lsh.tanh_temp}"
            )
            if conf.lsh.linear_every_dim:
                conf.task.name += "_linperd"
            if conf.lsh.third_loss_type == 'ad_decorr':
                conf.task.name += "_ad_decorr"
        else:
            msg = f"rel_mode '{conf.dataset.rel_mode}' is not supported"
            raise NotImplementedError(msg)

    def set_trainer(self, datasets=None):
        self.model = HashModel(self.conf.aug_embedder.output_dim, self.conf.lsh.output_dim,
            gen_out_dim=self.conf.gmn.filters_3, linear_every_dim=self.conf.lsh.linear_every_dim
        ).to(self.conf.training.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training.learning_rate,
            weight_decay=self.conf.training.weight_decay,
        )

        self.trainer = LSHTrainer(
            self.conf, self.model, self.optimizer, self.q_hats, self.x_hats
        )


class FMapLSHRunner(FMapRunner):
    def __init__(self, overrides: list[str] | None = None):
        self.load_config(overrides=overrides)
        set_seed(self.conf.training.seed)
        super().set_task_name()  # FMapRunner.set_task_name → sets full FMap task name

        self.q_hats = {}
        self.x_hats = {}

        for which in ["train", "val", "test"]:
            self.q_hats[which] = torch.load(get_full_fmap_dump_dir(self.conf, which) / "q_hat.pt",
                map_location=self.conf.training.device)
            self.x_hats[which] = torch.load(get_full_fmap_dump_dir(self.conf, which) / "c_hat.pt",
                map_location=self.conf.training.device)

        self.set_task_name()
        RunnerBase._finalize_task_name(self.conf)
        self.init_wandb()

    def load_config(self, overrides: list[str] | None = None):
        super().load_config(overrides=overrides)
        self.conf.dataset.data_type = 'idx'

    def set_task_name(self):
        conf = self.conf

        if conf.dataset.rel_mode == "sub_iso":
            conf.task.name += (
                f"_lsh_{conf.lsh.model}_outdim{conf.lsh.output_dim}"
                f"_FS{conf.lsh.lambda_fence_sit}_CM{conf.lsh.lambda_coll_min}_BB{conf.lsh.lambda_bit_bal}"
                f"_collMargin{conf.lsh.coll_min_margin}"
                f"_tTemp{conf.lsh.tanh_temp}"
            )
            if conf.lsh.linear_every_dim:
                conf.task.name += "_linperd"
            if conf.lsh.third_loss_type == 'ad_decorr':
                conf.task.name += "_ad_decorr"
        else:
            msg = f"rel_mode '{conf.dataset.rel_mode}' is not supported"
            raise NotImplementedError(msg)

    def set_trainer(self, datasets=None):
        self.model = HashModel(self.conf.fmap_embedder.output_dim, self.conf.lsh.output_dim,
            gen_out_dim=self.conf.gmn.filters_3, linear_every_dim=self.conf.lsh.linear_every_dim
        ).to(self.conf.training.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training.learning_rate,
            weight_decay=self.conf.training.weight_decay,
        )

        self.trainer = LSHTrainer(
            self.conf, self.model, self.optimizer, self.q_hats, self.x_hats
        )


if __name__ == "__main__":
    overrides = sys.argv[1:]
    runner = LSHRunner(overrides=overrides)
    runner.set_trainer()
    runner.run(runner.trainer)
