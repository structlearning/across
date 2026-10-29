# ruff: noqa: E402
from dotenv import load_dotenv

# Load environment variables before importing torch
load_dotenv()

import torch
from loguru import logger

from reform.runner.base import RunnerBase
from reform.training.gnn import GNNTrainer
from reform.utils.model_utils import get_default_gmn_config, modify_gmn_main_config
from reform.utils.utils import get_class, set_seed


class GENRunner(RunnerBase):
    def load_config(self, overrides=None):
        super().load_config(overrides=overrides)

        self.gmn_config = modify_gmn_main_config(
            get_default_gmn_config(self.conf), self.conf, logger)

    def set_task_name(self):
        super().set_task_name()
        conf = self.conf

        if conf.training.batch_size != 8192:
            conf.task.name += f"_bs={conf.training.batch_size}"

        if conf.training.oracle_dataset is not None:
            conf.task.name += f"_oracle_ds-{conf.training.oracle_dataset}"

        if conf.training.oracle_fresh:
            conf.task.name += "_oracle_fresh"

        if conf.training.relationship_load != 'base' and conf.use_cluster is not None:
            conf.task.name += f"_rels-{conf.use_cluster}-{conf.training.relationship_load}"

        if conf.lsh.subtract_mean:
            conf.task.name += "_lsh_sub_mean"

        if conf.cluster_suffix is not None:
            conf.task.name += f"_cs-{conf.cluster_suffix}"

        if conf.training.loss != 'pairwise-ranking':
            conf.task.name += f"_loss-{conf.training.loss}"

        if not conf.training.stratified_batching:
            conf.task.name += "_nostrat"

        if conf.early_stopping.validate_every_n_steps is not None:
            conf.task.name += f"_valstep={conf.early_stopping.validate_every_n_steps}"

        if conf.val_use_test_set:
            conf.task.name += "_valtest"

        if conf.cluster_rel_subdir is not None:
            conf.task.name += f"/{conf.cluster_rel_subdir}"

    def set_trainer(self):
        self.model = get_class(
            f"{self.conf.model.classPath}.{self.conf.model.name}"
        )(self.conf, self.gmn_config).to(self.conf.training.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training.learning_rate,
            weight_decay=self.conf.training.weight_decay
        )
        self.trainer = GNNTrainer(self.conf, self.model, self.optimizer)


if __name__ == "__main__":
    runner = GENRunner()
    runner.set_trainer()
    runner.run(runner.trainer)