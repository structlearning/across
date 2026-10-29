import hashlib
import os

import torch
from loguru import logger
from omegaconf import OmegaConf

from reform.utils.utils import set_seed

import wandb


class RunnerBase:
    def __init__(self, overrides: list[str] | None = None):
        self.load_config(overrides=overrides)
        set_seed(self.conf.training.seed)

        self.set_task_name()
        RunnerBase._finalize_task_name(self.conf)
        self.init_wandb()

    @staticmethod
    def _finalize_task_name(conf):
        """Store full task name and shorten any path component of conf.task.name that exceeds the 255-byte filesystem limit."""
        conf.task.full_name = conf.task.name
        parts = conf.task.name.split('/')
        changed = False
        for i, part in enumerate(parts):
            if len(part.encode()) > 251:  # 251 = 255 - len(".log"), the longest file extension used
                name_hash = hashlib.md5(conf.task.name.encode()).hexdigest()[:8]
                parts[i] = f"{part[:240]}_{name_hash}"
                changed = True
        if changed:
            conf.task.name = '/'.join(parts)

    def set_task_name(self):
        conf = self.conf
        name_removal_set = {'classPath', 'name'}
        model_params = ",".join(f"{k}={v}" for k, v in conf.model.items() if k not in name_removal_set)

        if conf.dataset.rel_mode == "sub_iso":
            conf.task.name = (
                f"{conf.model.name}_{conf.dataset.name}"
                f"_numC={conf.dataset.aug_num_cgraphs}"
                f"_MinR={conf.dataset.MinR}_MaxR={conf.dataset.MaxR}"
                f"_{model_params}"
                f",stemp={conf.training.sinkhorn_temp},margin={conf.training.margin}"
            )

            if conf.training.seed != 0:
                conf.task.name += f"_seed={conf.training.seed}"

            if conf.model.name == "GEN" and conf.model.scoring_function != "cos":
                conf.task.name += f"_score={conf.model.scoring_function}"

            if conf.early_stopping.tolerance != 5e-3:
                conf.task.name += f"_estol={conf.early_stopping.tolerance}"

            if conf.early_stopping.metric != 'map_score':
                conf.task.name += f"_esmetric={conf.early_stopping.metric}"
        else:
            raise NotImplementedError(f"rel_mode '{conf.dataset.rel_mode}' is not supported")

    def load_config(self, overrides: list[str] | None = None):
        main_conf = OmegaConf.load("configs/config.yaml")
        cli_conf = (
            OmegaConf.from_dotlist(overrides) if overrides is not None else OmegaConf.from_cli()
        )
        data_conf = OmegaConf.load(f"configs/data_configs/{cli_conf.dataset.rel_mode}/{cli_conf.dataset.name}.yaml")
        model_conf = OmegaConf.load(f"configs/model_configs/{cli_conf.model.name}.yaml")
        conf = OmegaConf.merge(main_conf, data_conf, model_conf, cli_conf)

        self.conf = conf

    def init_wandb(self):
        conf = self.conf

        if conf.task.wandb_enabled:
            name_removal_set = {'classPath', 'name'}
            model_params = ",".join(f"{k}={v}" for k, v in conf.model.items() if k not in name_removal_set)

            task_name_clean = conf.task.name.replace(f"_numC={conf.dataset.aug_num_cgraphs}"
                    f"_MinR={conf.dataset.MinR}_MaxR={conf.dataset.MaxR}"
                    f"_{model_params}"
                    f",stemp={conf.training.sinkhorn_temp},margin={conf.training.margin}", "")

            self.wandb_run = wandb.init(
                project=conf.task.wandb_project,
                name=task_name_clean,
                notes=conf.task.full_name,
                group=conf.task.wandb_group,
                mode=conf.task.wandb_mode,
                config={
                    'learning_rate': conf.training.learning_rate,
                    'weight_decay': conf.training.weight_decay,
                    'num_epochs': conf.training.num_epochs,
                    'seed': conf.training.seed,
                    'batch_size': conf.training.batch_size,
                    'dataset': conf.dataset.name,
                    'gpu': torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU",
                }
            )
            logger.info(f"WandB run initialized with directory: {self.wandb_run.dir}, mode: {conf.task.wandb_mode}")
            logger.info("Use the directory to `wandb sync` afterwards, if mode is offline.")

        # TODO
        # if conf.early_stopping.shorten:
        #     conf.task.name = ShortEncoder().encode(conf.task.name)

    def set_trainer(self):
        raise NotImplementedError("set_trainer() must be implemented in the subclass")

    def run(self, trainer):
        logger.info(f"Task name: {self.conf.task.name}")
        if self.conf.task.name != self.conf.task.full_name:
            logger.info(f"Full task name: {self.conf.task.full_name}")
        log_path = f"{self.conf.log.dir}/{self.conf.task.name}.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger.add(log_path)
        logger.info(OmegaConf.to_yaml(self.conf))

        if self.conf.training.scalable or self.conf.early_stopping.validate_every_n_steps is not None:
            trainer.train_scalable()
        else:
            trainer.train()
