# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

import torch
from loguru import logger

from reform.models import AugEmbeddingMap
from reform.runner.base import RunnerBase
from reform.training.fmap_train import FMapTrainer
from reform.utils.model_utils import get_default_gmn_config, modify_gmn_main_config
from reform.utils.utils import get_class


class FMapRunner(RunnerBase):
    def load_config(self, overrides: list[str] | None = None):
        super().load_config(overrides=overrides)

        self.gmn_config = modify_gmn_main_config(
            get_default_gmn_config(self.conf), self.conf, logger
        )

    def set_task_name(self):
        super().set_task_name()
        conf = self.conf

        if conf.dataset.rel_mode == "sub_iso":
            conf.task.name += (
                f"_fmap_{conf.fmap_embedder.model}"
                f"_{'sym' if conf.fmap_embedder.symmetric else 'asym'}"
                f"_h{conf.fmap_embedder.hidden_dim}"
                f"_l{conf.fmap_embedder.num_layers}"
                f"_o{conf.fmap_embedder.output_dim}"
                f"_m{conf.fmap_embedder.m_use}"
                f"_T{conf.fmap_embedder.T}"
                f"_loss{conf.training.loss}"
                + ("_flat" if conf.fmap_embedder.flat_mlp else "")
            )
        else:
            msg = f"rel_mode '{conf.dataset.rel_mode}' is not supported"
            raise NotImplementedError(msg)

    def get_saved_graph_models(self):
        graph_model = get_class(
            f"{self.conf.model.classPath}.{self.conf.model.name}"
        )(self.conf, self.gmn_config).to(self.conf.training.device)
        graph_model_best = get_class(
            f"{self.conf.model.classPath}.{self.conf.model.name}"
        )(self.conf, self.gmn_config).to(self.conf.training.device)

        assert self.conf.fmap_embedder.best_checkpoint_path is not None, \
            "fmap_embedder.best_checkpoint_path must be provided"
        assert self.conf.fmap_embedder.latest_ckpts_path is not None, \
            "fmap_embedder.latest_ckpts_path must be provided"

        best_checkpoint = torch.load(
            self.conf.fmap_embedder.best_checkpoint_path,
            map_location=self.conf.training.device
        )
        graph_model_best.load_state_dict(best_checkpoint['model_state_dict'])
        logger.info(f"Loaded best graph embedding model from epoch {best_checkpoint['epoch']}")

        best_epoch = int(best_checkpoint['epoch'])
        go_back_epochs = int(self.conf.fmap_embedder.go_back_epochs)
        assert best_epoch - go_back_epochs > 0, \
            f"go_back_epochs ({go_back_epochs}) must be less than best_epoch ({best_epoch})"

        checkpoint = torch.load(
            Path(self.conf.fmap_embedder.latest_ckpts_path) / f"{best_epoch - go_back_epochs}.pt",
            map_location=self.conf.training.device
        )
        graph_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded 'initial' graph embedding model from epoch {best_epoch - go_back_epochs}")

        return graph_model, graph_model_best

    def set_trainer(self, datasets=None):
        graph_model, graph_model_best = self.get_saved_graph_models()
        m = self.conf.fmap_embedder.m_use
        d = self.conf.gmn.filters_3
        input_dim = 4 * m * d if self.conf.fmap_embedder.flat_mlp else 4 * m

        self.model = AugEmbeddingMap(
            input_dim,
            (self.conf.fmap_embedder.hidden_dim,),
            self.conf.fmap_embedder.output_dim,
            symmetric=self.conf.fmap_embedder.symmetric,
            aug_emb_model=self.conf.fmap_embedder.model,
            gen_out_dim=self.conf.gmn.filters_3,
        ).to(self.conf.training.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training.learning_rate,
            weight_decay=self.conf.training.weight_decay,
        )
        self.trainer = FMapTrainer(
            self.conf, self.model, self.optimizer, self.gmn_config,
            graph_model, graph_model_best, datasets=datasets,
        )


class FMapEmbRunner(FMapRunner):
    """FMapRunner that loads a saved FMap checkpoint for inference (no training)."""

    def set_trainer(self, datasets=None):
        graph_model, graph_model_best = self.get_saved_graph_models()
        m = self.conf.fmap_embedder.m_use
        d = self.conf.gmn.filters_3
        input_dim = 4 * m * d if self.conf.fmap_embedder.flat_mlp else 4 * m

        self.model = AugEmbeddingMap(
            input_dim,
            (self.conf.fmap_embedder.hidden_dim,),
            self.conf.fmap_embedder.output_dim,
            symmetric=self.conf.fmap_embedder.symmetric,
            aug_emb_model=self.conf.fmap_embedder.model,
            gen_out_dim=self.conf.gmn.filters_3,
        ).to(self.conf.training.device)

        ckpt = torch.load(
            self.conf.lsh.fmap_checkpoint_path,
            map_location=self.conf.training.device,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded fmap model from {self.conf.lsh.fmap_checkpoint_path}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training.learning_rate,
            weight_decay=self.conf.training.weight_decay,
        )
        self.trainer = FMapTrainer(
            self.conf, self.model, self.optimizer, self.gmn_config,
            graph_model, graph_model_best, datasets=datasets,
        )


if __name__ == "__main__":
    runner = FMapRunner()
    runner.set_trainer()
    runner.run(runner.trainer)
