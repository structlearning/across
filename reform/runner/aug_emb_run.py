# ruff: noqa: E402
from dotenv import load_dotenv

# Load environment variables before importing torch
load_dotenv()

import torch
from loguru import logger
from pathlib import Path

from reform.models import AugEmbeddingMap
from reform.runner.base import RunnerBase
from reform.training.aug_emb_train import AugEmbTrainer
from reform.utils.model_utils import get_default_gmn_config, modify_gmn_main_config
from reform.utils.utils import get_class


class AugEmbRunner(RunnerBase):
    def load_config(self, overrides: list[str] | None = None):
        super().load_config(overrides=overrides)

        # FIXME: remove below line, possibly redundant
        self.gmn_config = modify_gmn_main_config(
            get_default_gmn_config(self.conf), self.conf, logger
        )

        # Load pretrained GNN embeddings for theta_0
        self.q_embeds_0 = {}
        self.c_embeds_0 = {}

        # XXX: hardcoding the pretrained embs dir
        split = "train"
        path = Path("allPklDumps") / self.conf.dataset.name / "pretrained_embs" / split
        embs = torch.load(path / f"embeddings_theta_0.pt",
            map_location="cpu")
        self.q_embeds_0[split] = embs["q_embeds"]
        self.c_embeds_0[split] = embs["c_embeds"]

        if self.conf.aug_embedder.use_pretrained_embs:
            logger.info(f"Using pretrained embeddings for theta_0 from {path}")
            assert self.conf.dataset.data_type == "idx", "Pretrained embeddings can only be used with idx dataset type"

    def set_task_name(self):
        super().set_task_name()
        conf = self.conf

        if conf.dataset.rel_mode == "sub_iso":
            conf.task.name += (
                f"_aug_emb_{conf.aug_embedder.model}"
                f"_{'sym' if conf.aug_embedder.symmetric else 'asym'}"
                f"_h{conf.aug_embedder.hidden_dim}"
                f"_l{conf.aug_embedder.num_layers}"
                f"_o{conf.aug_embedder.output_dim}"
                f"_loss{conf.training.loss}"
                f"_af{conf.aug_embedder.aug_form}"
            )

            if conf.aug_embedder.model_every_dim:
                conf.task.name += "_everydim"
            if conf.aug_embedder.push_minus:
                conf.task.name += "_pushminus"

            if conf.aug_embedder.aug_corpus_model == "init":
                conf.task.name += "_aug_corpus_initfix"

            if conf.aug_embedder.use_pretrained_embs:
                conf.task.name += "_pretrained_embs"

            if conf.aug_embedder.num_hyperplanes != 4:
                conf.task.name += f"_hyp={conf.aug_embedder.num_hyperplanes}"

            if conf.aug_embedder.pgd_epsilon != 0.1:
                conf.task.name += f"_eps={conf.aug_embedder.pgd_epsilon}"
        else:
            msg = f"rel_mode '{conf.dataset.rel_mode}' is not supported"
            raise NotImplementedError(msg)

    def get_aug_map_input_dim(self):
        if self.conf.aug_embedder.aug_form == "long":
            return 2 * (2 + self.conf.gmn.filters_3) * self.conf.aug_embedder.num_hyperplanes
        elif self.conf.aug_embedder.aug_form == "short":
            return 2 * (1 + self.conf.gmn.filters_3) * self.conf.aug_embedder.num_hyperplanes
        else:
            raise ValueError(f"Unsupported aug_form: {self.conf.aug_embedder.aug_form}")

    def get_saved_graph_models(self):
        graph_model = get_class(
            f"{self.conf.model.classPath}.{self.conf.model.name}"
        )(self.conf, self.gmn_config).to(self.conf.training.device)
        graph_model_best = get_class(
            f"{self.conf.model.classPath}.{self.conf.model.name}"
        )(self.conf, self.gmn_config).to(self.conf.training.device)
        assert self.conf.aug_embedder.best_checkpoint_path is not None, "Best checkpoint path for graph embedding model must be provided"
        assert self.conf.aug_embedder.latest_ckpts_path is not None, "Latest checkpoints directory path for graph embedding model must be provided"

        # Load from checkpoint
        best_checkpoint = torch.load(self.conf.aug_embedder.best_checkpoint_path,
            map_location=self.conf.training.device)
        graph_model_best.load_state_dict(best_checkpoint['model_state_dict'])
        logger.info(f"Loaded best graph embedding model from epoch {best_checkpoint['epoch']}")

        best_epoch = int(best_checkpoint['epoch'])
        go_back_epochs = int(self.conf.aug_embedder.go_back_epochs)
        assert best_epoch - go_back_epochs > 0, f"Go back epochs ({go_back_epochs}) must be less than best epoch ({best_epoch})"

        checkpoint = torch.load(Path(self.conf.aug_embedder.latest_ckpts_path) / f"{best_epoch - go_back_epochs}.pt")
        graph_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded 'initial' graph embedding model from epoch {best_epoch - go_back_epochs}")

        return graph_model, graph_model_best

    def set_trainer(self, datasets=None):
        graph_model, graph_model_best = self.get_saved_graph_models()
        input_dim = self.get_aug_map_input_dim()

        self.model = AugEmbeddingMap(
            input_dim,
            (self.conf.aug_embedder.hidden_dim,),
            self.conf.aug_embedder.output_dim,
            symmetric=self.conf.aug_embedder.symmetric,
            aug_emb_model=self.conf.aug_embedder.model,
            model_every_dim=self.conf.aug_embedder.model_every_dim,
            gen_out_dim=self.conf.gmn.filters_3,
        ).to(self.conf.training.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training.learning_rate,
            weight_decay=self.conf.training.weight_decay,
        )
        self.trainer = AugEmbTrainer(
            self.conf, self.model, self.optimizer, self.gmn_config, graph_model,
            graph_model_best, datasets=datasets, q_embeds_0=self.q_embeds_0, c_embeds_0=self.c_embeds_0
        )


if __name__ == "__main__":
    runner = AugEmbRunner()
    runner.set_trainer()
    runner.run(runner.trainer)
