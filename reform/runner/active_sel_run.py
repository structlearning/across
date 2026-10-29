# ruff: noqa: E402
import typing

from dotenv import load_dotenv

from reform.lsh import create_bit_indices
from reform.training.active_sel_training import ActiveSelectionTrainer

# Load environment variables before importing torch
load_dotenv()

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from reform.aug_embedder import AugmentedEmbedder
from reform.lsh.lsh import LSH
from reform.models import AugEmbeddingMap, HashModel
from reform.runner.lsh_run import LSHRunner
from reform.utils.utils import get_class


class ActiveSelectionRunner(LSHRunner):
    def __init__(self, overrides: list[str] | None = None):
        super().__init__(overrides=overrides)

        self.graph_model, _ = self.get_saved_graph_models()
        self.graph_model = self.graph_model.to(self.conf.training.device)
        pretrained_embs_base_path = (
            Path(self.conf.base_dir)
            / "allPklDumps"
            / self.conf.dataset.name
            / "pretrained_embs"
            / "train"
        )
        pretrained_embs = torch.load(
            pretrained_embs_base_path / "embeddings.pt", map_location="cpu"
        )
        self.q_embd = typing.cast(Tensor, pretrained_embs["q_embeds"]).to(self.conf.training.device)
        self.c_embd = typing.cast(Tensor, pretrained_embs["c_embeds"]).to(self.conf.training.device)
        pretrained_pre_embs = torch.load(
            pretrained_embs_base_path / "pre_last_layer/embeddings.pt", map_location="cpu"
        )
        self.q_embd_pre = typing.cast(Tensor, pretrained_pre_embs["q_embeds"]).to(self.conf.training.device)
        self.c_embd_pre = typing.cast(Tensor, pretrained_pre_embs["c_embeds"]).to(self.conf.training.device)

        self.aug_embedder = AugmentedEmbedder(self.conf)
        self.aug_emb_model = AugEmbeddingMap(
            self.get_aug_map_input_dim(),
            (self.conf.aug_embedder.hidden_dim,),
            self.conf.aug_embedder.output_dim,
            symmetric=self.conf.aug_embedder.symmetric,
            aug_emb_model=self.conf.aug_embedder.model,
            model_every_dim=self.conf.aug_embedder.model_every_dim,
            gen_out_dim=self.conf.gmn.filters_3,
        ).to(self.conf.training.device)
        aug_ckpt = torch.load(
            self.conf.lsh.aug_checkpoint_path,
            map_location=self.conf.training.device,
        )
        self.aug_emb_model.load_state_dict(aug_ckpt["model_state_dict"])
        logger.info(f"Loaded aug model from {self.conf.lsh.aug_checkpoint_path}")

        hash_model = HashModel(
            self.conf.aug_embedder.output_dim,
            self.conf.lsh.output_dim,
            gen_out_dim=self.conf.gmn.filters_3,
            linear_every_dim=self.conf.lsh.linear_every_dim,
        ).to(self.conf.training.device)
        hash_model_ckpt_path = (
            Path(self.conf.base_dir)
            / self.conf.early_stopping.best_val_name
            / self.conf.task.name
            / "best.pt"
        )
        hash_model_ckpt = torch.load(hash_model_ckpt_path)
        hash_model.load_state_dict(hash_model_ckpt["model_state_dict"])
        logger.info(f"Loaded hash model from {hash_model_ckpt_path}")

        self.rng = np.random.default_rng(self.conf.training.seed)
        bit_indices = create_bit_indices(
            self.conf.lsh.output_dim, self.conf.lsh.subset_size, self.conf.gmn.filters_3, self.rng,
            same_idxs_across_dims=self.conf.lsh.linear_every_dim
        )
        self.lsh = LSH.from_model(hash_model.linears.cpu(), bit_indices,
            self.x_hats["train"].permute(1, 0, 2).contiguous().cpu())

        conf = self.conf.active_selection

        self.conf.task.name += (
            f"/active_num_qs{conf.num_query_samples}_"
            f'{"random_" if conf.random_selection else ""}'
            f"K_{conf.K}"
            f"_oracle_{conf.oracle}"
            f"_patience_{conf.patience}"
            f"_n_iters_{conf.num_iters}"
            f"_freeze_{conf.freeze_layers_before_last_layer}"
        )

        # self.conf.task.name = ShortEncoder().encode(self.conf.task.name)
        Path(self.conf.task.name).parent.mkdir(exist_ok=True, parents=True)

    def load_config(self, overrides: list[str] | None = None):
        super().load_config(overrides=overrides)
        self.conf.dataset.data_type = 'gmn'  # GNN training needs graph objects, not just indices

    def set_trainer(self, datasets=None):
        self.model = get_class(
            f"{self.conf.model.classPath}.{self.conf.model.name}"
        )(self.conf, self.gmn_config).to(self.conf.training.device)
        model_ckpt = torch.load(self.conf.aug_embedder.best_checkpoint_path, map_location=self.conf.training.device)
        self.model.load_state_dict(model_ckpt["model_state_dict"])
        logger.info(f"Loaded {self.conf.model.name} model from {self.conf.aug_embedder.best_checkpoint_path}")

        if self.conf.active_selection.freeze_layers_before_last_layer:
            if self.conf.model.name == "GEN":
                self.model.freeze_layers_before_last_layer = True
                self.model.requires_grad_(False)
                self.model.aggregator.MLP2.requires_grad_(True)
                self.optimizer = torch.optim.Adam(
                    self.model.aggregator.MLP2.parameters(),
                    lr=self.conf.training.learning_rate,
                    weight_decay=self.conf.training.weight_decay,
                )
            else:
                raise NotImplementedError()
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.conf.training.learning_rate,
                weight_decay=self.conf.training.weight_decay,
            )

        self.trainer = ActiveSelectionTrainer(
            self.conf,
            self.model,
            self.optimizer,
            self.lsh,
            self.graph_model,
            self.q_embd,
            self.c_embd,
            self.q_embd_pre,
            self.c_embd_pre,
            self.aug_embedder,
            self.aug_emb_model,
            self.rng,
        )


def main():
    runner = ActiveSelectionRunner()
    runner.set_trainer()
    runner.run(runner.trainer)


if __name__ == "__main__":
    main()
