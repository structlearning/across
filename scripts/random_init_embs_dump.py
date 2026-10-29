# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()

import argparse
import os
import sys
import json
from tqdm import tqdm

import torch
from loguru import logger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from reform.utils.model_utils import gen_map_from_embeds
from reform.models import AugEmbeddingMap
from reform.runner.aug_emb_run import AugEmbRunner
from reform.training.aug_emb_train import AugEmbTrainer
from reform.utils.utils import get_full_lsh_dump_dir, get_class
from reform.utils.script_utils import CORPUS_SIZE, add_aug_args

from aug_map_dump import DumpAugEmbeddingsRunner


class DumpRandomInitEmbeddingsRunner(DumpAugEmbeddingsRunner):
    """Dumps embeddings from a fully randomly-initialized GNN + AugNet MLP.

    Neither the GNN nor the AugNet checkpoint is loaded. Both are at
    their default PyTorch initialization state. The full forward pass
    (GNN → AugmentedEmbedder → MLP) is run, producing (N, D, output_dim)
    tensors — structurally identical to the trained case, just with random weights.
    """

    def set_task_name(self):
        super().set_task_name()
        self.conf.task.name = self.conf.task.name.replace(
            "_dump_aug_embs", "_dump_random_init_embs"
        )

    def set_trainer(self):
        # GNN models with random init — no checkpoint loaded
        graph_model = get_class(
            f"{self.conf.model.classPath}.{self.conf.model.name}"
        )(self.conf, self.gmn_config).to(self.conf.training.device)
        graph_model_best = get_class(
            f"{self.conf.model.classPath}.{self.conf.model.name}"
        )(self.conf, self.gmn_config).to(self.conf.training.device)

        input_dim = self.get_aug_map_input_dim()

        # AugNet MLP with random init — no checkpoint loaded
        aug_model = AugEmbeddingMap(
            input_dim,
            (self.conf.aug_embedder.hidden_dim,),
            self.conf.aug_embedder.output_dim,
            symmetric=self.conf.aug_embedder.symmetric,
            aug_emb_model=self.conf.aug_embedder.model,
            model_every_dim=self.conf.aug_embedder.model_every_dim,
            gen_out_dim=self.conf.gmn.filters_3,
        ).to(self.conf.training.device)

        optimizer = torch.optim.Adam(aug_model.parameters(), lr=1e-3)

        self.trainer = AugEmbTrainer(
            self.conf,
            aug_model,
            optimizer,
            self.gmn_config,
            graph_model,
            graph_model_best,
        )

    @torch.no_grad()
    def dump_split(self, which: str):
        out_dir = get_full_lsh_dump_dir(self.conf, which, dump_suffix="_dump_random_init_embs")

        if not self.conf.aug_embedder.dont_skip_dump and \
                (out_dir / "q_hat.pt").exists() and (out_dir / "c_hat.pt").exists():
            logger.info(f"Dump files for split '{which}' already exist at {out_dir}. Skipping.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Conf task name: {self.conf.task.name}")

        self.trainer.model.eval()
        self.trainer.graph_model.eval()
        self.trainer.graph_model_best.eval()

        # Full pipeline (pre_augnet=False): GNN → AugmentedEmbedder → MLP
        q_hat, c_hat, all_gt = self.trainer._validate_common(which=which, preserve_dim=True)

        Q = q_hat.shape[0]
        C = c_hat.shape[0]
        all_gt = all_gt.reshape(Q, C)

        logger.info(
            f"Dumping split={which}: q_hat={tuple(q_hat.shape)}, "
            f"c_hat={tuple(c_hat.shape)}, all_gt={tuple(all_gt.shape)} "
            f"to {out_dir}"
        )

        torch.save(q_hat.cpu(), out_dir / "q_hat.pt")
        torch.save(c_hat.cpu(), out_dir / "c_hat.pt")

        metadata = {
            "split": which,
            "task_name": self.conf.task.name,
            "q_shape": list(q_hat.shape),
            "c_shape": list(c_hat.shape),
            "gt_shape": list(all_gt.shape),
            "random_init": True,
        }

        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Finished dumping {which} split to {out_dir}")

        q_flat = torch.load(out_dir / "q_hat.pt").reshape(Q, -1)
        c_flat = torch.load(out_dir / "c_hat.pt").reshape(C, -1)
        map_value, _ = gen_map_from_embeds(
            self.conf, q_flat, c_flat, all_gt.reshape(-1)
        )
        logger.info(
            f"Random-init MAP from dumped embeddings for split '{which}': {map_value:.4f}"
        )

    def run_dump(self):
        splits = self.conf.lsh.dump_splits
        if isinstance(splits, str):
            splits = [splits]
        for which in tqdm(splits, desc="Computing random-init embeddings for given split"):
            self.dump_split(which)


def main():
    parser = argparse.ArgumentParser(
        description="Dump random-init embeddings (GNN + AugNet both randomly initialized, no checkpoint loaded)."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--dump-splits", type=str, nargs="+",
                        default=["train", "val", "test"])
    add_aug_args(parser)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dump_splits_hydra = "[" + ",".join(args.dump_splits) + "]"

    runner = DumpRandomInitEmbeddingsRunner(overrides=[
        f"dataset.name={args.dataset}",
        "dataset.rel_mode=sub_iso",
        "model.name=GEN",
        "model.scoring_function=cos",
        f"dataset.aug_num_cgraphs={CORPUS_SIZE}",
        "dataset.MinR=0.05",
        "dataset.MaxR=0.25",
        "training.sinkhorn_temp=0.1",
        "training.margin=1.0",
        f"training.loss={args.aug_loss}",
        "training.batch_size=8192",
        "aug_embedder.model=mlp",
        f"aug_embedder.symmetric={args.aug_symmetric}",
        f"aug_embedder.hidden_dim={args.aug_hidden_dim}",
        f"aug_embedder.num_layers={args.aug_num_layers}",
        f"aug_embedder.output_dim={args.aug_output_dim}",
        f"aug_embedder.aug_form={args.aug_form}",
        f"aug_embedder.model_every_dim={args.aug_model_every_dim}",
        f"aug_embedder.push_minus={args.aug_push_minus}",
        f"lsh.dump_splits={dump_splits_hydra}",
        "lsh.dump_dir=./allPklDumps",
        "aug_embedder.use_pretrained_embs=False",
        "aug_embedder.random_init_dumps=True",
        "dataset.data_type=idx",
        "use_cluster=0",
        "task.wandb_enabled=false",
    ])
    runner.set_trainer()
    runner.run_dump()


if __name__ == "__main__":
    main()
