# ruff: noqa: E402
from dotenv import load_dotenv

# Load environment variables before importing torch
load_dotenv()

import argparse
import os
import sys
from pathlib import Path
import json
from tqdm import tqdm

import torch
from loguru import logger

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from reform.utils.model_utils import gen_map_from_embeds
from reform.models import AugEmbeddingMap
from reform.runner.aug_emb_run import AugEmbRunner
from reform.training.aug_emb_train import AugEmbTrainer
from reform.utils.utils import get_full_lsh_dump_dir
from reform.utils.script_utils import (
    CORPUS_SIZE, add_aug_args, _build_ckpt_tag, _build_aug_ckpt_tag,
)


class DumpAugEmbeddingsRunner(AugEmbRunner):
    def set_task_name(self):
        super().set_task_name()
        self.conf.task.name += "_dump_aug_embs"

    def set_trainer(self, datasets=None):
        graph_model, graph_model_best = self.get_saved_graph_models()
        input_dim = self.get_aug_map_input_dim()

        # load trained aug map
        aug_model = AugEmbeddingMap(
            input_dim,
            (self.conf.aug_embedder.hidden_dim,),
            self.conf.aug_embedder.output_dim,
            symmetric=self.conf.aug_embedder.symmetric,
            aug_emb_model=self.conf.aug_embedder.model,
            model_every_dim=self.conf.aug_embedder.model_every_dim,
            gen_out_dim=self.conf.gmn.filters_3,
        ).to(self.conf.training.device)

        ckpt = torch.load(
            self.conf.lsh.aug_checkpoint_path,
            map_location=self.conf.training.device,
        )
        aug_model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded aug model from {self.conf.lsh.aug_checkpoint_path}")

        # dummy optimizer
        optimizer = torch.optim.Adam(
            aug_model.parameters(),
            lr=self.conf.training.learning_rate,
            weight_decay=self.conf.training.weight_decay,
        )

        self.trainer = AugEmbTrainer(
            self.conf,
            aug_model,
            optimizer,
            self.gmn_config,
            graph_model,
            graph_model_best,
            datasets=datasets,
        )

    @torch.no_grad()
    def dump_split(self, which: str):
        out_dir = get_full_lsh_dump_dir(self.conf, which)

        # Check if the dump files already exist. If so, skip.
        if not self.conf.aug_embedder.dont_skip_dump and (out_dir / "q_hat.pt").exists() and (out_dir / "c_hat.pt").exists():
            logger.info(f"Dump files for split '{which}' already exist at {out_dir}. Skipping dump.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Conf task name: {self.conf.task.name}")

        self.trainer.model.eval()
        self.trainer.graph_model.eval()
        self.trainer.graph_model_best.eval()

        q_hat, c_hat, all_gt = self.trainer._validate_common(which=which, preserve_dim=True)

        Q = q_hat.shape[0]
        C = c_hat.shape[0]
        all_gt = all_gt.reshape(Q, C)

        logger.info(
            f"Dumping split={which}: q_hat={tuple(q_hat.shape)}, "
            f"c_hat={tuple(c_hat.shape)}, all_gt={tuple(all_gt.shape)} "
            f"to {out_dir}"
        )

        # B x D x output_dim
        torch.save(q_hat.cpu(), out_dir / "q_hat.pt")
        torch.save(c_hat.cpu(), out_dir / "c_hat.pt")

        metadata = {
            "split": which,
            "task_name": self.conf.task.name,
            "q_shape": list(q_hat.shape),
            "c_shape": list(c_hat.shape),
            "gt_shape": list(all_gt.shape),
            "aug_checkpoint_path": str(self.conf.lsh.aug_checkpoint_path),
            "graph_best_checkpoint_path": str(self.conf.aug_embedder.best_checkpoint_path),
            "graph_latest_ckpts_path": str(self.conf.aug_embedder.latest_ckpts_path),
            "go_back_epochs": int(self.conf.aug_embedder.go_back_epochs),
            "aug_output_dim": int(self.conf.aug_embedder.output_dim),
        }

        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Finished dumping {which} split to {out_dir}")

        # Load and check the MAP from the dumped files
        q_hat_loaded = torch.load(out_dir / "q_hat.pt")
        c_hat_loaded = torch.load(out_dir / "c_hat.pt")
        map_value, aps = gen_map_from_embeds(self.conf, q_hat_loaded.sum(dim=1), c_hat_loaded.sum(dim=1), all_gt.reshape(-1))

        logger.info(f"MAP computed from dumped embeddings for split '{which}': {map_value:.4f}")

    def run_dump(self):
        splits = self.conf.lsh.dump_splits
        if isinstance(splits, str):
            splits = [splits]

        for which in tqdm(splits, desc="Computing embeddings for given split"):
            self.dump_split(which)


def main():
    parser = argparse.ArgumentParser(
        description="Dump augmented embeddings (q_hat/c_hat) for all splits."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. ptc_fr)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value (default: inherit from environment)")
    parser.add_argument("--dump-splits", type=str, nargs="+",
                        default=["train", "val", "test"],
                        help="Splits to dump (default: train val test)")
    add_aug_args(parser)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ckpt_tag     = _build_ckpt_tag(args)
    aug_ckpt_tag = _build_aug_ckpt_tag(args)

    dump_splits_hydra = "[" + ",".join(args.dump_splits) + "]"

    runner = DumpAugEmbeddingsRunner(overrides=[
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
        # e.g. bestValidationModels/GEN_ptc_fr_numC=100000_MinR=0.05_MaxR=0.25_scoring_function=cos,stemp=0.1,margin=1.0/best.pt
        f"aug_embedder.best_checkpoint_path=bestValidationModels/{ckpt_tag}/best.pt",
        # e.g. latestModels/GEN_ptc_fr_numC=100000_MinR=0.05_MaxR=0.25_scoring_function=cos,stemp=0.1,margin=1.0
        f"aug_embedder.latest_ckpts_path=latestModels/{ckpt_tag}",
        # e.g. bestValidationModels/GEN_ptc_fr_numC=100000_..._aug_emb_mlp_sym_h64_l2_o10_lossce-adv-max_afshort_everydim_pushminus_aug_corpus_initfix_pretrained_embs/best.pt
        f"lsh.aug_checkpoint_path=bestValidationModels/{aug_ckpt_tag}/best.pt",
        f"lsh.dump_splits={dump_splits_hydra}",
        "lsh.dump_dir=./allPklDumps",
        "aug_embedder.use_pretrained_embs=True",
        "dataset.data_type=idx",
        "use_cluster=0",
        "task.wandb_enabled=false",
        f"early_stopping.tolerance={args.estol}",
    ])
    runner.set_trainer()
    runner.run_dump()


if __name__ == "__main__":
    main()
