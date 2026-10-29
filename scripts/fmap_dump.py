# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import os
import sys

import torch
from loguru import logger
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from reform.runner.fmap_run import FMapEmbRunner
from reform.utils.model_utils import gen_map_from_embeds
from reform.utils.utils import get_full_fmap_dump_dir
from reform.utils.script_utils import (
    CORPUS_SIZE, _build_ckpt_tag, _build_fmap_ckpt_tag, add_fmap_args,
)


class DumpFMapEmbeddingsRunner(FMapEmbRunner):
    def set_task_name(self):
        super().set_task_name()
        self.conf.task.name += "_dump_fmap_embs"

    @torch.no_grad()
    def dump_split(self, which: str):
        out_dir = get_full_fmap_dump_dir(self.conf, which)

        if not self.conf.fmap_embedder.dont_skip_dump and \
                (out_dir / "q_hat.pt").exists() and (out_dir / "c_hat.pt").exists():
            logger.info(f"Dump files for split '{which}' already exist at {out_dir}. Skipping.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

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

        torch.save(q_hat.cpu(), out_dir / "q_hat.pt")
        torch.save(c_hat.cpu(), out_dir / "c_hat.pt")

        metadata = {
            "split": which,
            "task_name": self.conf.task.name,
            "q_shape": list(q_hat.shape),
            "c_shape": list(c_hat.shape),
            "gt_shape": list(all_gt.shape),
            "fmap_checkpoint_path": str(self.conf.lsh.fmap_checkpoint_path),
            "graph_best_checkpoint_path": str(self.conf.fmap_embedder.best_checkpoint_path),
            "graph_latest_ckpts_path": str(self.conf.fmap_embedder.latest_ckpts_path),
            "go_back_epochs": int(self.conf.fmap_embedder.go_back_epochs),
            "fmap_output_dim": int(self.conf.fmap_embedder.output_dim),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Finished dumping {which} split to {out_dir}")

        # Verify MAP from dumped files
        q_hat_loaded = torch.load(out_dir / "q_hat.pt")
        c_hat_loaded = torch.load(out_dir / "c_hat.pt")
        if not self.conf.fmap_embedder.flat_mlp:
            q_for_map = q_hat_loaded.sum(dim=1)
            c_for_map = c_hat_loaded.sum(dim=1)
        else:
            q_for_map = q_hat_loaded
            c_for_map = c_hat_loaded
        map_value, _ = gen_map_from_embeds(self.conf, q_for_map, c_for_map, all_gt.reshape(-1))
        logger.info(f"MAP from dumped embeddings for split '{which}': {map_value:.4f}")

    def run_dump(self):
        splits = self.conf.lsh.dump_splits
        if isinstance(splits, str):
            splits = [splits]
        for which in tqdm(splits, desc="Dumping FMap embeddings"):
            self.dump_split(which)


def main():
    parser = argparse.ArgumentParser(
        description="Dump FMap embeddings (q_hat/c_hat) for all splits."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. ptc_fr)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value (default: inherit from environment)")
    parser.add_argument("--dump-splits", type=str, nargs="+",
                        default=["train", "val", "test"],
                        help="Splits to dump (default: train val test)")
    add_fmap_args(parser)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ckpt_tag      = _build_ckpt_tag(args)
    fmap_ckpt_tag = _build_fmap_ckpt_tag(args)
    dump_splits_hydra = "[" + ",".join(args.dump_splits) + "]"

    runner = DumpFMapEmbeddingsRunner(overrides=[
        f"dataset.name={args.dataset}",
        "dataset.rel_mode=sub_iso",
        "model.name=GEN",
        "model.scoring_function=cos",
        f"dataset.aug_num_cgraphs={CORPUS_SIZE}",
        "dataset.MinR=0.05",
        "dataset.MaxR=0.25",
        "training.sinkhorn_temp=0.1",
        "training.margin=1.0",
        f"training.loss={args.fmap_loss}",
        "training.batch_size=8192",
        "fmap_embedder.model=mlp",
        f"fmap_embedder.symmetric={args.fmap_symmetric}",
        f"fmap_embedder.hidden_dim={args.fmap_hidden_dim}",
        f"fmap_embedder.num_layers={args.fmap_num_layers}",
        f"fmap_embedder.output_dim={args.fmap_output_dim}",
        f"fmap_embedder.m_use={args.fmap_m_use}",
        f"fmap_embedder.T={args.fmap_T}",
        f"fmap_embedder.flat_mlp={args.fmap_flat_mlp}",
        f"fmap_embedder.best_checkpoint_path=bestValidationModels/{ckpt_tag}/best.pt",
        f"fmap_embedder.latest_ckpts_path=latestModels/{ckpt_tag}",
        f"lsh.fmap_checkpoint_path=bestValidationModels/{fmap_ckpt_tag}/best.pt",
        f"lsh.dump_splits={dump_splits_hydra}",
        "lsh.dump_dir=./allPklDumps",
        "task.wandb_enabled=false",
        f"early_stopping.tolerance={args.estol}",
    ])
    runner.set_trainer()
    runner.run_dump()


if __name__ == "__main__":
    main()
