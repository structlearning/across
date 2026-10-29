# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()

import os
import sys
import torch
from pathlib import Path

from reform.runner.gen_run import GENRunner
from reform.utils.dataset_loader import SubgraphIsomorphismDataset
from reform.utils.model_utils import compute_gnn_embeds
from reform.utils.utils import get_class

from loguru import logger


def main():
    _defaults = ["task.wandb_enabled=false", "training.batch_size=8192"]
    runner = GENRunner(overrides=_defaults + sys.argv[1:])
    conf = runner.conf

    logger.info(f"Loaded config: {conf}")
    logger.info("NOTE: This script produces GNN embeddings (full weights) for the trainset of the chosen cluster.")

    parent_path = Path(conf.base_dir) / "bestValidationModels"
    parent_dump_path = Path(conf.base_dir) / "allPklDumps"

    model = get_class(f"{conf.model.classPath}.{conf.model.name}")(conf, runner.gmn_config).to(conf.training.device)

    ckpt_path = parent_path / conf.task.name / "best.pt"
    best_checkpoint = torch.load(ckpt_path, map_location=conf.training.device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_epoch = int(best_checkpoint["epoch"])
    logger.info(f"Loaded best checkpoint from epoch {best_epoch}: {ckpt_path}")

    dataset = SubgraphIsomorphismDataset(conf, mode="train")

    cluster_str = str(conf.use_cluster) if conf.use_cluster is not None else "all"
    print(f"Generating GNN embeddings for {conf.dataset.name}/train (cluster={cluster_str})")

    dump_path = parent_dump_path / conf.dataset.name / "gnn_embs" / f"cluster_{cluster_str}" / "train"
    os.makedirs(dump_path, exist_ok=True)

    q_embeds, c_embeds = compute_gnn_embeds(model, dataset, pre_last_layer=False)

    assert q_embeds.shape[0] == len(dataset.query_graphs)
    assert c_embeds.shape[0] == len(dataset.corpus_graphs)
    assert q_embeds.shape[1] == c_embeds.shape[1] == conf.gmn.filters_3

    torch.save({"q_embeds": q_embeds.cpu(), "c_embeds": c_embeds.cpu()},
               dump_path / "embeddings.pt")
    print(f"Saved GNN embeddings → {dump_path / 'embeddings.pt'}")


if __name__ == "__main__":
    main()
