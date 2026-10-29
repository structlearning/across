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
    logger.info("NOTE: This script produces GNN embeddings (PRE-LAST-LAYER) for the chosen model both using the best checkpoint, and a checkpoint that is _two_ epochs behind the best checkpoint.")
    logger.info("This is meant to correspond to the models parameterized by theta_t and theta_0 from our formulation.")

    parent_path = Path(conf.base_dir) / "bestValidationModels"
    parent_path_latest = Path(conf.base_dir) / "latestModels"
    parent_dump_path = Path(conf.base_dir) / "allPklDumps"

    model_0 = get_class(f"{conf.model.classPath}.{conf.model.name}")(conf, runner.gmn_config).to(conf.training.device)
    model_t = get_class(f"{conf.model.classPath}.{conf.model.name}")(conf, runner.gmn_config).to(conf.training.device)

    ckpt_path = parent_path / conf.task.name / "best.pt"
    best_checkpoint = torch.load(ckpt_path, map_location=conf.training.device)
    model_t.load_state_dict(best_checkpoint['model_state_dict'])
    best_epoch = int(best_checkpoint['epoch'])
    go_back_epochs = int(conf.aug_embedder.go_back_epochs)
    assert best_epoch - go_back_epochs > 0, f"Go back epochs ({go_back_epochs}) must be less than best epoch ({best_epoch})"

    ckpt = torch.load(parent_path_latest / conf.task.name / f"{best_epoch - go_back_epochs}.pt")
    model_0.load_state_dict(ckpt["model_state_dict"])

    assert conf.use_cluster == 0, "Pretrained embeddings dump is only supported for cluster (0) setting currently"
    # For now, do only trainset
    splits_to_run = (
        ("train", SubgraphIsomorphismDataset(conf, mode="train")),
    )

    for split, dataset in splits_to_run:
        print(f"Generating embeddings for {conf.dataset.name}/{split}")
        dump_path = parent_dump_path / conf.dataset.name / "pretrained_embs" / split
        os.makedirs(dump_path, exist_ok=True)

        q_embeds, c_embeds = compute_gnn_embeds(model_0, dataset)
        q_embeds_t, c_embeds_t = compute_gnn_embeds(model_t, dataset)

        assert q_embeds.shape[0] == len(dataset.query_graphs)
        assert c_embeds.shape[0] == len(dataset.corpus_graphs)
        assert q_embeds.shape[1] == c_embeds.shape[1] == conf.gmn.filters_3

        assert q_embeds_t.shape[0] == len(dataset.query_graphs)
        assert c_embeds_t.shape[0] == len(dataset.corpus_graphs)
        assert q_embeds_t.shape[1] == c_embeds_t.shape[1] == conf.gmn.filters_3

        torch.save({"q_embeds": q_embeds.cpu(), "c_embeds": c_embeds.cpu()},
                   dump_path / "embeddings_theta_0.pt")
        torch.save({"q_embeds": q_embeds_t.cpu(), "c_embeds": c_embeds_t.cpu()},
                   dump_path / "embeddings_theta_t.pt")
        print(f"Saved embeddings → {dump_path / 'embeddings_theta_0.pt'}")
        print(f"Saved embeddings → {dump_path / 'embeddings_theta_t.pt'}")


if __name__ == "__main__":
    main()
