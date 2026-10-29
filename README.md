# ACROSS — Active Corpus Selection for Training Subgraph Retrievers Using OOD Queries

Reference implementation for the NeurIPS 2026 paper
*Active Corpus Selection for Training Subgraph Retrievers Using OOD Queries*
([paper link TBD]).

## Quickstart

```bash
make install    # pinned env (torch 2.8.0 + cu128, pyg)
make download   # data + checkpoints from HF Hub
make eval       # reproduces Tables 1, 2, 3, 4, 6–9, 18–21, 22–25, 26–29
```

`make eval` writes `testMAPs/<task>.txt` per task, then renders the LaTeX
tables under `scripts/latex/`.

Any single CUDA GPU with ≥10 GB free is enough.

## Full retraining

The train-from-scratch pipeline (AugNet → embedding dump → LSH sweep →
data gen → GNN training → eval) is documented in
[`docs/retraining.md`](docs/retraining.md). Expect multiple days on a
multi-GPU cluster.

## Repo layout

```
reform/           Core library.
configs/          Hydra configs.
scripts/
  eval_at_epoch.py   Per-task eval entry point.
  reproduce/         Fast-path wrappers.
  latex/             Table renderers.
  active_sel/        Data-generation scripts for the retrain path.
```

## Datasets

PTC-FR, PTC-FM, PTC-MR, and COX2 from
[TUDataset](https://chrsmrrs.github.io/datasets/). Preprocessed splits
and checkpoints are shipped via the HF Hub bundle fetched by `make download`.

## Citation

```bibtex
@inproceedings{across2026,
  title     = {Active Corpus Selection for Training Subgraph Retrievers Using OOD Queries},
  author    = {Chakraborty, Pritish and Singh, Aditya and Roy, Indradyumna and
               Nagalapatti, Lokesh and Dutta, Himanshu and Iyer, Arun and
               Prabhu, Yashoteja and Sinha, Gaurav and Varma, Manik and
               Chakrabarti, Soumen and De, Abir},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```

## License

Code: see `LICENSE`. Datasets inherit their original TUDataset licenses.
