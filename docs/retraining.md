# Retraining from scratch

There are six stages. All commands assume the active env is the one
pinned in `requirements.txt` and that `.env` is present.

Expect multiple days on a multi-GPU cluster (AugNet training and GNN training).

## Batching

Every stage emits many commands with the same shape. Dump them to a file and
run with `scripts/cmd_queue.py`, which handles GPU assignment, resume via a
JSON state file, and max-concurrency:

```bash
python scripts/cmd_queue.py <commands_file> --max-jobs 4 --min-free-mb 8000
```

Re-running the same file skips commands marked `done` in the state file.

## Stage 0 — Base NSR on Qbase

Trains the shared GNN backbone on the base query distribution using cosine
scoring and a pairwise ranking loss. The resulting checkpoint anchors the
Taylor linearization in stage 1.

```bash
python -B -m reform.runner.gen_run \
    dataset.name=ptc_fr model.name=GEN dataset.rel_mode=sub_iso \
    dataset.aug_num_cgraphs=100000 dataset.MinR=0.05 dataset.MaxR=0.25 \
    training.sinkhorn_temp=0.1 training.margin=1.0 \
    model.scoring_function=cos
```

Repeat for `ptc_fm`, `ptc_mr`, `cox2`. Output lands in
`bestValidationModels/GEN_<dataset>_numC=100000_...scoring_function=cos.../best.pt`.

### 0b. Pretrained GNN embeds dump

Dumps the base-NSR GNN embeddings (pre-last-layer) at both the best epoch
(`θ_t`) and `go_back_epochs=2` epochs before (`θ_0`). AugNet training in
stage 1 loads these via `aug_embedder.use_pretrained_embs=True`.

```bash
python -B scripts/pretrained_embs_dump.py \
    dataset.name=ptc_fr model.name=GEN dataset.rel_mode=sub_iso \
    use_cluster=0 training.batch_size=8192
```

Repeat per dataset. Output lands under `allPklDumps/<dataset>/pretrained_embs/`.

## Stage 1 — AugNets

Trains `fϕ` (the perturbation-tolerant head). One run per
(dataset × scheme × form), where scheme ∈ {`ce-adv-max` adversarial,
`ce-avg` sampling-based} and form ∈ {`short`, `long`}. The paper reports
the adversarial scheme as the main configuration.

```bash
python -B -m reform.runner.aug_emb_run \
    model.name=GEN dataset.name=ptc_fr dataset.rel_mode=sub_iso \
    dataset.aug_num_cgraphs=100000 dataset.MinR=0.05 dataset.MaxR=0.25 \
    dataset.data_type=idx \
    training.sinkhorn_temp=0.1 training.margin=1.0 \
    training.loss=ce-adv-max training.batch_size=1024 \
    early_stopping.tolerance=1e-5 \
    aug_embedder.model=mlp aug_embedder.symmetric=True \
    aug_embedder.hidden_dim=64 aug_embedder.num_layers=2 \
    aug_embedder.output_dim=10 \
    aug_embedder.aug_form=short aug_embedder.push_minus=True \
    aug_embedder.model_every_dim=False \
    aug_embedder.aug_corpus_model=init \
    aug_embedder.use_pretrained_embs=True \
    aug_embedder.best_checkpoint_path=<stage-0 best.pt path>
```

`early_stopping.tolerance=1e-5` is **required** — it is part of the Hydra
task name and all downstream stages look up the checkpoint by that name.
Omitting it will silently produce or consume an untagged dump directory and
numbers will not match.

## Stage 2 — Corpus embedding dumps

Two separate dumps, each feeding a different downstream pipeline.

### 2a. Pre-AugNet dump

Dumps the lifted + randomly-projected corpus representations with no AugNet
MLP applied. Feeds the "ACROSS (aug)" ablation and the no-AugNet baselines.

```bash
python -B scripts/pre_augnet_embs_dump.py --dataset ptc_fr --aug-form short --aug-push-minus
```

8 commands: `{short, long} × {ptc_fr, ptc_fm, ptc_mr, cox2}`. Only
`--aug-form` and `--aug-push-minus` affect the output.

### 2b. AugNet-trained dump

Dumps the final corpus representations `xϕ,d(Gc)` with the trained AugNet
`fϕ` applied. Feeds the main ACROSS pipeline. Must be run per
(dataset × scheme × form), matching the AugNets trained in stage 1.

```bash
python -B scripts/aug_map_dump.py \
    model.name=GEN dataset.name=ptc_fr dataset.rel_mode=sub_iso \
    dataset.aug_num_cgraphs=100000 dataset.MinR=0.05 dataset.MaxR=0.25 \
    dataset.data_type=idx \
    training.sinkhorn_temp=0.1 training.margin=1.0 \
    training.loss=ce-adv-max training.batch_size=1024 \
    model.scoring_function=cos \
    aug_embedder.model=mlp aug_embedder.symmetric=True \
    aug_embedder.hidden_dim=64 aug_embedder.num_layers=2 \
    aug_embedder.output_dim=10 \
    aug_embedder.aug_form=short aug_embedder.push_minus=True \
    aug_embedder.model_every_dim=False \
    aug_embedder.aug_corpus_model=init \
    aug_embedder.use_pretrained_embs=True \
    aug_embedder.best_checkpoint_path=<stage-0 best.pt path> \
    lsh.aug_checkpoint_path=<stage-1 AugNet best.pt path> \
    use_cluster=0
```

16 commands: `{ce-adv-max, ce-avg} × {short, long} × {4 datasets}`.

## Stage 3 — LSH retrieval sweep

Picks a random-hyperplane configuration whose set-recall divided by (k/C) is maximal
with `k/C ∈ [0.2, 0.6]`. Writes the chosen hyperplanes + subset size to
`allPklDumps/.../hyperplanes_<...>.pkl`.

```bash
python -B scripts/lsh_retr_sweep.py \
    --dataset ptc_fr --aug-loss ce-adv-max \
    --aug-form short --aug-symmetric --no-aug-model-every-dim \
    --aug-push-minus --estol 1e-5
```

Known skips (no valid operating point exists):
- `ptc_fm` long form on pre-augnet embeddings
- `cox2` short form on pre-augnet-replica embeddings

These cells are intentionally empty in downstream tables.

## Stage 4 — Dataset generation

Two scripts run back-to-back per cluster.

### 4a. gen_dataset_carefully

Runs the active-selection procedure (ACROSS retrieval + solver labeling) on
each OOD query cluster, producing per-query positive/negative label sets.

```bash
python -B scripts/active_sel/gen_dataset_carefully.py \
    --dataset ptc_fr --cluster 1 \
    --aug-loss ce-adv-max --aug-form short --aug-symmetric \
    --no-aug-model-every-dim --aug-push-minus \
    --estol 1e-5 --mode lsh --ratio 0.13 --K_val 40
```

`--K_val 40` is the solver budget `B` used in the paper. `--ratio 0.13`
fixes the positive-to-negative ratio in the generated training set.

### 4b. noisy_rerank_plot

Takes the per-query candidate pool from 4a, injects reranking noise, writes
the final LSH-relationship pkl that stage 5 trains against.

```bash
python -B scripts/active_sel/noisy_rerank_plot.py \
    --dataset ptc_fr --cluster 1 \
    --aug-loss ce-adv-max --aug-form short --aug-symmetric \
    --no-aug-model-every-dim --aug-push-minus \
    --estol 1e-5 --mode lsh --ratio 0.13 --K_val 40 \
    --sigmas 0.01 0.1 0.5 1 2 5 10 100 --overwrite
```

`--ratio 0.13` is **required**. Without it the script dumps the entire
corpus as negatives (~100 K per query instead of ~111); the file name
also drops `_ratio=0.13`, which breaks the `cluster_suffix` lookup in
stage 5.

## Stage 5 — Downstream GNN training

One run per (dataset × cluster × rel × suffix × augnet) tuple. 272 tasks
cover Tables 1-9 in the paper. `scripts/reproduce/tasks.jsonl` lists the
exact grid.

```bash
python -B -m reform.runner.gen_run \
    dataset.name=ptc_fr model.name=GEN dataset.rel_mode=sub_iso \
    training.batch_size=128 \
    use_cluster=1 training.relationship_load=lsh \
    cluster_rel_subdir=AugMap_CEMax_Symm_SameModelPerD_Minus_Short_SameLinearPerD \
    cluster_suffix=all_corpus_negs_K=40_ratio=0.13_noise_sigma=1 \
    model.scoring_function="hinge" \
    training.loss=celoss training.stratified_batching=False \
    early_stopping.patience=200 early_stopping.metric=ap_score
```

Baselines (random, kcenter, kmeans, infomax, wl_kernel, gnn_dist) use the
same shape with the appropriate `training.relationship_load=<baseline>`
and `cluster_suffix` values. See `scripts/reproduce/tasks.jsonl` for the
full override set per baseline.

**Do not** add `early_stopping.tolerance=1e-5` here. GNN training uses the
config default (`5e-3`); the `1e-5` invariant is specific to the AugNet
pipeline (stages 1-4).

Checkpoints land in
`latestModels/<task>/{<augnet>/}<epoch>.pt` and
`bestValidationModels/<task>/{<augnet>/}best.pt`.

## Stage 6 — Evaluation

Same entry point as the fast path:

```bash
python -B scripts/reproduce/eval_main_tables.py
```

Reads `logs/<task>.log`, replays EarlyStopping at `delta=5e-3` to pick
the best epoch ≤ 200, loads the corresponding `latestModels/<task>/<epoch>.pt`,
runs test eval, writes `testMAPs/<task>.txt`, then renders Tables 1, 2, 3,
4, 6-9, 18-21, 22-25, 26-29.
