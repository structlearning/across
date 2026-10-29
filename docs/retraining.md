# Retraining from scratch

There are six stages. All commands assume the active env is the one
pinned in `requirements.txt` and that `.env` is present and sourced:

```bash
cp .env.example .env
set -a; source .env; set +a      # export CUBLAS_WORKSPACE_CONFIG etc. to children
```

Expect multiple days on a multi-GPU cluster (AugNet training and GNN training).

## Batching

Every stage emits many commands with the same shape. Dump them to a file and
run with `scripts/cmd_queue.py`, which handles GPU assignment, resume via a
JSON state file, and max-concurrency:

```bash
python scripts/cmd_queue.py <commands_file> --max-jobs 4 --min-free-mb 8000
```

Re-running the same file skips commands marked `done` in the state file.

## Cluster assignments and splits

Each dataset's queries are partitioned into clusters `0, 1, 2, 3, 4`. Cluster
0 is the in-distribution base (used in stage 0 and the AugNet pipeline);
clusters 1-4 are the OOD partitions trained on in stage 5. The validation
split is shared across all clusters — only `train` and `test` are
partitioned, producing `data/<ds>/preprocessed/splits/{train,test}/cluster_assignments.json`.

These JSON files ship pre-computed in the HF bundle fetched by `make
download`, so the retraining pipeline below does not regenerate them. The
code that produced them originally is not part of this public release.

## The task grid (`scripts/reproduce/tasks.jsonl`)

The 272-row frozen grid that enumerates every `(dataset, cluster, rel,
suffix, augnet)` tuple needed for Tables 1–9, 18–21, 22–25, 26–29. Each row
lists the Hydra overrides, the `task_name`, the expected `testMAPs`,
checkpoint, and LSH-rel pkl paths, and which `paper_tables` the row feeds.
`scripts/reproduce/eval_main_tables.py` iterates it, runs
`scripts/eval_at_epoch.py` with the right overrides per row, then renders
the LaTeX tables. Stage 5 of retraining produces one checkpoint per row.

## Stage 0 — Base NSR on Qbase

Trains the shared GNN backbone on the base query distribution using cosine
scoring and a pairwise ranking loss. The resulting checkpoint anchors the
Taylor linearization in stage 1.

```bash
python -B -m across.runner.gen_run \
    dataset.name=ptc_fr model.name=GEN dataset.rel_mode=sub_iso \
    dataset.aug_num_cgraphs=100000 dataset.MinR=0.05 dataset.MaxR=0.25 \
    training.sinkhorn_temp=0.1 training.margin=1.0 \
    model.scoring_function=cos \
    use_cluster=0
```

Repeat for `ptc_fm`, `ptc_mr`, `cox2`. Output lands in
`bestValidationModels/GEN_<dataset>_numC=100000_...scoring_function=cos.../best.pt`.

`use_cluster=0` selects the base-cluster query set (clusters 1-4 are the OOD
partitions used in stage 5). Base NSR training relies on the config defaults
for early stopping (`patience=50`, `tolerance=5e-3`) and typically converges
in ~70 epochs.

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

### 0c. Hyperplane generation

AugNet (stage 1) loads random projection hyperplanes at init from
`allPklDumps/across_hyperplanes_<form>_<K>.pt`. Generate once for each
(form, K) pair used:

```bash
python -B scripts/generate_hyperplanes.py --num-hyperplanes 4
python -B scripts/generate_hyperplanes.py --num-hyperplanes 8 --filters-3 10
```

`--num-hyperplanes` matches `aug_embedder.num_hyperplanes` in stage 1.
`--filters-3` is the GNN last-layer output dim (default 10 matches the
paper's model config). Short form adds one leading dim; long form adds two.

## Stage 1 — AugNets

Trains `fϕ` (the perturbation-tolerant head). One run per
(dataset × scheme × form), where scheme ∈ {`ce-adv-max` adversarial,
`ce-avg` sampling-based} and form ∈ {`short`, `long`}. The paper reports
the adversarial scheme as the main configuration.

```bash
python -B -m across.runner.aug_emb_run \
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
    aug_embedder.num_hyperplanes=4 \
    aug_embedder.use_pretrained_embs=True \
    aug_embedder.best_checkpoint_path=<stage-0 best.pt path> \
    aug_embedder.latest_ckpts_path=<stage-0 latestModels dir> \
    use_cluster=0
```

`use_cluster=0` is required so the AugNet trains against the base-cluster
query set (same cluster as stage 0). `aug_embedder.latest_ckpts_path` is the
stage-0 `latestModels/GEN_<dataset>_...` directory — AugNet needs it to
locate the checkpoint `go_back_epochs=2` steps before the best one.
`aug_embedder.num_hyperplanes` must match the K you generated in stage 0c
(paper uses 4 for the main setting).

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
    aug_embedder.latest_ckpts_path=<stage-0 latestModels dir> \
    lsh.aug_checkpoint_path=<stage-1 AugNet best.pt path> \
    aug_embedder.dont_skip_dump=True \
    use_cluster=0
```

16 commands: `{ce-adv-max, ce-avg} × {short, long} × {4 datasets}`.
`aug_embedder.dont_skip_dump=True` is optional; omit on a first run (default
behavior dumps when files don't yet exist), include it to force regeneration
on a re-run. `aug_embedder.latest_ckpts_path` is the stage-0 `latestModels/`
directory — used to load the `go_back_epochs=2` checkpoint as `θ_0`.

### 2c. Random-init dump

Dumps embeddings produced by running the forward pass with both the GNN
and the AugNet MLP at their random PyTorch init (no checkpoints loaded).
Feeds the `AugMap_..._random_init` variant of the ablation tables.

```bash
python -B scripts/random_init_embs_dump.py \
    model.name=GEN dataset.name=ptc_fr dataset.rel_mode=sub_iso \
    dataset.aug_num_cgraphs=100000 dataset.MinR=0.05 dataset.MaxR=0.25 \
    aug_embedder.aug_form=short aug_embedder.push_minus=True \
    aug_embedder.model_every_dim=False aug_embedder.symmetric=True \
    aug_embedder.hidden_dim=64 aug_embedder.num_layers=2 \
    aug_embedder.output_dim=10 \
    use_cluster=0
```

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

`--ratio ...` is **required**. Without it the script dumps the entire
corpus as negatives (~100 K per query instead of ~111); the file name
also drops `_ratio=0.13`, which breaks the `cluster_suffix` lookup in
stage 5.

## Stage 5 — Downstream GNN training

One run per (dataset × cluster × rel × suffix × augnet) tuple. 272 tasks
cover Tables 1-9 in the paper. `scripts/reproduce/tasks.jsonl` lists the
exact grid.

```bash
python -B -m across.runner.gen_run \
    dataset.name=ptc_fr model.name=GEN dataset.rel_mode=sub_iso \
    training.batch_size=128 \
    use_cluster=1 training.relationship_load=lsh \
    cluster_rel_subdir=AugMap_CEMax_Symm_SameModelPerD_Minus_Short_SameLinearPerD \
    cluster_suffix=all_corpus_negs_K=40_ratio=0.13_noise_sigma=1 \
    model.scoring_function="hinge" \
    training.loss=celoss training.stratified_batching=False \
    training.num_epochs=200 \
    early_stopping.patience=200 early_stopping.metric=ap_score
```

`training.num_epochs=200` caps the run; `early_stopping.patience=200` with
`run_till_early_stopping=True` (config default) typically stops training well
before that cap.

Baselines (`random`, `kcenter`, `kmeans`, `infomax`, `wl_kernel`, `gnn_dist`)
use the same shape with the appropriate `training.relationship_load=<baseline>`
and `cluster_suffix` values (no `cluster_rel_subdir` for these — they don't
nest under an AugNet subdir). See `scripts/reproduce/tasks.jsonl` for the
full override set per baseline. The relationship pkls for each baseline are
produced by the scripts under `scripts/active_sel/baselines/` (see the
Baseline-relationship generation section below).

**Do not** add `early_stopping.tolerance=1e-5` here. GNN training uses the
config default (`5e-3`); the `1e-5` invariant is specific to the AugNet
pipeline (stages 1-4).

Checkpoints land in
`latestModels/<task>/{<augnet>/}<epoch>.pt` and
`bestValidationModels/<task>/{<augnet>/}best.pt`.

## AugNet ablation (Table 4)

Table 4 compares the main method against two "what if we didn't train the
AugNet" variants. Both variants reuse stages 3–5 with a different corpus
embedding feeding the LSH sweep.

**Variant A — No-training replicas** (`AugMap_..._pre_augnet_replicas`).
Starts from stage 2a's pre-AugNet dump instead of stage 2b's trained one,
then reshapes `(D, N, K) → (D·R, N, K/R)` so each hyperplane gets its own
hash table. Re-run stages 3, 4, 5 with `--pre-augnet-replicas` on the
argparse-driven scripts and `cluster_rel_subdir=AugMap_..._pre_augnet_replicas`
at stage 5.

**Variant B — Random init** (`AugMap_..._random_init`). Uses the random-init
dump from stage 2c (both GNN and AugNet MLP at PyTorch random init). Re-run
stages 3, 4, 5 and set `cluster_rel_subdir=AugMap_..._random_init` at stage 5.

Known skip for Variant A (no valid LSH operating point): `cox2` short form.

## Baseline-relationship generation (non-AugNet)

The baselines `random`, `kcenter`, `kmeans`, `infomax`, `wl_kernel`, and
`gnn_dist` each have a dedicated script that writes the training-relationship
pkl stage 5 then loads via `training.relationship_load=<name>`.

```bash
# random, lsh — produced by gen_dataset_carefully.py (stage 4a)
# passing --mode random instead of --mode lsh.

# kcenter, kmeans, infomax, wl_kernel
python -B scripts/active_sel/baselines/gen_kcenter_greedy.py \
    --dataset ptc_fr --cluster 1 --K_val 40 --ratio 0.13
python -B scripts/active_sel/baselines/gen_kmeans.py \
    --dataset ptc_fr --cluster 1 --K_val 40 --ratio 0.13
python -B scripts/active_sel/baselines/gen_infomax.py \
    --dataset ptc_fr --cluster 1 --K_val 40 --ratio 0.13
python -B scripts/active_sel/baselines/gen_wl_kernel.py \
    --dataset ptc_fr --cluster 1 --K_val 40 --ratio 0.13

# gnn_dist — requires GNN embeddings to be dumped first.
python -B scripts/gnn_embs_dump.py \
    dataset.name=ptc_fr model.name=GEN dataset.rel_mode=sub_iso \
    use_cluster=1 training.batch_size=8192
python -B scripts/active_sel/baselines/gen_gnn_distance.py \
    --dataset ptc_fr --cluster 1 --dist l2 --K_val 40
```

`gen_kcenter_greedy` and `gen_kmeans` are query-agnostic (same K
representatives for every query). `gen_infomax` runs the ICLR 2025 InfoMax
objective adapted to query-conditioned selection. `gen_wl_kernel` is the
Weisfeiler-Leman subtree kernel. `gen_gnn_distance` supports `--dist
{l1, l2, hinge}` and optional `--faiss` for IVF-based approximate retrieval.

## FMap baseline (Table 3)

FMap is a separate (non-LSH, non-AugNet) corpus-side method. The pipeline
has its own dumper and relationship generator:

```bash
bash scripts/fmap_dumpers/fmap_dump_ptc_fr.sh    # also _ptc_fm, _ptc_mr, _cox2
# then stage 5 with cluster_rel_subdir=FMap_CEMax_Asymm or FMap_CEAvg_Asymm
```

See `scripts/fmap_dump.py` for the Hydra-driven dump entry point, and the
four shell wrappers under `scripts/fmap_dumpers/` for the exact per-dataset
invocations.

## Stage 6 — Evaluation

To evaluate any downstream NSR's performance on the corresponding test set cluster:

```bash
python -B scripts/reproduce/eval_main_tables.py
```

Reads `logs/<task>.log`, replays EarlyStopping at `delta=5e-3` to pick
the best epoch ≤ 200, loads the corresponding `latestModels/<task>/<epoch>.pt`,
runs test eval, writes `testMAPs/<task>.txt`, then renders Tables 1, 2, 3,
4, 6-9, 18-21, 22-25, 26-29.
