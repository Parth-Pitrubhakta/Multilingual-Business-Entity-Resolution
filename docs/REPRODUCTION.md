# Business Entity Resolution: reproduction guide

Pipeline: **normalise → GPU TF-IDF retrieval → learned candidate pruning →
pairwise LightGBM → group-aware LightGBM stacker (with XLM-R cross-encoder features) →
one-S1-per-record assignment + F0.5-tuned threshold → France post-processing
(structural corrections + self-trained French cross-encoder)**.

Only the provided training/test files are used. There are no external APIs, no geocoding
and no look-up services. Models: LightGBM (MIT) and XLM-RoBERTa base/large (MIT, 278M/560M
parameters, pretrained weights from the Hugging Face hub), fine-tuned on the training pairs
and, for France, on label-free pseudo-labels of the test candidates.

## Environment

* Python 3.11, `pip install -r requirements.txt`
* 1+ CUDA GPU for the retrieval and cross-encoder steps (developed on 4× H200: retrieval about 20 minutes per split,
  cross-encoder training 4–10 minutes per half).
  `blocking.py` shards queries over GPUs 0-3; edit `gpus=` in `retrieve()` for fewer.
* A many-core machine with large RAM (developed on 192 cores / 1 TB; peak RAM is about 250 GB).

## Data layout

The competition data is not part of this repository. Point the pipeline at it, and at a work folder
(≈ 250 GB) and an output folder, with environment variables:
`BER_DATA=/path/to/dataset BER_CACHE=/path/to/workdir BER_OUTPUT=/path/to/output`.

## Run end-to-end

```bash
# from the repository root
BER_DATA=/path/to/dataset BER_CACHE=/path/to/work BER_OUTPUT=/path/to/output PY=python bash run_pipeline.sh
```

This runs these steps (each step caches its result in `BER_CACHE`):

| # | command | output |
|---|---------|--------|
| 1 | `python translit.py` | `translit.json`: native-script → Latin word dictionary (IBM Model 1 on train pairs) |
| 2 | `python preprocess.py train test` | `{split}_norm.parquet`: normalised name/address fields |
| 3a | `python blocking.py {train,test} 10` | `{split}_retrieval.parquet`: top-10 S1 per record for each of the name, address and combined views |
| 3b | `python stage1.py train` / `test` | `{split}_stage1.parquet`: pruning scores `p1` (retrieval scores + cheap name / address / house-number similarity); each record's top-3 pairs with `p1 ≥ 0.02` form the candidate set |
| 4 | `python match.py features {train,test}` | `{split}_feats.parquet`: pairwise features for the candidates |
| 4b | `python simulate.py 0.19 1` | `train_sim_f19_s1_feats.parquet`: orphan simulation (19% of train S1 deleted, candidates and features rebuilt) |
| 4c | `python simulate.py decoys 0.08 2` | `train_dec_r8_s2_{feats,norm}.parquet`: near-miss decoys (8% of true records cloned at a nearby house number, with independent noise) |
| 5a | `USE_CE=0 python match.py train` | first stack (its out-of-fold scores pick hard positives for the cross-encoder) |
| 5b | `python cross_encoder.py prep`, `torchrun ... cross_encoder.py train/infer {0,1}`, `python cross_encoder.py merge` | XLM-RoBERTa-base, -large (two epochs) and mDeBERTa-v3-base (all MIT) cross-encoders cross-fitted on two entity halves → `ce_scores_{train,test}[_large,_large2,_mdeberta].parquet` |
| 5 | `python match.py train` | 5-fold stage-2/stage-3 LightGBM models (original + orphan + decoy data, target-encoding dropout), tuned threshold `thr.txt` |
| 5c | `MODEL_TAG={full_,vf_,vfb_,vfm_} ... match.py train` | four stacks: full-vocabulary (full), vocabulary-free-capable (vf), a differently seeded low-learning-rate vf (vfb) and vf + mDeBERTa cross-encoder (vfm); see `run_pipeline.sh` for the exact switches |
| 6 | `match.py predict test` (every tag) | `test_scores{full_,vf_,vfb_,vfm_}.parquet` |
| 7 | `make_submission.py` (both tags) | per-stack decisions in `output/full/`, `output/vf/` |
| 8 | `python combine.py output/full/matching_results.tsv output/vf/matching_results.tsv output/v12 v12` | `output/v12/matching_results.tsv` (+ `output/candidate_pairs.tsv`, the scored candidate set) |
| 9a | `python france_rules.py rules output/v12/... output/v16/...` | France steps v13 (drop descriptor-swap decoys), v15 ("groupe" noise word), v16 (vocabulary-free types, `typecal.py`) |
| 9b | `python ce_france.py prep`, `torchrun ... cross_encoder.py train/infer {0,1}` (`CE_TABLES=ce_fr`), `python ce_france.py merge`, `python france_rules.py selftrain` — twice | self-trained French cross-encoder, two rounds → v17, v18 |
| 9c | `python france_rules.py final output/v18/... cache/ce_scores_france2.parquet output/v20/...` | v19 (entity-aware gating), v20 (random-name same-address records) |
| 9d | `python predict_france.py` (full-stack switches), `python france_rules.py rescored ... output/v21/...` | v21: full stack re-scored on France with French target encodings + French cross-encoder, gated |
| 9e | `python france_rules.py tighten output/v17/... output/v21/... cache/ce_scores_france.parquet cache/ce_scores_france2.parquet output/v24/...` | v24: France back to the leaderboard-confirmed v17 state + post-v17 removals + only strictly confirmed post-v17 additions |
| 10 | `python ensemble_seen.py output/${FINAL_FRANCE:-v24}/matching_results.tsv output/matching_results.tsv full_:1,vf_:1,vfb_:3,vfm_:3 0.725` | US/India decisions from the blend of four stacks → **final `output/matching_results.tsv`** (default v24; `FINAL_FRANCE=v21` reproduces v23) |

Validate the output:

```bash
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

## Source files (`src/`)

| file | purpose |
|------|---------|
| `common.py` | paths, TSV readers, ground-truth loader, exact macro-F0.5 metric |
| `text_norm.py` | name/address normalisation: legal forms, abbreviations, states, numbers, DBA, websites, consonant skeletons |
| `translit.py` | learns the native-script → Latin name-token dictionary from the training pairs |
| `preprocess.py` | parallel normalisation of all records |
| `blocking.py` | GPU sparse TF-IDF retrieval (candidate generation, stage A) |
| `stage1.py` | LightGBM candidate pruner (candidate generation, stage B) |
| `features.py` | pairwise similarity features |
| `simulate.py` | orphan simulation (deletes a share of S1 entities) and near-miss decoy injection; both rebuild every downstream feature |
| `cross_encoder.py` | multilingual cross-encoder pair scorer (raw "name \| address" of both sides), two-half cross-fitting, multi-GPU training/inference |
| `match.py` | stage-2/3 models, out-of-fold target encoding (+dropout), label-free token features, group and name-twin features, decision rule, validation |
| `make_submission.py` | writes the candidate and matching TSVs for one stack |
| `combine.py` | full-stack matches + high-confidence vocabulary-free additions for countries absent from training (ruleset v12) |
| `typecal.py` | vocabulary-free pair types (address × legal-form × street × name relation) and their training truth |
| `france_rules.py` | France post-processing v13–v21 (each step one function, see its docstring) |
| `ce_france.py` | French pseudo-labels (label-free) and out-of-fold merge for the self-trained French cross-encoder |
| `predict_france.py` | re-scores French pairs with the full stack using French target encodings and the French cross-encoder |
| `ensemble_seen.py` | final decisions for countries seen in training from the blend of stacks |
| `expected_f.py` | entity-level expected-F0.5 decision rule (evaluated, +0.00002 out-of-fold; not used in the final pipeline) |

## Notes on exact reproduction

* The submitted models were trained in the order shown in `run_pipeline.sh` (the step 5a
  stack is the v6 model). All randomness is seeded or hash-based.
* In the original run, the two decoy simulations could give a cloned record the same ID,
  which duplicated ~43k decoy pairs (0.14% of training rows) in the cross-encoder tables.
  `simulate.py` now makes decoy IDs unique per simulation. A fresh run therefore differs
  from the submitted one only by this negligible cleanup.
* GPU top-k ties in the retrieval step can reorder a handful of equal-score candidates
  between runs.
* Cross-encoder fine-tuning (bf16, multi-GPU) is not bit-deterministic. Given the cached
  cross-encoder scores, every later step is deterministic: `france_rules.py` and `ensemble_seen.py`
  reproduce the submitted v16–v22 match sets exactly from `output/v12` (IDs within a list may be ordered
  differently, which does not affect scoring).

## Fixes after the competition

Three small fixes were applied to the submitted code so that `run_pipeline.sh` runs from an empty work folder. None
of them changes the results, which was verified in a fresh-environment reproduction test on the `vaishnavi` branch:
* `cross_encoder.py`: the held-out clone-decoy frame (scored, never trained on) is optional.
* `ce_france.py`: the French stack scores are computed with `france_rules.french_scores()`. This is identical to the
  development file it used to read: 1,257,775 pairs with the same scores.
* `make_submission.py`: a separate unseen-country threshold applies only to vocabulary-free stacks, which is how the
  submitted `full_` stack decided.

Full determinism fixes (torch seeds, sorted sampling, deterministic LightGBM) and a one-command, resumable Python
runner are on the `vaishnavi` branch.

