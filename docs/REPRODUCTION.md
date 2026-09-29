# Business Entity Resolution (TeamCook): reproduction guide

This folder regenerates both submission files, `candidate_pairs.tsv` and `matching_results.tsv`, from the provided
training and test data. System design: `ARCHITECTURE.md`. Method and results: `METHODOLOGY.md` (both in `docs/`).
All commands below run from the repository root.

Pipeline: normalise → GPU TF-IDF retrieval → learned candidate pruning → pairwise features → four LightGBM
stage-2/3 stacks with multilingual cross-encoder features → one-S1-per-record decision → France post-processing
(structural corrections, self-trained French cross-encoder) → widened US/India candidates → Qwen2.5-1.5B pair
classifier + stage-4 LightGBM on the ambiguous pairs → output files.

No external data, APIs, geocoders or look-up services are used. The only downloads are the Python packages and the
four public pretrained checkpoints (step 1).

## 1. Requirements

| | used for development and testing | minimum |
|---|---|---|
| OS | Linux x86_64 | Linux x86_64 |
| Python | 3.11.16 | 3.11 |
| GPUs | 4× NVIDIA H200 (141 GB), driver 580.159.03 (CUDA 13.0) | 2 GPUs with ≥ 80 GB (halves then run one after the other) |
| CPU / RAM | 192 cores, 1 TB RAM (peak use ≈ 250 GB) | ≥ 64 cores, ≥ 512 GB RAM (slower) |
| Disk | ≈ 250 GB free for `--work_dir` | |

## 2. Environment

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins every package the code imports, plus the tokenizer and checkpoint libraries that
transformers loads at run time. The PyTorch wheel from PyPI ships CUDA 13.0 (`torch 2.14.0+cu130`).

## 3. Data

The competition data is **not** included. Point `--data_dir` at the provided `dataset` folder:

```
<data_dir>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
<data_dir>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## 4. Run end to end

```bash
.venv/bin/python src/run_pipeline.py \
    --data_dir /path/to/student_resource/dataset \
    --output_dir /path/to/repro_output \
    --work_dir  /path/to/work \
    --gpus 0,1,2,3
```

* Use a new `--output_dir`, not the zip's `output/` folder, so that the submitted files are not overwritten.
* Each job writes a log to `<work_dir>/logs/NN_<job>.log`. Wall-clock times go to `<work_dir>/logs/timings.tsv`.
* Intermediate files stay in `<work_dir>`, so a run can be resumed with `--from_step N` (and stopped with `--to_step N`).
* `--list` prints the steps. `--model_dir DIR` uses checkpoints that were downloaded earlier (see section 5).

| step | what | modules |
|---|---|---|
| 1 | download the pretrained checkpoints (pinned revisions) | `download_models.py` |
| 2 | transliteration dictionary from the training pairs; normalise every record | `translit.py`, `preprocess.py`, `text_norm.py` |
| 3 | blocking A: GPU sparse TF-IDF retrieval per country, top-10 per view (name, address, both) | `blocking.py` |
| 4 | blocking B: LightGBM pruner `p1` (candidates: top-3 per record with `p1 ≥ 0.02`) | `stage1.py` |
| 5 | pairwise features for the candidates | `match.py features`, `features.py` |
| 6 | training simulations: 19% of S1 entities deleted (orphans); 8% of true records cloned as near-miss decoys | `simulate.py` |
| 7 | stack without cross-encoders (its out-of-fold scores pick hard positives for step 8) | `match.py train` |
| 8–11 | cross-encoders, cross-fitted on two S1 halves: XLM-R base, XLM-R large, XLM-R large second epoch, mDeBERTa-v3 base | `cross_encoder.py` |
| 12 | four stage-2/3 LightGBM stacks (`full_`, `vf_`, `vfb_`, `vfm_`), 5-fold by S1 | `match.py train` |
| 13 | score the test candidates with every stack | `match.py predict` |
| 14 | per-stack decisions; combine (ruleset v12) | `make_submission.py`, `combine.py` |
| 15 | France structural corrections v13, v15, v16 | `france_rules.py`, `typecal.py` |
| 16–17 | France: two rounds of the self-trained French cross-encoder (v17, v18) | `ce_france.py`, `cross_encoder.py`, `france_rules.py` |
| 18 | France v19–v21 (re-scored full stack), tightened to v24; blend of the four stacks for US/India | `france_rules.py`, `predict_france.py`, `ensemble_seen.py` |
| 19–21 | widened US/India candidates (top-4, `p1 ≥ 0.01`): features, cross-encoder scores, re-scoring by the trained stacks | `widen.py` |
| 22 | v25 decisions and the final `candidate_pairs.tsv` | `ensemble_seen.py`, `widen.py cands` |
| 23 | Qwen2.5-1.5B pair classifier on the ambiguous band, cross-fitted by S1 half (DDP, 2 GPUs per half) | `llm_ce.py` |
| 24 | stage-4 LightGBM re-decides the band pairs → the final `matching_results.tsv` | `stage4.py` |
| 25 | check both output files | `verify_outputs.py` |

## 5. Pretrained weights

Step 1 downloads these checkpoints from the public Hugging Face hub into `<work_dir>/models`
(`python src/download_models.py <dir>` does the same on its own):

| model | repository @ revision | licence | parameters |
|---|---|---|---|
| XLM-RoBERTa base | `FacebookAI/xlm-roberta-base` @ `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | MIT | 278,885,778 |
| XLM-RoBERTa large | `FacebookAI/xlm-roberta-large` @ `c23d21b0620b635a76227c604d44e43a9f0ee389` | MIT | 561,192,082 |
| mDeBERTa-v3 base | `microsoft/mdeberta-v3-base` @ `a0484667b22365f84929a935b5e50a51f71f159d` | MIT | 278,810,113 (fine-tuned model) |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` @ `8faed761d45a263340a0528343f099c05c9a4323` | Apache-2.0 | 1,543,714,304 |

All later steps run with `HF_HUB_OFFLINE=1`. Every model is fine-tuned inside the pipeline; no fine-tuned
weights are shipped.

## 6. Runtime

These times come from 4× H200 and 192 cores. They were measured in the reproduction test (section 8), with the machine
shared between the two test runs and another job:

| stage | time |
|---|---|
| 1: checkpoint download | 1.7 min |
| 2: normalisation | 4.4 min |
| 3: GPU retrieval, both splits | 45 min (GPUs shared with other jobs; ≈ 20 min per split when free) |
| 4: pruner | 23 min |
| 5: pairwise features | 8.4 min |
| 6: training simulations | 21 min |
| 7: first stack | 40 min |
| 8–12: four cross-encoders, four stacks | ≈ 5–7 h (development run; not re-timed) |
| 13: scoring with the four stacks | 10 min |
| 14–18: France chain (two self-training rounds) | 33 min |
| 19–22: widened candidates | 20 min |
| 23: Qwen2.5-1.5B fine-tuning and inference | 20 min |
| 24–25: stage 4, checks | 2.5 min |
| whole pipeline | ≈ 8–10 h in total (steps 8–12 not re-timed) |

## 7. Expected output and verification

`<output_dir>/candidate_pairs.tsv` and `<output_dir>/matching_results.tsv` are tab-separated. They have the
challenge headers and one row per test Source-1 entity (1,732,544 rows plus the header). The submitted files
hold 8,926,243 candidate pairs (5.152 per entity) and 5,854,185 matches (3.379 per entity).

```bash
# format, ID validity, matches ⊆ candidates, and the pair-level difference to the submitted files
.venv/bin/python src/verify_outputs.py --output_dir /path/to/repro_output \
    --data_dir /path/to/student_resource/dataset --reference_dir /path/to/submitted_output
# official validator
python3 student_resource/utils/validate_submission.py --matching /path/to/repro_output/matching_results.tsv \
    --candidate /path/to/repro_output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

## 8. Determinism and reproduction test

* **Fixed seeds and orders.** All randomness is seeded or hash-based: S1 folds `hash(s1_id, 42) % 5`, S1 halves
  `hash(s1_id, 99) % 2`, and sampling seeds. Torch seeds are fixed for classification-head initialisation.
  LightGBM runs with `deterministic=True`. Intermediate tables are sorted before sampling, and ties in the arg-max
  decision go to the smallest `s1_id`. Keep `polars==1.44.2`: folds and halves depend on its hash function.
* **What still varies.** GPU fine-tuning of the cross-encoders and the LLM (bf16, multi-GPU) is not bit-exact.
  A fresh run therefore reproduces the submitted files closely but not byte-for-byte.

**Reproduction test (29 Sep 2026).** The zip was unpacked into an empty folder, and a fresh venv was created from
`requirements.txt` (Python 3.11.16, `torch 2.14.0+cu130`). Two runs used only the zip's code and the provided data.

* **Test A: steps 1 and 13–25.** Checkpoint download → scoring with the four stacks → France chain with both
  self-training rounds → widened candidates → Qwen fine-tuning → stage 4 → checks. It started from copies of the
  trained stacks and cross-encoders of the development run, because retraining them (steps 8–12) takes 5–7 hours.
  * `candidate_pairs.tsv`: **byte-identical** to the submitted file (8,926,243 pairs).
  * Deterministic steps reproduce exactly: all stack scores (step 13, maximum difference 0.0); France v12 and v16
    (identical pair sets; IDs within a list may be ordered differently, which does not affect scoring); US/India
    decisions before stage 4.
  * `matching_results.tsv`: 5,854,007 vs 5,854,185 pairs, **99.96% identical** (+2,131 / −2,309 pairs). The pairs
    of 0.27% of US, 0.18% of Indian and 0.42% of French entities change. The cause is GPU re-training of the French
    cross-encoder (steps 16–17) and of Qwen (step 23), which is not bit-exact.
  * Both files pass `verify_outputs.py` and the official validator (`--check-ids`); every match is a candidate.
* **Test B: steps 2–7 and the cross-encoder table preparation, from an empty work folder.**
  * Step 2, normalisation: identical to the development run (`translit.json` and all 24.2M normalised records).
  * Step 3, GPU retrieval: 99.98% of the retrieved pairs are identical; the rest are ties at the top-10 cut-off.
  * Step 4, pruner: 97.0–97.3% of candidate pairs are identical and recall on the labelled training data is the same
    (0.98293 vs 0.98294). The pruner is fitted on a 60M-row sample, so borderline pairs change.
  * Step 5, pairwise features: identical on every shared pair (75 feature columns, 8.09M test pairs).
  * Step 6, training simulations: the same 419,895 deleted entities and the same 562,345 decoy records as in development; positives within 0.004% (orphan frame 6,084,237 vs 6,084,030; decoy frame 7,507,778 vs 7,507,750). 96.4% and 97.4% of the (non-decoy) pairs are identical, following the step-4 candidates; decoy record IDs use the new unique format.
  * Step 7, first stack: validation macro F0.5 (fold 0) 0.99007 on the original data (documented v6 model: 0.99007), 0.98921 with orphans (0.98918), 0.98888 with decoys (0.98881), 0.98869 with all vocabulary features hidden (0.98865).
  * Cross-encoder table preparation (first job of step 8): runs cleanly (the development script failed here): 2.75M training pairs per half; 10.26M train and 8.27M test pairs to score.
* **Not re-run before the deadline:** cross-encoder training (steps 8–11) and the four stack trainings (step 12),
  5–7 hours together. The same code ran in the tests: `match.py train` in step 7 and `cross_encoder.py train` in
  steps 16–17.

## 9. Source files (`src/`)

| file | purpose |
|---|---|
| `run_pipeline.py` | single entry point (steps above) |
| `download_models.py` | pinned download of the four pretrained checkpoints |
| `common.py` | paths (`BER_DATA`, `BER_CACHE`, `BER_OUTPUT`), TSV readers, ground truth, exact macro F0.5 |
| `text_norm.py`, `translit.py`, `preprocess.py` | normalisation; native-script → Latin dictionary (IBM Model 1 on training pairs) |
| `blocking.py`, `stage1.py` | GPU TF-IDF retrieval; LightGBM candidate pruner |
| `features.py`, `match.py` | pairwise features; stage-2/3 stacks, out-of-fold target encoding, group and name-twin features, decision |
| `simulate.py` | orphan and near-miss decoy simulations for training |
| `cross_encoder.py` | cross-encoders (two-half cross-fitting, DDP training and inference) |
| `make_submission.py`, `combine.py` | per-stack decision files; full stack + vocabulary-free additions (v12) |
| `typecal.py`, `france_rules.py`, `ce_france.py`, `predict_france.py` | France path v13–v24 (country absent from training) |
| `ensemble_seen.py` | blend of the four stacks for countries seen in training |
| `widen.py` | widened candidate set re-scored by the trained stacks; final candidate file |
| `llm_ce.py`, `stage4.py` | Qwen2.5-1.5B band classifier; stage-4 re-decision |
| `verify_outputs.py` | output checks and comparison with reference files |

## 10. Changes made when packaging

The submitted files were produced by these modules, run step by step during development. For this package:
* `run_pipeline.py` replaces the development shell script and adds three steps that script lacked:
  * scoring the widened *training* frame, which steps 23–24 need;
  * computing the French stack scores inside `ce_france.py` (identical to the development file it read:
    1,257,775 pairs, same scores);
  * making the held-out clone-decoy frame optional in `cross_encoder.py` (it is scored, never trained on).
* `make_submission.py` applies a separate unseen-country threshold only to vocabulary-free stacks. This is how the
  submitted `full_` stack decided.
* Determinism fixes (section 8), and GPU and thread settings made configurable.
* Unused experiment code was removed: the expected-F0.5 rule, the context-prompt and LoRA LLM variants, per-country
  thresholds (v26), and the clone-frame analysis switches.
