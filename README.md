# From 0.98958 to 0.98991: LLM Re-ranking and a Reproducible Release for Entity Resolution

**Branch `vaishnavi` · Vaishnavi Bholane · Amazon ML Challenge 2026 (Team TeamCook) · final submission v28 ·
public leaderboard 0.989905**

This branch is the team's final submission. It takes the base pipeline on [`main`](../../tree/main), a four-stack
LightGBM and cross-encoder system for multilingual entity resolution at 24M-record scale, and adds four things:

1. an **audit** that explains the offline-vs-leaderboard gap;
2. a **wider candidate set** scored without retraining;
3. a fine-tuned **Qwen2.5-1.5B re-ranker** with a stage-4 decision model on the ambiguous pairs;
4. a **one-command, verified reproducible release** of the whole pipeline.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)
![Qwen](https://img.shields.io/badge/LLM-Qwen2.5--1.5B-6F42C1)
![Transformers](https://img.shields.io/badge/HF%20Transformers-4.56-FFD21E?logo=huggingface&logoColor=black)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-9ACD32)
![GPU](https://img.shields.io/badge/4×%20NVIDIA%20H200-DDP-76B900?logo=nvidia&logoColor=white)

---

## Results

| version | change | out-of-fold macro F0.5 (US/India) | public leaderboard |
|---|---|---|---|
| v24 | base pipeline (`main`) | 0.991303 | 0.989584 |
| v25 | + wider candidate set, re-scored by the trained stacks | 0.991625 | 0.989747 |
| v26 | v25 with per-country thresholds (a deliberate test, reverted) | – | 0.989694 |
| **v28** | **+ Qwen2.5-1.5B re-ranker and stage-4 re-decision** | **0.991843** | **0.989905** |

## 1. Audit: why offline ≠ leaderboard

* **The data mix.** The test set is 46.75 % India, 38.27 % US and 14.98 % France, while training is US and India
  only. The US/India out-of-fold score, re-weighted to the test mix, together with the leaderboard score implies
  France ≈ 0.980. A US/India-only offline score of ≈ 0.992 therefore corresponds to ≈ 0.990 on the leaderboard,
  which is exactly what was observed.
* **Leakage and optimism checks.**
  * Fold grouping by entity is sound.
  * Target encodings are strictly out-of-fold.
  * The in-sample pruner costs nothing measurable: fold 0, where it is held out, scores 0.991296, against
    0.991234–0.991369 on the other folds.
  * The fold-to-fold noise floor is ≈ 5·10⁻⁵, and every later decision was held to it.
* **Evidence-based rejection.** Purging French "name twins" and per-country thresholds both looked plausible. Both
  were rejected on labelled out-of-fold evidence. v26 tested the threshold hypothesis on the leaderboard and
  confirmed that 0.725 stays optimal.

## 2. Wider candidates without retraining

The pruner kept each record's top-3 entities with p1 ≥ 0.02. Widening this to **top-4 with p1 ≥ 0.01** recovers
0.19 points of pair recall. The new pairs get features and cross-encoder scores, and are then **re-scored by the
already-trained stacks** with no retraining. All training scores stay strictly out-of-fold.

| out-of-fold, 2.2M training entities | thr 0.70 | **thr 0.725** | thr 0.75 | thr 0.80 |
|---|---|---|---|---|
| original candidates (4.17 / entity) | 0.991305 | 0.991303 | 0.991294 | 0.991239 |
| widened candidates (4.46 / entity) | 0.991638 | **0.991625** | 0.991608 | 0.991524 |

The gain is positive on every fold (+0.00030 to +0.00033). It also holds on a held-out clone-decoy simulation that no
model was trained on (+0.00027), at the cost of 2.6 % more decoy merges. Leaderboard: 0.989584 → 0.989747.

## 3. LLM re-ranker on the ambiguous band

**Where the loss is.**
* 93.5 % of the remaining US/India loss is *recall*.
* Records with at least one pair whose blended score is in [0.005, 0.999) hold 7.6 % of pairs but **99.8 % of the
  misses** and 95.8 % of the wrong acceptances.

So the expensive model runs only there.

**Qwen2.5-1.5B pair classifier** (`src/llm_ce.py`, Apache-2.0, 1.54B parameters):
* **Input:** the prompt `Entity: <name | address>` / `Record: <name | address>` → "Same business:", scored through a
  last-token classification head with BCE loss.
* **Cross-fitting:** by entity halves. Each half's model scores the other half, so train and test scores have the
  same distribution.
* **Training:** PyTorch **DDP on 2 H200s per half, both halves at once**, bf16 autocast and fused AdamW, at 98 % GPU
  utilisation. The 1.01M training pairs are split between the two halves, each half trains in about 13–15
  minutes, and all 1.77M train and test pairs are scored in 4 minutes.

**Stage 4** (`src/stage4.py`) is an out-of-fold LightGBM over the four stack scores, the LLM score with its
within-record rank, gap and margin, record and entity context, 115 pairwise features and 4 cross-encoder logits. It
re-decides only the band pairs.

| out-of-fold, 2.2M training entities | AUC on the band | macro F0.5 |
|---|---|---|
| 4-stack blend (v25) | 0.98082 | 0.991625 |
| stage 4, pairwise features only | 0.98153 | 0.991704 |
| stage 4, LLM score only (v27) | 0.98195 | 0.991788 |
| **stage 4, LLM + features + cross-encoders (v28)** | **0.98240** | **0.991843** |

The gain is positive at every threshold from 0.60 to 0.80 and is about 4× the fold-to-fold noise. Leaderboard:
**0.989905**, the team's best score.

## 4. A reproducible release, verified

The whole 25-step pipeline runs behind one resumable command (`src/run_pipeline.py`). The release has:
* pinned dependencies and **pinned Hugging Face model revisions** (`src/download_models.py`);
* fixed seeds, deterministic LightGBM, sorted sampling and deterministic tie-breaking;
* three steps that the development script lacked, added so that a run from an empty folder works;
* an output checker (`src/verify_outputs.py`).

**Reproduction test:** the released zip was unpacked into an empty folder with a fresh environment.

| check | result |
|---|---|
| `candidate_pairs.tsv` | **byte-identical** to the submission (8,926,243 pairs) |
| `matching_results.tsv` | **99.96 %** identical (+2,131 / −2,309 of 5.85M pairs; GPU retraining is not bit-exact) |
| normalisation, pairwise features, stack scores | identical |
| GPU retrieval | 99.98 % identical (ties at the top-10 cut-off) |
| retrained pruner | same recall (0.98293 vs 0.98294) |
| first stack retrained from scratch | validation F0.5 0.99007, exactly the documented value |
| official validator (`--check-ids`) | PASS |

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/run_pipeline.py --data_dir /path/to/dataset --output_dir out --work_dir work --gpus 0,1,2,3
.venv/bin/python src/run_pipeline.py --list          # the 25 steps; resume any run with --from_step N
```

It takes about 8–10 h on 4× H200. `docs/REPRODUCTION.md` has the per-step runtime, verification and the full test
report, `docs/METHODOLOGY.md` the complete write-up, and `docs/ARCHITECTURE.md` the system design. The competition
data is not included.

## What this branch adds to `main`

| file | purpose |
|---|---|
| `src/run_pipeline.py` | single entry point: 25 steps, resumable, per-job logs and timings |
| `src/widen.py` | widened candidate set, re-scored by the trained stacks; final candidate file |
| `src/llm_ce.py` | Qwen2.5-1.5B band classifier (prep, DDP training, inference, merge) |
| `src/stage4.py` | stage-4 re-decision of the ambiguous pairs |
| `src/download_models.py` | pinned download of the four pretrained checkpoints |
| `src/verify_outputs.py` | format and ID checks, matches ⊆ candidates, comparison with reference files |
| `docs/ARCHITECTURE.md` | component and data-flow description |

## Author

**Vaishnavi Bholane**. Team TeamCook: Parth Pitrubhakta, Vaishnavi Bholane.

## Data, models and licences

No external data, APIs or look-up services are used. The pretrained models are XLM-RoBERTa base and large (MIT),
mDeBERTa-v3-base (MIT) and Qwen2.5-1.5B (Apache-2.0), all under 8B parameters. The competition dataset and the
predictions are not redistributed. Code released under the [MIT License](LICENSE).
