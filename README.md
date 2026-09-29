# Entity Resolution Pipeline: How the Base System Was Built (v1 → v24)

**Branch `parth` · Parth Pitrubhakta · Amazon ML Challenge 2026 (Team TeamCook) · public leaderboard 0.989584**

This branch holds the base pipeline behind the team's submissions up to v24, and the engineering story of how it got
there. Each version fixed a specific, measured failure mode. For the project overview see [`main`](../../tree/main);
for the final LLM-augmented version (0.989905) see [`vaishnavi`](../../tree/vaishnavi).

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-9ACD32)
![Transformers](https://img.shields.io/badge/HF%20Transformers-4.56-FFD21E?logo=huggingface&logoColor=black)
![Polars](https://img.shields.io/badge/Polars-1.44-CD792C?logo=polars&logoColor=white)

---

## What this pipeline does

It links 20M noisy business records (US, India, and France, which is test-only) to 3.9M reference entities. The
metric is macro F0.5 per entity, which penalises a wrong merge about four times more than a missed match.

```
normalise ─► GPU TF-IDF retrieval ─► LightGBM pruner ─► pairwise features + cross-encoders
          ─► 4 LightGBM stage-2/3 stacks (collective features) ─► one entity per record, tuned threshold
          ─► unseen-country path (vocabulary-free mode, label-free rules, self-trained cross-encoder)
```

## The build, version by version

| version | problem found | what was built | result |
|---|---|---|---|
| v1 | baseline | GPU retrieval + LightGBM pruner + pairwise LightGBM + collective stacker | validation 0.9875 |
| v2–v4 | decoys look like true matches | house-number relations, name-frequency and name-twin features; **out-of-fold target encoding** of added/dropped name tokens, legal-form, swap and city transitions | validation 0.9903; leaderboard 0.9797 |
| v5 | offline 0.990 vs leaderboard 0.980 | diagnosed test-only orphans and unseen French vocabulary; orphan simulation, target encodings missing for unseen tokens + 30 % dropout | leaderboard 0.9812 |
| v6 | test has 1.3–1.8× more same-street look-alikes | **decoy injection** (8 % of true records cloned at nearby house numbers); stronger pruner, top-3 candidates | clone-decoy merges 42 % → 4.7 % |
| v7–v8 | string features miss semantics | **XLM-R base and large cross-encoders**, fine-tuned with DDP and cross-fitted on entity halves | validation 0.99112; leaderboard 0.98658 |
| v10–v11 | English meanings transferred wrongly to French ("et Fils" read as "& Sons") | **label-free token statistics**, vocabulary-free inference mode, high-confidence French additions | leaderboard 0.98826 |
| v12–v17 | France still the main error source | structural rules from the data generator; **self-trained French cross-encoder** on pseudo-labels | leaderboard 0.989458 |
| v22–v24 | single stack plateau | mDeBERTa-v3 cross-encoder; **blend of four stacks** (1:1:3:3); France tightened back to the leaderboard-confirmed v17 state | leaderboard 0.989584 |

## Design decisions that mattered

**1. Retrieve from the record side, exactly, on the GPU.** Every record belongs to at most one entity, so each
record retrieves its top-10 entities per view (name, address, both). Cosine similarity is computed exactly as
`sparse(entities) × dense(query chunk)` with `torch.sparse.mm`, sharded over 4 H200s at about 3.4k queries/s per
GPU. There is no approximate index, so no recall is lost to one.

| candidate rule (held-out fold) | recall of true pairs | candidates / entity | ceiling F0.5 |
|---|---|---|---|
| retrieval union (top-10 × 3 views) | 0.98835 | 107.8 | 0.99651 |
| p1 ≥ 0.01 (v4/v5) | 0.98711 | 4.67 | 0.99609 |
| **top-3, p1 ≥ 0.02 (final)** | **0.98565** | **4.30** | **0.99555** |

**2. Encode *differences*, not similarities.** The generator's decoys are "same name + one extra word" (Group,
Holdings) at a nearby house number, while its noise words (Center, Services) sit at the same address. Out-of-fold
target encoding of the tokens a record adds or drops lets the model learn which is which. This was the largest single
gain (+0.0018).

**3. Decide collectively.** Stage 3 sees each pair in context: its margin over the record's best competing entity,
the entity's most confident sibling record, and where other records with the same name went.

**4. Validate on the test's difficulty, not the train's.** Four frames are scored for every model, with results on
the same held-out fold:

| validation frame | v5 | v6 | v8 | v10 |
|---|---|---|---|---|
| original | 0.99035 | 0.99007 | 0.99112 | 0.99124 |
| orphan simulation (19 % of entities deleted) | 0.98936 | 0.98918 | 0.99065 | 0.99074 |
| decoy injection (noisy, training style) | – | 0.98881 | 0.99046 | 0.99063 |
| clone decoys (never trained on) | 0.97121 | 0.98756 | – | – |

**5. Transfer to an unseen country without labels.**
* For every word a record adds, measure the *leave-one-out share of pairs with an identical house number*. The
  measure is label-free and language-independent: decoy words score 0.02–0.08, and noise words score 0.73–0.90 in
  both English and French.
* Train with vocabulary features hidden for 30 % of entities, so the model can decide without them.
* Self-train a French cross-encoder on structurally confident pseudo-labels, cross-fitted by entity halves.

**6. Blend what generalises.** The four stacks differ in features, seeds and learning rates. Their 1:1:3:3 blend
beats the best single stack on every frame, and the gain grows on harder frames: from +0.00017 on the original data
to +0.00033 on the held-out clone decoys.

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
BER_DATA=/path/to/dataset BER_CACHE=/path/to/work BER_OUTPUT=/path/to/output PY=.venv/bin/python bash run_pipeline.sh
```

`docs/REPRODUCTION.md` lists every step with its command and output. `docs/METHODOLOGY.md` is the full write-up
submitted to the competition. Hardware used: 4× NVIDIA H200, 192 cores, 1 TB RAM. The competition data is not
included.

## Code map

| module | role |
|---|---|
| `text_norm.py`, `translit.py`, `preprocess.py` | normalisation of legal forms, abbreviations, states and house numbers; native-script → Latin dictionary learned with IBM Model 1 from the training pairs |
| `blocking.py` | GPU sparse TF-IDF retrieval, sharded over 4 GPUs |
| `stage1.py` | LightGBM candidate pruner |
| `features.py` | 100+ pairwise similarity features (rapidfuzz, multi-process) |
| `match.py` | stage-2/3 stacks, out-of-fold target encoding with dropout, label-free token statistics, group and twin features |
| `simulate.py` | orphan and decoy simulations, rebuilt through the real pipeline |
| `cross_encoder.py` | cross-encoders: two-half cross-fitting, DDP training and inference |
| `ensemble_seen.py`, `combine.py`, `make_submission.py` | blend, combination and output files |
| `typecal.py`, `france_rules.py`, `ce_france.py`, `predict_france.py` | unseen-country path |

## Author

**Parth Pitrubhakta**. Team TeamCook: Parth Pitrubhakta, Vaishnavi Bholane.

Code released under the [MIT License](LICENSE).
