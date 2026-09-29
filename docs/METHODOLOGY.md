# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** TeamCook  
**Team Members:** Parth Pitrubhakta Wani, Suraj Kumar, Vaishnavi Bholane  
**Submission Date:** 29 Sep 2026 (final leaderboard file: v28, public score 0.989905)

---

## 1. Executive Summary

We resolve every Source 2/3 record to **at most one** Source 1 entity in five steps:

1. **Normalisation.** Names and addresses are normalised, and native-script names are
   translated with a dictionary learned from the training pairs (IBM Model 1).
2. **Two-stage blocking.** Exact sparse TF-IDF retrieval runs on the GPU (name, address and
   combined views), then a light LightGBM pruner keeps each record's best candidates: **5.15 per
   Source 1 entity on test** (v25; 4.81 with the leaner v6–v24 rule).
3. **Pairwise matching.** A LightGBM model scores each candidate pair on 160–176 features (depending on the stack):
   string similarities, house-number relations, out-of-fold target encodings of name/address
   differences, label-free token statistics, and fine-tuned multilingual cross-encoders
   (XLM-R base/large, mDeBERTa-v3, all MIT).
4. **Collective decision.** A stacked LightGBM uses group features (record-level competition
   between S1 entities, sibling records and name "twins"). Then a one-S1-per-record
   assignment and an F0.5-tuned threshold give the final matches. Several stacks are blended
   for countries seen in training (section 13).
5. **France (absent from training).** Structural corrections derived label-free from the data
   generator, plus a cross-encoder self-trained on French pseudo-labels, fix the stacks'
   English-vocabulary errors on French names (section 12).

Public leaderboard: v4 0.9797 → v8 0.98658 → v11 0.98826 → v17 0.989458 → v23 0.989578 → v24 0.989584 →
v25 0.989747 → **v28 0.989905, the final submission** (sections 14–15). v28 adds three things to v24:
* **Wider candidate set:** US/India records keep their top-4 S1 with p1 ≥ 0.01 (5.15 candidates per test S1).
* **LLM pair classifier:** Qwen2.5-1.5B (Apache-2.0, 1.54B parameters) is fine-tuned on the ambiguous pairs using all four H200s.
* **Stage 4:** an out-of-fold LightGBM re-decides those ambiguous pairs from the stack scores, the LLM score and the pairwise features.

Out-of-fold US/India macro F0.5 is 0.991303 (v24) → 0.991625 (v25) → **0.991843 (v28)**. France is unchanged since v24.
How the unlabelled test data and the public leaderboard were used is disclosed in section 7.

---

## 2. Methodology

### 2.1 Problem Analysis

* **Scale.** Train has 2.2M S1 and 10.3M S2+S3 records. Test has 1.73M S1 and 9.97M S2+S3.
  Each S2/S3 record matches **at most one** S1 (verified on train: no record appears in two
  ground-truth lists). There are about 3.46 matches per S1, and 5.6% of S1 are singletons.
  About 26% of train records match nothing; on test the expected share is about 40%
  (5.8 records per S1 vs 4.7 in train).
* **Name noise.** Case changes, accents, typos, legal-form variants (Pvt/Private, Ltd/Limited,
  S.A.R.L.), word reordering, bracketed legal forms, "(India)/(France)" tokens,
  honorifics (Shri, M/s), DBA strings ("X doing business as Y"), website forms
  (`patriotcoastal.com`), initials ("HF" for "Hotel Foods"), and completely unrelated trade
  names ("Arcwex"). These last ones can only be matched by address.
* **Native scripts.** About 10% of S2 names are in Devanagari, Telugu, Kannada, Tamil,
  Bengali, Gujarati, Gurmukhi, Odia or Malayalam, while S1 is always Latin. The word
  vocabulary is small (about 1.5k words). Native-script address components are only the
  16 Indian state names.
* **Address noise.** Abbreviations, component reordering, missing components (3% of
  addresses are empty), state names vs codes, city aliases (Calcutta/Kolkata), leading
  zeros, digit drops (824→24), ranges (1427-1431), "Â" mojibake and "N°".
* **Designed distractors (the key insight).** We looked at candidates whose name equals the
  S1 name plus one extra token. For tokens such as *group, holdings, public, exports,
  downtown, garments, foods* the label rate is **0%**. These are sibling businesses on the
  **same street at a nearby house number**: house numbers are equal for only ~5% of them,
  against 60–73% for true matches. Among all non-matching candidates, 20% have a house
  number within ±20 and 18% differ by one substituted digit, against 0.8% and 2.1% for
  true matches. House-number relations are therefore the strongest precision signal, and
  they are country-agnostic.
* **France (test only).** It is handled by the same code path. Country is treated as an
  open set of blocking labels and is **never used as a model feature**. The street-type
  dictionary includes French forms (rue/r, avenue/av, allée, chemin, bis/ter, N°), and the
  legal-form dictionary includes SARL, SAS, SASU, EURL, SCI, SNC and EI.

### 2.2 Solution Strategy

**Approach Type:** blocking + learned pruning + pairwise GBDT + collective (group-aware)
stacking + constrained assignment, with an LLM pair classifier and a final re-decision model on the ambiguous pairs (v28).  
**Core Innovation:** record-side retrieval plus collective, house-number-aware and out-of-fold target-encoded
features that separate the generator's look-alike decoys from true matches, detailed below.

**Core innovations in detail**

1. **Record-side retrieval.** Because every record belongs to at most one S1, we retrieve S1
   entities *for each record*. An S1's candidate list is then the set of records that
   retrieved it, which is naturally tiny.
2. **Learned pruner** on retrieval scores and their margins, so the final candidate list is
   small.
3. **Collective features.** How this S1 compares with the record's other candidate S1s, how
   the record compares with the S1's most confident sibling record, and where other records
   with the same name go ("name twins").
4. **Out-of-fold target encoding of record-vs-entity differences.** For each pair we
   encode the tokens the record *adds* to or *drops* from the S1 name and address, the
   legal-form transition ("pvt ltd → ltd"), single-token name swaps and city transitions.
   Each is replaced by its smoothed empirical match rate, computed out-of-fold. This lets
   the GBDT learn that "+ group" marks a distractor while "+ center" is noise. Unseen
   tokens (e.g. French) fall back to the prior. It was the largest single gain (+0.0018).
5. **IBM Model 1 transliteration** dictionary learned only from the training pairs.

---

## 3. Candidate Generation (Blocking)

* **Blocking keys used:** the `country` label (one block per label, open set) combined with exact sparse TF-IDF
  cosine retrieval on a name view, an address view and both together, followed by a learned LightGBM pruner.

**Stage A: exact sparse TF-IDF retrieval on GPU (`blocking.py`)**

* Blocks are per `country` label, so any label (including France) gets its own block.
* **Name view:** char 3-grams, words and consonant skeletons of the core name (legal forms
  and honorifics removed, website stems added).
* **Address view:** address words, char 3-grams of words, word bigrams within a comma
  component, and the canonical state.
* TF-IDF (sublinear tf, IDF fitted per country block) with L2 normalisation.
* Cosine similarities are computed exactly as `sparse(S1) × dense(query chunk)` products
  with `torch.sparse.mm` on 4 GPUs, at about 3.4k queries/s per GPU. That is roughly
  20 minutes per split, with no approximate nearest-neighbour search.
* For each record we keep the top-10 S1 by name, by address and by name+address.
  Recall of this union is 98.84% of all true pairs; the combined view alone reaches 97.0%
  at top-1.

**Stage B: learned pruning (`stage1.py`)**

* The pruner is a LightGBM model (400 trees). Its features are the two cosines, the ranks in
  each view, the gap to the record's best S1, the runner-up margin per view, how many records
  retrieved the S1 and the rank among them, and flags for empty address or non-Latin name.
* It is trained on 4/5 of the S1 folds. From v6 on, it also uses cheap per-pair signals (name
  token-sort ratio, address token-set ratio, house-number agreement and difference; section 8).
  **Candidate rule v6–v24: each record's top-3 S1 by p1, kept if p1 ≥ 0.02.** From v25 (and in the final v28)
  US/India records keep their top-4 S1 with p1 ≥ 0.01, while France keeps the top-3 rule (section 14). The resulting
  pairs are written to `candidate_pairs.tsv`. This is exactly the set every matching model scores: the file equals
  the rule's output and the scored set of all four stacks, and every final match is in it.

Held-out S1 fold 0 (441k entities). The ceiling is the macro F0.5 of a perfect matcher on the
candidate set:

| candidate rule (v6+ pruner) | recall of true pairs | candidates / S1 | ceiling F0.5 |
|---|---|---|---|
| retrieval union (top-10 × 3 views) | 0.98835 | 107.8 | 0.99651 |
| top-5 per record, p1 ≥ 0.005 | 0.98760 | 5.00 | 0.99627 |
| p1 ≥ 0.01 (v4/v5 rule) | 0.98711 | 4.67 | 0.99609 |
| **top-3 per record, p1 ≥ 0.02 (final)** | **0.98565** | **4.30** | **0.99555** |

The leaner rule costs 0.00054 of ceiling against the v4/v5 rule. The realized cost is smaller
(≈ 0.00025 in v5), because the extra pairs are mostly no-address records with widely shared
names, which no model resolves.

* **Candidate pairs generated:** v6–v24: 9.19M on train and 8.34M on test (4.81 per test S1, about
  1.4 per true match). **Final (v25–v28): 8,926,243 pairs on test (5.152 per test S1)**; US/India use
  top-4/p1 ≥ 0.01, France keeps top-3/p1 ≥ 0.02 (section 14). Pair recall of this rule on the labelled
  training data is 0.98487 (all folds). Against all same-country Source-1 × record pairs on test
  (6.72 × 10¹²), the reduction ratio is 99.99987%.
* **How we avoided losing true matches:**
  * three complementary retrieval views, so records with an unrelated trade name are found
    through the address and records with no address through the name;
  * translated native-script names;
  * a pruner whose threshold was chosen on held-out recall.

  82% of the remaining misses are records with an empty address and a generic name that
  is shared by many S1 entities; these are not resolvable from the data.

---

## 4. Matching Model

**Features used** (`features.py`, `match.py`; all country-agnostic). Stage 2 uses 160 (full stack), 172 (vf, vfb)
or 176 (vfm) features; stage 3 adds p2 and 17 group and twin features (178–194).

* **Name:** rapidfuzz ratio, token sort/set, partial ratio, Jaro-Winkler and Levenshtein on
  the core, full and skeleton names; ratio with spaces removed (for website names);
  similarity of the DBA alternative and the website stem; token Jaccard; exact and fuzzy
  (JW ≥ 0.88) containment both ways; first-token equality; acronym match; legal-form
  equality, conflict and missing; name-frequency counts (how many S1s and records share the
  normalised name key).
* **Address:** ratio, token sort/set and partial token set; token Jaccard and containment;
  fuzzy word containment; comma-component match rates; city containment; state agreement;
  postal-code agreement; empty-address flag.
* **Numbers:** composite house-number string ("2-37-2" vs "2-37-6", "25/2" vs "25/23":
  equality, ratio and containment); number-set Jaccard and containment; house number of each side found in the
  other; relative house-number difference; categorical relation (equal / suffix / prefix /
  indel / substitution / near ±20 / other) for the house number and for the best number
  pair; a "near-only" flag (some number is near or one substitution away, none equal).
* **Target-encoded differences** (out-of-fold by S1 fold, smoothing weight 20): for the
  sets of record-only and S1-only name tokens and non-numeric address tokens we take the
  min, max and mean of the encoded values plus the support count. We also encode the
  legal-form transition, the single name-token swap and the city transition.
* **Retrieval context:** all stage-A/B features and p1.

**Model type:** LightGBM binary GBDT (255 leaves, at least 100 rows per leaf, feature fraction 0.7, bagging 0.8,
L2 1.0). The full, vf and vfm stacks use learning rate 0.1 with 400 stage-2 and 250 stage-3 trees; vfb uses 0.05
with 800 and 500 trees and seed 7. All are 5-fold cross-fitted by S1 id, giving out-of-fold p2 for every training pair. Test scores are the
average of the 5 fold models.

**Stage 3, collective stacker:** a LightGBM model on the stage-2 features plus p2 and group
features computed from p2:

* the record's best, second-best and sum of scores, and this pair's margin over the best
  competing S1;
* the S1's sum and count of confident records and the pair's rank within them;
* rapidfuzz similarity of the record to the S1's most confident *other* record (sibling);
* name-twin features: among records with the same name key, how many are confidently
  assigned to this S1, to another S1, or to nothing.

**Decision:** each record is assigned to its arg-max S1 (one-to-one on the record side) and
kept if its score ≥ θ.

**Threshold selection method:** macro F0.5 (singletons included) maximised on held-out S1 folds. Each stack's own
θ is the best mean over the original, orphan-simulated and decoy-injected validation frames (0.725 for full,
0.75 for vf); the final US/India decisions use the four-stack blend at θ = 0.725 (section 13), and in v28 the band
pairs are re-scored by stage 4 before the same θ is applied (section 15). A per-entity expected-F0.5 decision rule was also tried and gave only
+0.00006, so we kept the simpler global threshold.

---

## 5. Results & Error Analysis

* **F_0.5 Score (macro):** 0.991843 out-of-fold on all 2.2M training S1 entities (US/India, final v28 rule).
  Public leaderboard (full test, including France): **0.989905**.

Validation uses held-out S1 fold 0 (20% of train S1 entities, about 441k entities) with the
exact challenge metric, singletons included; the rows from v23 on use all five folds out-of-fold.

| model | macro F0.5 |
|---|---|
| stage-1 pruner score only | 0.9071 |
| v1 pairwise LightGBM | 0.9844 |
| v1 + collective stacker | 0.9875 |
| v2 (+house-number relations, runner-up margins, name frequency) pairwise | 0.9859 |
| v2 + collective stacker with name-twin features | 0.9885 |
| v3 (+composite house number, target-encoded name / legal differences) pairwise | 0.9886 |
| v3 + collective stacker | 0.9902 |
| v4 (+target-encoded address / city / swap differences) + collective stacker | 0.9903 |
| v23/v24: blend of four stacks (all 5 folds out-of-fold, US/India) | 0.991303 |
| v25: + widened final candidate set (all 5 folds out-of-fold, US/India) | 0.991625 |
| **v28: + LLM stage 4 on the ambiguous band (all 5 folds out-of-fold, US/India)** | **0.991843** |
| ceiling with a perfect matcher on the v25 candidates (all folds) | 0.99544 |

**Expected test score before v28 (v25).** US/India 0.99165 re-weighted to the test mix (0.991625 train mix;
+0.000266 to +0.000357 over v24 across the held-out frames) and France ≈ 0.9799 (implied by v24's
leaderboard score; French decisions are unchanged). The weighted total is ≈ 0.9898–0.9899. France is
14.98% of test entities and carries 2.3× the error rate of US/India (section 14).

* **Common false positives (≈2.1k on validation, down from 4.4k in v1):** designed distractors, meaning the same or
  a slightly extended name on the same street at a nearby house number (2501 vs 2508 Adams
  Ave), and generic names with no address where several S1 entities share the name.
* **Common false negatives:** records with no address and a name shared by 2 or more S1
  entities. Recall is 98.5% when the name is unique but 31% / 13% when 2–3 / 4+ S1 entities
  share it. Also records with an unrelated trade name and only a fragment of the address.
  About 1.3% of true pairs never reach the candidate set.

---

## 6. Conclusion

Entity resolution at this scale is won in two places: blocking that exploits the
"each record belongs to at most one entity" structure, and collective features that let
the model reason about competing entities and sibling records. The data generator's
distractors are separable mainly through fine-grained house-number relations and through
*which* tokens a record adds to or drops from the entity's name. Target-encoding those
differences out-of-fold gave the largest late gain. None of this depends on the country,
so the same pipeline runs unchanged on the unseen French records.

**Things tried that did not help:** a per-entity expected-F0.5 decision rule (+0.00006),
more boosting rounds (no gain beyond the final tree counts; the slower vfb stack is kept only as a blend member), and a fourth stacking
level (+0.0001), which was not kept for simplicity.

---

## 7. Fair Play, Data Use and Model Licences

**External data.** No external database, API, web service, geocoder or downloaded dataset is used to look up
or resolve entities. The only network access is downloading the four public pretrained checkpoints listed below
from the Hugging Face hub. `text_norm.py` contains hand-written normalisation tables: US state names and codes,
Indian states, French regions and their departments, legal-form variants, and English, Indian and French address
abbreviations. These are general knowledge written into the code; nothing is looked up per record.

**Labels.** The only labels used are `train_ground_truth.tsv`. No test labels exist, and no test record was
labelled by hand.

**Use of the unlabelled test inputs (disclosed for review).**
* *Label-free statistics computed on each split, including test:* TF-IDF document frequencies per country block,
  name-frequency counts, token document frequencies, and the same-house-number rate of words that records add
  or drop (sections 10–11).
* *French self-training (section 12, v17 onward):* France has no training data, so XLM-R large is fine-tuned on
  pseudo-labels of French **test** candidate pairs (`ce_france.py`). Positives are our own accepted pairs;
  negatives are structurally confident decoys. The models are cross-fitted by halves of the French Source-1
  entities. `predict_france.py` fits French target encodings on the same pseudo-labels. No ground truth is involved.
* *Rules designed from label-free test analysis:* France's noise words (`services`, `groupe`, listed in
  `typecal.py`) and the French decoy patterns in `france_rules.py` were derived from label-free statistics and
  inspection of French test records. The orphan and decoy simulation rates in `simulate.py` were set to match
  test statistics (records per entity, near-miss density).

**Use of the public leaderboard.** Public scores were used to choose between submissions, and they informed two
decisions that are part of the final pipeline: France was tightened back to its v17 state in v24 (section 13),
and per-country thresholds were rejected after v26 (section 14). We also prepared diagnostic files with the
French predictions removed, to separate France's share of the public score. The final ranking uses the private
leaderboard.

**Models.** All are within the MIT / Apache-2.0, ≤ 8B-parameter rule. Licence and size were checked on the model
cards before use; parameter counts are those of the fine-tuned models.

| model | Hugging Face id @ revision | licence | parameters | used for |
|---|---|---|---|---|
| XLM-RoBERTa base | `FacebookAI/xlm-roberta-base` @ `e73636d4` | MIT | 278,885,778 | cross-encoder (section 9) |
| XLM-RoBERTa large | `FacebookAI/xlm-roberta-large` @ `c23d21b0` | MIT | 561,192,082 | cross-encoder (two epochs); French self-training |
| mDeBERTa-v3 base | `microsoft/mdeberta-v3-base` @ `a0484667` | MIT | 278,810,113 | cross-encoder of the vfm stack (section 13) |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` @ `8faed761` | Apache-2.0 | 1,543,714,304 | pair classifier on the ambiguous band (section 15) |
| LightGBM | – | MIT | – | pruner, stage-2/3 stacks, stage 4 |

**Libraries** (pinned in `requirements.txt`): numpy, scipy and scikit-learn (BSD); polars, LightGBM and RapidFuzz
(MIT); PyTorch (BSD-style); transformers, huggingface-hub, tokenizers, sentencepiece and safetensors (Apache-2.0);
protobuf (BSD-3). One library is copyleft: **Unidecode (GPL-2.0-or-later)**. It only transliterates characters
to ASCII in `translit.py` and `text_norm.py`, and it is not a model.

---

## 8. Near-miss Density and Leaner Blocking (v6)

v5 scored **0.9812** on the leaderboard (offline 0.9904). Diffing v4 and v5 decisions on test
without labels showed that the France fix mostly *swapped* doubtful decisions: it added 12.5k
French pairs with a near house number while dropping 13.1k. The v5 changes did not regress
records without rare words (entity F0.5 0.98936 → 0.98939).

The missing factor was **near-miss density**. High-scoring same-street look-alikes (house
number within 20, name similarity ≥ 90) per entity are 1.8× (US) and 1.3× (India) more
common in test than in train. Our earlier simulations never added such competitors. We clone
8% of true records at a house number 1–20 away, sometimes with a changed legal form or an
added business word, and push the clones through the full pipeline (`simulate.py decoys`).
On that set v5 merges 42% of the decoys, and its F0.5 falls to 0.9712.

**v6 changes**

* **Training data:** v6 trains on original + orphan simulation + noisy decoy injection. Half
  of the decoys start from the clean Source-1 text, and all get independent surface noise so
  they are not verbatim clones.
* **Blocking:** the stage-1 pruner gains cheap per-pair signals (name token-sort ratio,
  address token-set ratio, house-number agreement and difference). The candidate set is
  each record's top 3 pairs with p1 ≥ 0.02. Validation: 4.30 candidates per entity at
  98.57% recall (v5: 5.18 at 98.71%). **Test: 4.81 candidates per entity (v5: 6.26).**

| validation set (fold 0) | v5 | v6 |
|---|---|---|
| original | 0.99035 | 0.99007 |
| orphan simulation | 0.98936 | 0.98918 |
| pseudo-new-country | 0.98906 | 0.98865 |
| decoy injection, clone style (never used in training) | 0.97121 (42% of decoys merged) | 0.98756 (4.7% merged) |
| decoy injection, noisy style (training style, held-out fold) | – | 0.98881 (1.8% merged) |

---

## 9. Cross-encoder Feature (v7)

A multilingual cross-encoder, **XLM-RoBERTa-base** (MIT licence, 278M parameters), reads the
raw `"name | address"` strings of both sides and outputs a match logit. Two models are
cross-fitted on hash halves of the Source-1 entities. Each is trained on one half (all
distinct negatives, including decoy and orphan pairs, plus an equal number of positives
weighted towards hard ones) and scores only the other half, so every training pair has an
out-of-fold score. Test pairs get the average of the two models. The score, its margin over
the record's best other candidate, and its rank within the record are stack features.
Training takes about 8 minutes per half on 2 H200 GPUs (bf16, dynamic padding).

Pretrained multilingual knowledge is what the target encodings lack for France: "Groupe"
sits close to "Group".

| validation set (fold 0) | v6 | v7 |
|---|---|---|
| original | 0.99007 | 0.99098 |
| orphan simulation | 0.98918 | 0.99042 |
| decoy injection (noisy, training style) | 0.98881 | 0.99033 |
| decoy injection (clone style, never trained on) | 0.98756 | 0.98879 |
| pseudo-new-country (all tokens unseen) | 0.98865 | 0.99063 |

The cross-encoder alone reaches AUC 0.9978 (the stack reaches 0.9995). As a feature it adds
+0.0009 on the original data and +0.0020 when every token is unseen.

**v8 (submitted):** adds a second cross-encoder, **XLM-RoBERTa-large** (MIT licence, 560M
parameters, one epoch, AUC 0.9982 alone). Its score features sit next to the base model's.

| validation set (fold 0) | v7 | v8 |
|---|---|---|
| original | 0.99098 | 0.99112 |
| orphan simulation | 0.99042 | 0.99065 |
| decoy injection (noisy) | 0.99033 | 0.99046 |
| pseudo-new-country | 0.99063 | 0.99083 |

---

## 10. Language-independent Noise-word Statistics and Vocabulary-free Mode (v10, submitted)

v8 scored **0.98658** on the leaderboard. Its French predictions had fallen to 3.17 matches
per entity (US/India: 3.39/3.37; training rate: 3.46), and it rejected 7× more same-address
French pairs than US. The cause was vocabulary transfer in the wrong direction.

The generator adds language-specific *noise* words to true records: "Center", "Services" in
English; "et Fils", "Associés", "Développement" in French. It adds *decoy* words to
look-alike businesses: "Group", "Holdings"; "Holding", "Participations". In training,
"Sons" and "Associates" are **decoy** words, while in French "Fils" and "Associés" are
**noise** words. The multilingual cross-encoder carries the English lesson into French and
scored those same-address French true pairs at a median logit of −4.6.

**Label-free token statistic.** For each word a record adds (or drops) relative to the entity
name, we take the leave-one-out share of candidate pairs in the same split and country that
carry that word and have an *identical* house number. Decoy words sit on neighbouring house
numbers and noise words on the entity's own number:

| word | share, train | word | share, France test |
|---|---|---|---|
| group / holdings / sons / associates | 0.02 – 0.08 | holding / participations | 0.06 – 0.08 |
| center / services / trust | 0.73 – 0.77 | fils / associés / services | 0.85 – 0.90 |

**Vocabulary-free mode.** The target encodings and cross-encoder scores are vocabulary-
dependent. During training they are hidden together for 30% of entities, and at inference
they are hidden for any country label absent from training, which gets its own threshold
tuned on this path (0.65 vs 0.75).

| validation set (fold 0) | v8 | v10 |
|---|---|---|
| original | 0.99112 | 0.99124 |
| orphan simulation | 0.99065 | 0.99074 |
| decoy injection (noisy) | 0.99046 | 0.99063 |
| vocabulary-free path (all vocabulary features hidden) | – | 0.98824 |

On test, France returns to 3.46 matches per entity. Same-address French pairs with a single
noise word are accepted at 97–98% (v8: 7–28%), decoy-word pairs at 0–2%, and US/India are
unchanged.

---

## 11. Final Decision: Full Stack + Vocabulary-free Additions for Unseen Countries (v11, submitted)

Leaderboard feedback located the French problem precisely:

* v8, the full stack with target encodings and cross-encoders, scored 0.98658.
* v10, which used the vocabulary-free path for France, scored ≈ 0.986. It added 81k French
  matches, but they were a mix of true matches (~62–71%) and decoys, around break-even for
  F0.5.

**v11** keeps every full-stack decision and adds a vocabulary-free French match only in change
types that are ~99% true in the training data (`src/combine.py`):

* same address (identical house number, address token-set ≥ 95), and
* the record only *drops* name words, or adds / swaps in one word whose label-free
  same-house-number rate is ≥ 0.7 (a noise word).

That adds 30,671 French matches (3.17 → 3.29 per French entity). **Leaderboard: 0.988258
(+0.00168 over v8).**

| version | change | public leaderboard |
|---|---|---|
| v4 | pairwise + collective LightGBM | 0.9797 |
| v5 | robustness to unseen vocabulary, orphan-augmented training | 0.9812 |
| v8 | + decoy-injection training, leaner blocking, XLM-R base + large cross-encoders | 0.98658 |
| v10 | + vocabulary-free path for unseen countries (all of it) | ≈ 0.986 |
| **v11** | **v8 + high-confidence vocabulary-free additions for unseen countries** | **0.98826** |

The pipeline trains both stacks on the same candidates and features (`MODEL_TAG=full_` and
`MODEL_TAG=vf_` in `run_pipeline.py`), writes each stack's decisions, and combines them. Running
`make_submission.py` for each stack and then `combine.py` reproduces the submitted file exactly.

---

## 12. France: Structural Corrections and a Self-trained French Cross-encoder (v13–v24)

France is absent from training, so the stacks judge French name changes with US/Indian word
meanings. All the corrections below are **label-free**. They use structure that the data generator
shares across countries. That structure was measured on the training labels and checked against US
test decisions, and it was never tuned to French labels (there are none). They live in
`src/france_rules.py`, `src/typecal.py` and `src/ce_france.py` (pipeline steps 15–18).

**Generator structure used**

* *Noise words are appended after the legal form.* Among same-address one-word name swaps, the US
  noise words (center, services, service, partners) sit after the legal form 49% of the time
  ("Hitech Ventures Limited" → "Hitech Limited Services"). Real descriptor swaps almost never do.
  In training, real-word descriptor swaps ("Solutions" → "Technologies") are 0–15% true: they are
  decoys. In France, only **services** and **groupe** show the appended-noise signature (55%).
  French descriptors (club, amicale, école, comité, …) sit there 1–2% of the time. The stacks read
  "groupe" as the US decoy word "group" (US same-house-number rate 0.017 vs 0.50 in France) and
  rejected 98% of these pairs.
* *Decoys are "same name, different legal form, next door", or "name + extra token".* Same core
  name at a near house number is 93–98% true when the legal form is unchanged, and 6–20% true when
  it differs or one is added. In France the extra token is often "France" before the legal form
  ("Calais Hopital France SARL" #67 vs "Calais Hopital SARL" #62). "France" appended *after* the
  legal form is noise.
* *Entity sizes follow one distribution.* The number of true records per Source-1 entity has
  exactly the same distribution in US and India training data (0: 5.58/5.59%, 1: 5.4%,
  2: 17.0%, …). The distance of France's predicted histogram to it is a label-free progress
  measure: v11 0.088 → v15 0.074 → v16 0.065 → v17 0.055 (US test: 0.035).

**Steps** (each is one function in `france_rules.py`)

| step | change | pairs |
|---|---|---|
| v12 | ruleset v12 in `combine.py` (vocabulary-free additions with a different address format / missing or dropped house-number digit; drops near-house-number full-stack pairs the vocabulary-free path rejects) | +5,443 / −911 |
| v13 | remove accepted same-address one-word swaps to mid-rate words (French descriptor decoys) | −5,925 |
| v15 | add clean "groupe" add/swap pairs (same address, same house number, missing or dropped digit) | +8,175 |
| v16 | add pairs in **vocabulary-free types** (address relation × legal-form relation × street similarity × name relation) whose training truth is ≥ 0.9 and whose US test acceptance is ≥ 0.8: same core / one word dropped / genuine typo | +4,732 |
| v17 | **self-trained French cross-encoder** (below), gated by the training-truth type prior | +5,753 / −517 |
| v18 | second self-training round | +903 / −121 |
| v19 | entity-aware gating: break-even precision is ~57% for an entity with ≤ 1 match vs ~72% otherwise; plus unique-name name-only records | +336 |
| v20 | random-token name replacements at an address held by exactly one candidate entity | +30 |
| v21 | full stack **re-scored on France** with French target encodings (fitted on French pseudo-labels of the other entity half) and the French cross-encoder logits (`predict_france.py`); its pairs are added only where the type prior agrees | +959 / −91 |

**Self-trained French cross-encoder (`ce_france.py`).** XLM-R large (second-epoch checkpoint) is
fine-tuned on French pseudo-labels, cross-fitted by two halves of the French Source-1 entities.
Each model scores only the other half, so every French pair gets an out-of-fold score.
Positives are the accepted pairs. Negatives are only structurally confident decoys: records
already matched elsewhere, descriptor swaps, "+France" away from the same address, legal-form
changes at a near house number, and pairs both stacks reject in low-truth types. Ambiguous pools
are left out, so that possible false negatives are not taught as negatives. Out-of-fold
agreement with the decisions: AUC 0.9948 (round 1), 0.9958 (round 2). A pair is added only when
the cross-encoder (q ≥ 0.9, no competing candidate ≥ 0.5) **and** the training-truth type prior
(≥ 0.8) agree.

**Leaderboard:** v17 = **0.989458** (+0.0012 over v11; US/India decisions identical).

The re-scored full stack alone brings France's entity-size histogram closest to the generator
(L1 0.040 vs 0.053). It is over-confident on name-only records whose name is shared by several
entities, though: the cross-encoder cannot see competition between entities. So only its
type-prior-confirmed pairs are used.

## 13. Countries Seen in Training: Blend of Stacks

The out-of-fold comparison on the training data (US/India macro F0.5) showed that the
vocabulary-free-capable stack is slightly *better* in-domain than the full stack used for
US/India decisions since v11. Two more stacks were then trained for diversity:
* **vfb**: vf configuration, differently seeded, learning rate 0.05, 800/500 rounds;
* **vfm**: vf configuration plus a fourth cross-encoder, **mDeBERTa-v3-base** (MIT, 279M
  parameters, out-of-fold AUC 0.99809, between XLM-R base 0.99782 and large 0.99816).

Out-of-fold US/India macro F0.5 on all training folds, with each stack's saved test-time models,
on four frames. The clone-style decoy frame was never used to train any stack:

| decision rule (thr 0.725) | original | orphan-simulated | decoy-injected (noisy) | decoy-injected (clone, held out) |
|---|---|---|---|---|
| full stack (US/India decisions up to v21) | 0.991136 | 0.990593 | 0.990486 | 0.989170 |
| vf stack | 0.991240 | 0.990678 | 0.990649 | 0.989348 |
| vfb stack | 0.991254 | 0.990745 | 0.990660 | 0.989387 |
| vfm stack | 0.991261 | 0.990746 | 0.990649 | 0.989370 |
| blend 0.3·full + 0.7·vf (v22) | 0.991265 | 0.990731 | 0.990684 | 0.989428 |
| **blend full 1 : vf 1 : vfb 3 : vfm 3 (v23, v24)** | **0.991303** | **0.990812** | **0.990718** | **0.989497** |

The blend's gain over the full stack *grows* on harder frames (+0.000167 → +0.000327). Its
weights and threshold are within 3·10⁻⁶ of each frame's optimum. The threshold profile is
flat on average (0.725: +0.000236, 0.75: +0.000238 mean gain over the four frames), so the
blend was not re-tuned.

`ensemble_seen.py` makes the final US/India decisions from the blend. French decisions are not
affected. Threshold 0.725 was kept rather than the marginally better 0.70 (+0.000003), because the
decoy-injected frame, which is closest to the test set's decoy density, prefers higher thresholds.

| version | change | public leaderboard |
|---|---|---|
| v4 | pairwise + collective LightGBM | 0.9797 |
| v5 | robustness to unseen vocabulary, orphan-augmented training | 0.9812 |
| v8 | + decoy-injection training, leaner blocking, XLM-R base + large cross-encoders | 0.98658 |
| v10 | + vocabulary-free path for unseen countries (all of it) | ≈ 0.986 |
| v11 | v8 + high-confidence vocabulary-free additions for unseen countries | 0.98826 |
| v17 | + France structural corrections and self-trained French cross-encoder (v12–v17) | 0.989458 |
| v23 | + second self-training round, entity-aware gating, re-scored full stack (France); 4-stack blend (US/India) | 0.989578 |
| v24 | v23 with France tightened back to the v17 state (below) | 0.989584 |
| v25 | v24 + widened final candidate set for US/India, re-scored by the trained stacks (section 14) | 0.989747 |
| v26 | v25 with per-country thresholds US 0.85 / India 0.80 (a deliberate bet, section 14) | 0.989694 |
| v27 | v25 + stage 4 with the LLM score (section 15) | not submitted |
| **v28** | v25 + stage 4 with the LLM score, pairwise features and cross-encoder logits (section 15) | **0.989905 (final submission)** |

**Reading the v23 score (label-free diagnosis).** The held-out 0.991302 is a US/India-only figure. With
France at ≈ 0.980 (15% of entities), the expected full-test score is 0.9896, which matches the
leaderboard. So there is no reopened gap: the remaining distance is France. v23 − v17 = +0.00012 on
the leaderboard, while the frames above put the US/India blend alone at +0.00014 to +0.00028. So the
post-v17 French additions (v18–v21: 2,228 pairs, 735 of them at a near house number) were net
break-even or harmful, although the train-truth type prior (built on US/India, where same-city name
twins are 1% of entities vs 29% in France) estimated ≈ 94% precision for them. **v24** keeps v17's
French decisions, the 212 post-v17 removals, and only the 357 post-v17 additions that both
cross-encoder rounds (q ≥ 0.9) and the type prior (≥ 0.9) confirm on the same street, excluding
name-only and near-house-number records (`france_rules.py tighten`). US/India decisions are identical
to v23, so v24 − v23 measures this one change.

---

## 14. Final Audit (V2): Gap Diagnosis, Leakage Checks and Last Experiments

This section was recomputed end to end on 27 Sep with the shipped code and caches. The audit scripts and their
logs are in the development workspace (`analysis/`); they are analysis only and not part of the pipeline package.

**Measured test country mix.** India 809,986 (46.75%), US 663,106 (38.27%), France 259,452
(14.98%) of 1,732,544 Source 1 entities. Training is 60% US and 40% India, so offline numbers are
re-weighted to the test mix before comparing with the leaderboard.

**Expected score, per country.** Out-of-fold macro F0.5 of the shipped US/India blend (all 5 folds, 2.2M
entities, singletons included): India 0.991251, US 0.991338. Re-weighted to the test mix, that is
F_us+in = 0.99129. With w_fr = 0.1498:

    LB = (1 − w_fr)·F_us+in + w_fr·F_fr   →   0.989584 = 0.8502·0.99129 + 0.1498·F_fr   →   F_fr ≈ 0.9799

Reaching 0.99024 needs **F_fr ≥ 0.98428** (France error −22%) with US/India unchanged, or
**F_us+in ≥ 0.99206** (US/India error −9%) with France unchanged. A US/India-only offline figure of
≈ 0.9920 therefore does not imply a leaderboard ≈ 0.9920. Blending in France at ≈ 0.980 turns it into
≈ 0.9903, and a figure that ignores the harder test frames (below) overstates it further.

**Leakage and optimism audit.**

| check | result |
|---|---|
| Fold grouping | All models (pruner, stage 2, stage 3, target encodings) are grouped by S1 fold (`hash(s1_id) % 5`); cross-encoders by S1 half (`hash(s1_id, 99) % 2`). Since every record belongs to at most one S1, no labelled pair is shared between folds. |
| Target encodings | Train: fold k is encoded with tables fitted on the other folds only (`te_features`). Test: tables fitted on all training pairs. |
| Stage-1 pruner | It is fitted on folds 1–4 only, so p1 is in-sample there. Measured effect on the stacks' out-of-fold F0.5: fold 0 (pruner held out) 0.991296 vs folds 1–4 0.991304 / 0.991369 / 0.991312 / 0.991234. Candidate recall per fold is 0.9828–0.9831. **No measurable optimism.** |
| Threshold | One scalar (0.725) for 2.2M entities, flat within ±0.00001 across 0.70–0.75. The fold-to-fold SD of F0.5 is ≈ 5·10⁻⁵, which is the noise floor used below. |
| Test resemblance | Share of records whose best score falls in the ambiguous band 0.3–0.85, per accepted record: test US 1.26%, test India 0.80%, OOF clean 0.98%, orphan-simulated 1.24%, noisy decoys 1.19%, clone decoys 1.35%. **Test US looks like the harder frames**, test India like the clean one. **Test France: 8.4%**, 7–8× more ambiguous; this is where the remaining error is. |
| Matches ⊆ candidates | Checked programmatically and by the official validator (PASS, including `--check-ids`). |

**What France's error is not.** France's matched-count histogram per entity (0: 5.85%, 1: 6.42%,
2: 17.9%, 3: 24.5%, 4: 21.4%, …) and its S2/S3 split match US/India's almost exactly. So the remaining
French errors are *which* records are matched, not *how many*. No label-free count diagnostic can
see them.

**Experiments run in this audit** (each against labelled US/India out-of-fold data):

| experiment | evidence | decision |
|---|---|---|
| Purge French no-address matches whose core name is shared by ≥ 2 entities ("twins"; 3,853 pairs) | The model separates twins with a large margin (chosen twin median 0.997, competing twin ≈ 10⁻⁴). The same bucket on US/India OOF is 95.8% precise (2 twins) / 96.1% (3) / 81–86% (4+). Purging it costs −5.4·10⁻⁴ (≥ 2 twins), −1.9·10⁻⁴ (≥ 3) and −0.5·10⁻⁴ (≥ 4) on both the clean and decoy frames. | rejected |
| Per-country thresholds | US and India optima are both 0.70–0.725 on the clean, orphan and noisy-decoy frames. Only the clone-decoy frame prefers 0.80–0.85 (+0.0003), so the choice depends on which frame the test resembles. | rejected (not robust) |
| Dense multilingual retrieval view | 82% of the remaining blocking misses are no-address records with names shared by many entities; a dense view cannot rank those either. | not built |
| Wider final candidate set (**shipped as v25**, see below) | +0.000322 out-of-fold on all 5 folds, +0.000330 on fold 0 (pruner held out); held-out clone-decoy frame +0.000219 (test mix +0.000266), positive at every threshold 0.70–0.85, decoy false merges +2.6% | **kept** |

**v25: a wider final candidate set, re-scored by the trained stacks.** The ceiling table (below) shows that
top-4 / p1 ≥ 0.01 recovers 0.19 points of pair recall (+0.0007 ceiling F0.5) for +0.29 candidates per
S1 on train. `widen.py` builds that set, featurises only the new pairs (637k train / 588k test), scores
them with the four existing cross-encoders (out-of-fold halves on train; one H200 per model variant),
and re-scores the whole widened frame with each stack's saved fold models. Group features are
recomputed on the wider set; nothing is retrained. On the training data this is strictly out-of-fold:
fold-k models score fold-k entities, and target encodings for fold k use tables fitted on the
original frame's other folds.

| out-of-fold, all 2.2M training S1 (blend 1:1:3:3) | thr 0.70 | **thr 0.725** | thr 0.75 | thr 0.80 |
|---|---|---|---|---|
| shipped candidates (4.17 / S1 on train) | 0.991305 | 0.991303 | 0.991294 | 0.991239 |
| widened candidates (4.46 / S1 on train) | 0.991638 | **0.991625** | 0.991608 | 0.991524 |
| gain, re-weighted to the test US/India mix | +0.000368 | **+0.000357** | +0.000348 | +0.000321 |

Per fold at 0.725: +0.000330 / +0.000318 / +0.000329 / +0.000334 / +0.000297 (fold 0 is the one where the
pruner is held out). 6,444 entities improve and 2,285 get worse. 5,932 pairs are added (94.8% true) and
3,030 removed (2,021 true; records that move to a better candidate or fall below the threshold after the
group features change). Shared pairs keep their scores (correlation 0.99958). On test the widened rule
adds 4,093 and removes 1,379 US/India matches. France's candidates and matches are byte-identical to
v24. Candidates grow from **4.81 to 5.15 per S1** (US 4.65 → 5.03, India 4.93 → 5.35, France 4.85
unchanged). This trade costs candidate-set size, which is graded separately, in exchange for the
largest validated F0.5 gain available.

**Held-out clone-decoy check.** Test US looks like the harder frames, so the widened rule was also checked
on a fresh clone-decoy simulation that no model was trained on: 489k decoy records, each a verbatim clone of
a true record at a nearby house number or with a tweaked name. Both rules were scored on the same
simulated data (development analysis scripts, not part of the package):

| clone-decoy frame (blend 1:1:3:3) | thr 0.70 | **thr 0.725** | thr 0.75 | thr 0.80 | thr 0.85 |
|---|---|---|---|---|---|
| shipped candidates (4.38 / S1) | 0.988141 | 0.988244 | 0.988340 | 0.988509 | 0.988652 |
| widened candidates (4.69 / S1) | 0.988374 | **0.988462** | 0.988550 | 0.988702 | 0.988826 |
| gain, test US/India mix | +0.000279 | **+0.000266** | +0.000255 | +0.000238 | +0.000215 |
| decoys merged: shipped → widened | 37,890 → 38,872 | 36,576 → 37,513 | 35,193 → 36,092 | 32,306 → 33,128 | 29,034 → 29,732 |

The gain holds on the decoy-heavy frame (US +0.00009, India +0.00041). The widened set lets 2.6% more
decoys through, but that costs less than the recall it recovers.

**Leaderboard feedback and v26.** v25 scored **0.989747** (+0.000163 over v24). That is 0.71× the clone-frame
prediction and 0.53× the clean-frame one, so the test set is harder than even the clone-decoy frame. On the
widened scores the two frames disagree on the threshold (`analysis/thr_widened.log`; change vs 0.725, ×10⁻⁶):

| threshold | clean US | clean India | clone US | clone India |
|---|---|---|---|---|
| 0.80 | −107 | −91 | +327 | +110 |
| 0.85 | −263 | −193 | +500 | +158 |
| 0.90 | −519 | −387 | +654 | +132 |

**v26** follows the frame the leaderboard points to: US 0.85, India 0.80 (a development switch, removed from the final package).
It is v25 minus 6,462 low-margin US/India matches; France and the candidates are unchanged. Expected
change vs v25: +0.00024 if the test behaves like the clone frame, −0.00014 if like the clean frame.
This was a deliberate bet on the leaderboard evidence, not a validated improvement. **Result: v26 scored
0.989694 (−0.000053 vs v25).** Solving −143 + h·(243 + 143) = −53 (×10⁻⁶) gives h ≈ 0.23. So on the threshold axis
the test behaves ≈ 23% like the clone frame and ≈ 77% like the clean frame. Under that mix, 0.725 is optimal
for both countries (US 0.80: −7, India 0.80: −45, US 0.70: −13, India 0.70: −6 ×10⁻⁶). **Thresholds therefore
stay at 0.725 (v25 and the final v28).**

**Expected leaderboard (before v25 was scored).** v24 scored 0.989584. With 85% of test entities in US/India, a gain of
+0.000266 (clone frame) to +0.000357 (clean frame) there gives **≈ 0.98981–0.98989**. That is still
≈ 0.0004 short of 0.99024. The rest has to come from France, where no label-free diagnostic available today can
measure which records are right.

**Blocking trade-off (recomputed on all training folds; perfect-matcher ceiling).** This table uses all
five folds, so the numbers are slightly below the fold-0 table in section 3:

| rule | candidates / S1 | pair recall | ceiling F0.5 (US / India) |
|---|---|---|---|
| top-1, p1 ≥ 0.02 | 3.98 | 0.97497 | 0.99213 (0.99273 / 0.99124) |
| top-2, p1 ≥ 0.02 | 4.10 | 0.98068 | 0.99400 (0.99453 / 0.99321) |
| top-3, p1 ≥ 0.05 | 3.91 | 0.98092 | 0.99382 (0.99458 / 0.99268) |
| **top-3, p1 ≥ 0.02 (shipped)** | **4.17 (test 4.81)** | **0.98294** | **0.99472 (0.99533 / 0.99380)** |
| top-4, p1 ≥ 0.01 | 4.46 (test 5.15) | 0.98487 | 0.99544 (0.99602 / 0.99458) |
| top-4, p1 ≥ 0.005 | 4.70 | 0.98517 | 0.99556 |

Full grid: `analysis/blocking_tradeoff.tsv` (top-1…4 × p1 floor 0.005…0.2). Relative to all
same-country S1 × record pairs, the shipped set is a reduction ratio of > 99.9999%.

**Hardware during this audit.** From 21:35 the four H200s were shared with another user's job (67–133 GB
and 100% utilisation on every GPU). The widening experiment's cross-encoder stage therefore ran as
concurrent single-GPU jobs, one model variant per H200 (XLM-R base, XLM-R large ×2, mDeBERTa-v3), with
smaller batches after two out-of-memory failures. Main stage, 8 jobs (4 models × train/test), 1.23M new
pairs: all four GPUs at 82–92% SM utilisation (`nvidia-smi dmon` logs, shared with
the other job). Clone-frame stage, 4 jobs, 544k pairs, 73–104 s: 19–51% SM utilisation, because the
batch of 256 was kept after the GPUs were freed. Stack re-scoring (LightGBM) ran as 4–8 concurrent CPU
jobs on 192 cores. The v28 LLM stage added Qwen2.5-1.5B (Apache-2.0, 1.54B parameters; section 15). Its licence and
parameter count were checked on the model card before use and are listed in Appendix A.

---

## 15. LLM Stage 4 on the Ambiguous Band (v27, v28)

**Where the remaining US/India loss is.** Out-of-fold (v25 rule, all 2.2M training S1):
* **93.5% of the macro-F0.5 loss is recall.** Entities with only missed matches account for it; wrong acceptances are 5.7% and entities with both kinds of error 0.7%.
* **About half of it is blocking.** The perfect-matcher ceiling on the widened candidates is 0.99544.
* **In-candidate misses are mostly name twins.** Of the 77,643 true pairs that are in the candidates but rejected, 79% are no-address records, and 71–75% of those carry a name shared by ≥ 2 entities. They are close to irreducible.
* **The band covers nearly all errors.** Records with at least one pair whose blended score is in [0.005, 0.999) (527k records, 1.01M pairs; 7.6% of pairs) hold 99.8% of the in-candidate misses and 95.8% of the wrong acceptances.

**LLM pair classifier (`llm_ce.py`).**
* **Model:** Qwen2.5-1.5B, Apache-2.0 licence, 1,543,714,304 parameters (checked on its model card). It is fine-tuned as a sequence classifier (last-token head, BCE loss) on `Entity: <name | address>` / `Record: <name | address>`.
* **Cross-fitting:** by S1 half, as for the other cross-encoders. Model m is trained on the band pairs of half m and scores the other half. Test pairs are scored by the model of the other half of their entity, so train and test scores have the same distribution.
* **Training:** 1 epoch, lr 1.5e-5, fused AdamW, bf16 autocast, batch 64 per GPU. Both halves train at once with DDP on 2 H200s each, which keeps all four GPUs at 98% utilisation and 56–75 GB (`nvidia-smi dmon`). It takes 13 minutes per half.
* **Inference:** 1.01M train and 761k test pairs in 4 minutes on the four GPUs.

**Stage 4 (`stage4.py`).** A LightGBM (400 trees, 5-fold by S1 fold, all inputs out-of-fold) re-scores only the band pairs. Pairs outside the band keep the blend, and the decision rule is unchanged (best S1 per record, threshold 0.725). Its inputs:
* the four stack scores and their blend;
* the LLM score and its rank, gap and margin within the record, plus its rank within the S1;
* record and S1 context (number of candidates, blend rank, margin and gap; the S1's confident-record count and score sum);
* all 115 pairwise features and the 4 cross-encoder logits.

| out-of-fold, all 2.2M training S1 | band AUC | macro F0.5 @ 0.725 | vs v25 |
|---|---|---|---|
| v25 (4-stack blend) | 0.98082 | 0.991625 | — |
| LLM score alone | 0.93680 | — | — |
| stage 4, pairwise features only (no LLM) | 0.98153 | 0.991704 | +0.000080 |
| stage 4, LLM only (v27) | 0.98195 | 0.991788 | +0.000164 |
| **stage 4, LLM + pairwise features + cross-encoders (v28)** | **0.98240** | **0.991843** | **+0.000218** |

The gain is positive at every threshold from 0.60 to 0.80 (+97 to +218·10⁻⁶), about 4× the fold-to-fold noise. On test,
v28 adds 8,751 and removes 2,809 US/India matches relative to v25. France and the candidate set (5.15 per S1) are unchanged.
The official validator passes with `--check-ids`, every match is in its entity's candidate list, and France is identical to v24.

**Tried and not shipped.**
* **Context-aware LLM variant:** the entity's two most confident other records and the record's strongest competing entity are added to the prompt (192 tokens). Its training loss fell faster, but it diverged with SDPA attention in bf16 and was restarted with eager attention and lr 1e-5. It was stopped when v28 was chosen as final.
* **Qwen2.5-7B:** Apache-2.0, 7.62B parameters, so within the ≤ 8B rule. A LoRA path was prepared but not run, and it is not part of the final package.
* **Orphan-frame check of stage 4:** stage 4 was trained on the clean training frame only. The planned check on the orphan-simulated frame, the offline frame whose unmatched-record density matches test, was not completed.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

* `README.md`: environment, data placement, run command, runtime, verification, reproduction test; `requirements.txt`: pinned dependencies
* `architecture.md`: pipeline diagram and component overview
* `src/run_pipeline.py`: **the entry point**. It runs the 25 steps from the provided data to both output files:
  `python src/run_pipeline.py --data_dir <dataset> --output_dir <out> --work_dir <work> --gpus 0,1,2,3`
* `src/download_models.py`: pinned download of the four pretrained checkpoints
* `src/verify_outputs.py`: format and ID checks, matches ⊆ candidates, pair-level comparison with reference files
* `src/common.py`: paths, I/O, exact macro-F0.5
* `src/text_norm.py`: normalisation
* `src/translit.py`: IBM-1 transliteration learner
* `src/preprocess.py`: parallel normalisation
* `src/blocking.py`: GPU TF-IDF retrieval
* `src/stage1.py`: candidate pruner
* `src/features.py`: pair features
* `src/simulate.py`: orphan and near-miss decoy simulations for training
* `src/cross_encoder.py`: cross-encoders (two-half cross-fitting, multi-GPU)
* `src/match.py`: stage-2/3 models, target encoding, group and twin features, decision rule
* `src/make_submission.py`: per-stack decision files
* `src/combine.py`: full stack + vocabulary-free additions for countries absent from training (ruleset v12)
* `src/typecal.py`: vocabulary-free pair types and their training truth
* `src/france_rules.py`: France post-processing steps v13–v24
* `src/ce_france.py`: French pseudo-labels and out-of-fold merge for the self-trained cross-encoder
* `src/predict_france.py`: re-scores French pairs with the full stack (French target encodings + French cross-encoder)
* `src/ensemble_seen.py`: US/India decisions from the blend of four stacks
* `src/widen.py`: widened final candidate set re-scored by the trained stacks (features, cross-encoders, scoring, candidate file)
* `src/llm_ce.py`: Qwen2.5-1.5B pair classifier for the ambiguous band (cross-fitted by S1 half, DDP on 4 GPUs)
* `src/stage4.py`: out-of-fold LightGBM that re-decides the band pairs

Models: LightGBM (MIT licence), XLM-RoBERTa-base (MIT, 278.9M parameters), XLM-RoBERTa-large (MIT, 561.2M parameters), mDeBERTa-v3-base (MIT, 278.8M parameters) and, from v28, Qwen2.5-1.5B (Apache-2.0, 1,543,714,304 parameters). All are fine-tuned on the training pairs (the large model twice: two epochs). For France, the large model is further fine-tuned on label-free pseudo-labels of the French test candidates (section 12). All are far below the 8B limit. No external data or look-up services are used.

### B. Additional Results

Detailed validation tables are in sections 8–15 (per version, per validation frame and per threshold).

**Compute.** The pipeline was developed on 192 CPU cores, 1 TB RAM and 4× NVIDIA H200 (141 GB each). Measured
stage times: GPU retrieval ≈ 20 minutes per split; each LightGBM stack ≈ 35–40 minutes with 150–160 threads;
XLM-R base cross-encoder ≈ 8 minutes per half on 2 GPUs; Qwen2.5-1.5B fine-tuning 13 minutes per half (both halves
at once on 4 GPUs) and 4 minutes of inference. The whole pipeline takes about 8–10 hours on 4× H200.

**Reproduction test (29 Sep 2026).** The submission zip was unpacked into an empty folder with a fresh environment
from `requirements.txt`, and its `run_pipeline.py` was run on the provided data. Steps 2–7 were run from scratch,
and steps 13–25 from the trained stacks and cross-encoders (their retraining, 5–7 hours, did not fit before the
deadline). The regenerated `candidate_pairs.tsv` is byte-identical to the submitted file. The regenerated
`matching_results.tsv` is 99.96% identical (+2,131 / −2,309 of 5,854,185 pairs; GPU re-training of the French
cross-encoder and of Qwen is not bit-exact). All deterministic steps reproduce exactly, and the retrained
pruner keeps the same candidate recall (0.98293 vs 0.98294). Details are in the package README, section 8.
