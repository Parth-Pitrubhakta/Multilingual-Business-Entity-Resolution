# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** TeamCook
**Team Members:** Parth Pitrubhakta Wani, Suraj Kumar, Vaishnavi Bholane
**Submission Date:** 27 Sep 2026

---

## 1. Executive Summary

We resolve every Source 2/3 record to **at most one** Source 1 entity in five steps:

1. **Normalisation.** Names and addresses are normalised, and native-script names are
   translated with a dictionary learned from the training pairs (IBM Model 1).
2. **Two-stage blocking.** Exact sparse TF-IDF retrieval runs on the GPU (name, address and
   combined views), then a light LightGBM pruner keeps about **1.4 candidates per true match**
   (4.81 per Source 1 entity on test).
3. **Pairwise matching.** A LightGBM model scores each candidate pair on about 160 features:
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

Public leaderboard: v4 0.9797 → v8 0.98658 → v11 0.98826 → v17 0.989458 → **v23 0.989578**; v24
(sections 12–13) tightens France based on a label-free diagnosis of the v23 score. On held-out training entities the US/India decisions reach macro
F0.5 ≈ 0.9913.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA)

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

**Approach type:** blocking + learned pruning + pairwise GBDT + collective (group-aware)
stacking + constrained assignment.

**Core innovations**

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
  **Final candidate rule: each record's top-3 S1 by p1, kept if p1 ≥ 0.02.** These pairs are
  written to `candidate_pairs.tsv`, and this is exactly the set every matching model scores
  (verified: the file equals the rule's output and the scored set of all four stacks).

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

* **Candidate pairs generated:** 9.19M on train and **8.34M on test (4.81 per test S1, about
  1.4 per true match)**. This is a reduction ratio of about 99.9997% relative to
  same-country all-pairs comparison.
* **How we avoided losing true matches:**
  * three complementary retrieval views, so records with an unrelated trade name are found
    through the address and records with no address through the name;
  * translated native-script names;
  * a pruner whose threshold was chosen on held-out recall.

  82% of the remaining misses are records with an empty address and a generic name that
  is shared by many S1 entities; these are not resolvable from the data.

---

## 4. Matching Model

**Features (`features.py`, about 100 in total, all country-agnostic)**

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

**Model type:** LightGBM binary GBDT (255 leaves, learning rate 0.05, 800 trees), 5-fold
cross-fitted by S1 id, giving out-of-fold p2 for every training pair. Test scores are the
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
kept if p3 ≥ θ. θ = 0.70 was chosen by maximising macro F0.5, singletons included, on the
held-out S1 fold. A per-entity expected-F0.5 decision rule was also tried and gave only
+0.00006, so we kept the simpler global threshold.

---

## 5. Results & Error Analysis

Validation uses held-out S1 fold 0 (20% of train S1 entities, about 441k entities) with the
exact challenge metric, singletons included.

| model | macro F0.5 |
|---|---|
| stage-1 pruner score only | 0.9071 |
| v1 pairwise LightGBM | 0.9844 |
| v1 + collective stacker | 0.9875 |
| v2 (+house-number relations, runner-up margins, name frequency) pairwise | 0.9859 |
| v2 + collective stacker with name-twin features | 0.9885 |
| v3 (+composite house number, target-encoded name / legal differences) pairwise | 0.9886 |
| v3 + collective stacker | 0.9902 |
| **v4 (+target-encoded address / city / swap differences) + collective stacker (submitted)** | **0.9903** |
| ceiling with a perfect matcher on our candidates | 0.9961 |

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
more boosting rounds (stage 3 has converged by about 800 trees), and a fourth stacking
level (+0.0001), which was not kept for simplicity.

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
`MODEL_TAG=vf_` in `run_pipeline.sh`), writes each stack's decisions, and combines them. Running
`make_submission.py` for each stack and then `combine.py` reproduces the submitted file exactly.

---

## 12. France: Structural Corrections and a Self-trained French Cross-encoder (v13–v24)

France is absent from training, so the stacks judge French name changes with US/Indian word
meanings. All the corrections below are **label-free**. They use structure that the data generator
shares across countries. That structure was measured on the training labels and checked against US
test decisions, and it was never tuned to French labels (there are none). They live in
`src/france_rules.py`, `src/typecal.py` and `src/ce_france.py` (pipeline step 9).

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
| **v24** | v23 with France tightened back to the v17 state (below) | provisional |

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

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

* `run_pipeline.sh`: end-to-end reproduction (data → blocking → matching → `output/`)
* `src/common.py`: paths, I/O, exact macro-F0.5
* `src/text_norm.py`: normalisation
* `src/translit.py`: IBM-1 transliteration learner
* `src/preprocess.py`: parallel normalisation
* `src/blocking.py`: GPU TF-IDF retrieval
* `src/stage1.py`: candidate pruner
* `src/features.py`: pair features
* `src/combine.py`: final decision (full stack + vocabulary-free additions for countries absent from training)
* `src/cross_encoder.py`: XLM-R cross-encoder (two-half cross-fitting, multi-GPU)
* `src/match.py`: stage-2/3 models, target encoding, group and twin features, decision rule
* `src/make_submission.py`: writes both TSVs
* `src/typecal.py`: vocabulary-free pair types and their training truth
* `src/france_rules.py`: France post-processing steps v13–v24
* `src/predict_france.py`: re-scores French pairs with the full stack (French target encodings + French cross-encoder)
* `src/ensemble_seen.py`: final US/India decisions from the blend of four stacks
* `src/ce_france.py`: French pseudo-labels and out-of-fold merge for the self-trained cross-encoder

Models: LightGBM (MIT licence), XLM-RoBERTa-base (MIT, 278M parameters), XLM-RoBERTa-large (MIT, 560M parameters) and mDeBERTa-v3-base (MIT, 279M parameters), all fine-tuned on the training pairs (the large model twice: two epochs). For France, the large model is further fine-tuned on label-free pseudo-labels of the French test candidates (section 12). All are far below the 8B limit. No external data or look-up services are used.

### B. Compute

The pipeline was developed on 192 CPU cores, 1 TB RAM and 4× NVIDIA H200. End-to-end
runtime is about 3 hours; model training (2 × 5 folds) is about 1.5 hours of that.
