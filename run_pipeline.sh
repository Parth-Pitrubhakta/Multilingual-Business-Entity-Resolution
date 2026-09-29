#!/usr/bin/env bash
# End-to-end reproduction: data -> normalisation -> blocking -> matching -> output/
# Usage: bash run_pipeline.sh            (run from anywhere)
# Env overrides: BER_DATA (dataset dir), BER_CACHE (work dir), BER_OUTPUT (output dir), PY (python)
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PY:-python}

$PY translit.py                    # 1. learn native-script -> Latin name dictionary (train pairs)
$PY preprocess.py train test       # 2. normalise names / addresses of all records
$PY blocking.py train 10           # 3a. GPU TF-IDF retrieval (top-10 per view, per record)
$PY blocking.py test 10
$PY stage1.py train                # 3b. learned candidate pruning: retrieval scores + cheap name/address/house-no.
                                   #     similarity; candidates = per record top-3 with p1 >= 0.02
$PY stage1.py test
$PY match.py features train        # 4. pairwise features on candidates
$PY match.py features test
$PY simulate.py 0.19 1             # 4b. orphan simulation: delete 19% of train S1 entities, rebuild candidates/features
$PY simulate.py decoys 0.08 2      # 4c. near-miss decoy injection: clone 8% of true records at a nearby house number
USE_CE=0 MASK_CE=0 TOKSTATS=0 $PY match.py train  # 5a. stack without cross-encoder (its OOF p2 selects hard positives for 5b)
TR=${TR:-torchrun}
$PY cross_encoder.py prep          # 5b. cross-encoder (XLM-R base, MIT): pair tables for two entity halves
# two models per variant, one per entity half, each on 2 GPUs (as in the submitted run)
for m in 0 1; do $TR --nproc_per_node=2 cross_encoder.py train $m; done
for m in 0 1; do for t in train test; do $TR --nproc_per_node=2 cross_encoder.py infer $m $t; done; done
$PY cross_encoder.py merge         #     out-of-fold train scores + averaged test scores  (XLM-R base)
L="CE_MODEL=xlm-roberta-large CE_OUT=ce_large CE_SUFFIX=_large CE_BS=128 CE_LR=1.5e-5"
for m in 0 1; do env $L $TR --nproc_per_node=2 cross_encoder.py train $m; done
for m in 0 1; do for t in train test; do env $L $TR --nproc_per_node=2 cross_encoder.py infer $m $t; done; done
env $L $PY cross_encoder.py merge  #     same for XLM-R large (MIT, 560M params)
CACHE=${BER_CACHE:-$(cd ../../.. && pwd)/cache}  # second epoch of the large model, continued per half
for m in 0 1; do env CE_MODEL=$CACHE/ce_large/model_$m CE_OUT=ce_large2 CE_SUFFIX=_large2 CE_BS=128 CE_LR=8e-6 \
  $TR --nproc_per_node=2 cross_encoder.py train $m; done
for m in 0 1; do for t in train test; do env CE_OUT=ce_large2 CE_SUFFIX=_large2 $TR --nproc_per_node=2 cross_encoder.py infer $m $t; done; done
env CE_OUT=ce_large2 CE_SUFFIX=_large2 $PY cross_encoder.py merge
M="CE_MODEL=microsoft/mdeberta-v3-base CE_OUT=ce_mdeberta CE_SUFFIX=_mdeberta CE_BS=256 CE_LR=3e-5"
for m in 0 1; do env $M $TR --nproc_per_node=2 cross_encoder.py train $m; done   # mDeBERTa-v3-base (MIT, 279M params)
for m in 0 1; do for t in train test; do env $M $TR --nproc_per_node=2 cross_encoder.py infer $m $t; done; done
env $M $PY cross_encoder.py merge
# 5c. two final stacks on the same candidates and features:
#     full    = all features incl. target encodings + XLM-R base/large cross-encoders (v8 configuration)
#     vf      = + label-free token statistics, 30% vocabulary dropout, vocabulary-free path + own threshold
#               for countries absent from training (v10 configuration)
FULL="MODEL_TAG=full_ CE_VARIANTS=ce,cel TOKSTATS=0 MASK_CE=0 VOCAB_FREE_UNSEEN=0"
VF="MODEL_TAG=vf_ CE_VARIANTS=ce,cel,cel2"
#     vfb     = vf configuration, differently seeded, learning rate 0.05, 800/500 rounds (blend member)
#     vfm     = vf configuration + mDeBERTa-v3 cross-encoder (blend member)
VFB="MODEL_TAG=vfb_ CE_VARIANTS=ce,cel,cel2 LGB_SEED=7 FIT_LR=0.05 ROUNDS2=800 ROUNDS3=500"
VFM="MODEL_TAG=vfm_ CE_VARIANTS=ce,cel,cel2,cem"
env $FULL $PY match.py train
env $VF $PY match.py train
env $VFB $PY match.py train
env $VFM $PY match.py train
env $FULL $PY match.py predict test  # 6. score test candidates with every stack
env $VF $PY match.py predict test
env $VFB $PY match.py predict test
env $VFM $PY match.py predict test
OUT=${BER_OUTPUT:-$(cd ../../.. && pwd)/output}
env $FULL BER_OUTPUT=$OUT/full $PY make_submission.py   # 7. per-stack decisions
env $VF BER_OUTPUT=$OUT/vf $PY make_submission.py
# 8. final: full-stack decisions + high-confidence vocabulary-free additions for countries absent from training
$PY combine.py $OUT/full/matching_results.tsv $OUT/vf/matching_results.tsv $OUT/v12 v12
cp $OUT/full/candidate_pairs.tsv $OUT/candidate_pairs.tsv   # identical candidate set for both stacks
# 9. France (absent from training): structural corrections + self-trained French cross-encoder (france_rules.py)
$PY france_rules.py rules $OUT/v12/matching_results.tsv $OUT/v16/matching_results.tsv      # v13 + v15 + v16
for r in 1 2; do                                                                           # two self-training rounds
  [ $r = 1 ] && BASE=$OUT/v16 || BASE=$OUT/v17
  D=ce_fr$([ $r = 2 ] && echo 2); SC=ce_scores_france$([ $r = 2 ] && echo 2).parquet
  CEFR_DIR=$D $PY ce_france.py prep $BASE/matching_results.tsv
  for m in 0 1; do env CE_MODEL=$CACHE/ce_large2/model_$m CE_TABLES=$D CE_OUT=$D CE_BS=128 CE_LR=8e-6 \
    $TR --nproc_per_node=2 cross_encoder.py train $m; done
  for m in 0 1; do env CE_TABLES=$D CE_OUT=$D $TR --nproc_per_node=2 cross_encoder.py infer $m fr; done
  CEFR_DIR=$D CEFR_OUT=$SC $PY ce_france.py merge
  NEXT=$OUT/v1$((6 + r)); $PY france_rules.py selftrain $BASE/matching_results.tsv $CACHE/$SC $NEXT/matching_results.tsv
done
$PY france_rules.py final $OUT/v18/matching_results.tsv $CACHE/ce_scores_france2.parquet $OUT/v20/matching_results.tsv
# full stack re-scored on France with French target encodings + French cross-encoder, gated by the type prior (v21)
env $FULL $PY predict_france.py $OUT/v20/matching_results.tsv $CACHE/ce_scores_france2.parquet $CACHE/test_scores_frfull.parquet
$PY france_rules.py rescored $OUT/v20/matching_results.tsv $CACHE/ce_scores_france2.parquet $CACHE/test_scores_frfull.parquet $OUT/v21/matching_results.tsv
# 9e. v24: tighten France back to the leaderboard-confirmed v17 state (+ strict post-v17 agreement); FINAL_FRANCE=v21 keeps v21 (= v23)
$PY france_rules.py tighten $OUT/v17/matching_results.tsv $OUT/v21/matching_results.tsv $CACHE/ce_scores_france.parquet \
  $CACHE/ce_scores_france2.parquet $OUT/v24/matching_results.tsv
# 10. countries seen in training: blend of four stacks (weights 1:1:3:3, threshold 0.725; robust on all held-out frames) -> final output
$PY ensemble_seen.py $OUT/${FINAL_FRANCE:-v24}/matching_results.tsv $OUT/matching_results.tsv full_:1,vf_:1,vfb_:3,vfm_:3 0.725
