"""Re-score French pairs with the full stack after giving it French vocabulary knowledge.

On France the full stack's target encodings are empty (every French word is unseen) and its
cross-encoder features carry English word meanings. Here, for French pairs only:
  * target-encoding tables are fitted on the training keys plus French pseudo-labelled keys
    (from a submission; confident negatives as in ce_france.py) of the *other* half of the
    French Source-1 entities, so every French pair is encoded out-of-fold;
  * the cross-encoder features (ce, cel) are replaced by the France-adapted cross-encoder's
    out-of-fold logit (ce_france.py).
The saved full-stack fold models (MODEL_TAG=full_) then score the pairs.

usage: MODEL_TAG=full_ CE_VARIANTS=ce,cel TOKSTATS=0 MASK_CE=0 VOCAB_FREE_UNSEEN=0 \
       python predict_france.py <submission.tsv> <ce_scores_france.parquet> <out_scores.parquet>
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import match as M
from common import cache
from ce_france import _pairs


def french_labels(sub_path):
    """1 accepted, 0 confident negative, null otherwise (same rule as ce_france.prep)."""
    v = _pairs(sub_path)
    te = pl.read_parquet(cache("typecal_test.parquet")).filter(pl.col("country") == "France")
    tr = pl.read_parquet(cache("typecal_train.parquet"))
    T = tr.group_by("type").agg(pl.col("y").mean().alias("tr_y"))
    s8 = pl.read_parquet(cache("test_scoresfull_.parquet"), columns=["rec_id", "s1_id", "p3"]).rename({"p3": "p8"})
    s10 = pl.read_parquet(cache("test_scoresvf_.parquet"), columns=["rec_id", "s1_id", "p3"]).rename({"p3": "p10"})
    te = te.join(T, on="type", how="left").join(s8, on=["rec_id", "s1_id"], how="left").join(s10, on=["rec_id", "s1_id"], how="left")
    te = te.join(v.with_columns(pl.lit(1).alias("acc")), on=["s1_id", "rec_id"], how="left").with_columns(pl.col("acc").fill_null(0))
    te = te.join(v.select("rec_id", pl.lit(1).alias("used")), on="rec_id", how="left").with_columns(pl.col("used").fill_null(0))
    neg = ((pl.col("used") == 1) | pl.col("nm").str.contains("desc")
           | ((pl.col("ctok") == "+C") & (pl.col("addr") != "same-addr"))
           | ((pl.col("addr") == "near") & pl.col("legal").is_in(["diff", "one"]))
           | ((pl.max_horizontal(pl.col("p8").fill_null(0), pl.col("p10").fill_null(0)) < 0.02) & (pl.col("tr_y").fill_null(0) < 0.5)))
    te = te.with_columns(pl.when(pl.col("acc") == 1).then(1).when(neg).then(0).otherwise(None).alias("y"),
                         (pl.col("s1_id").hash(seed=77) % 2).alias("half"))
    return te.select("rec_id", "s1_id", "y", "half")


def main(sub_path, ce_path, out_path):
    cols = json.load(open(cache(f"{M.MODEL_TAG}cols.json")))
    lab = french_labels(sub_path)
    f = pl.read_parquet(cache("test_feats.parquet")).join(lab.select("rec_id", "s1_id", "half"), on=["rec_id", "s1_id"])
    normk = pl.read_parquet(cache("test_norm.parquet"), columns=M.NC)
    fit_train = pl.read_parquet(cache("train_te_keys.parquet"))
    fr_keys = M.te_keys(lab.select("rec_id", "s1_id"), normk).with_columns(lab["y"], lab["half"]).filter(pl.col("y").is_not_null())
    parts = []
    for h in (0, 1):
        fh = f.filter(pl.col("half") == h)
        fit = pl.concat([fit_train.select(fr_keys.drop("half").columns), fr_keys.filter(pl.col("half") != h).drop("half")],
                        how="vertical_relaxed").with_columns(pl.col("y").cast(fit_train["y"].dtype))
        parts.append(M.te_features(fh.drop("half"), normk, fit=fit))
        print(f"half {h}: {fh.height} pairs encoded with {fit.height} keys", flush=True)
    f = pl.concat(parts, how="vertical_relaxed")
    ce = pl.read_parquet(ce_path).select("rec_id", "s1_id", "cefr")
    f = f.join(ce, on=["rec_id", "s1_id"], how="left")
    for var in M.CE_VARIANTS:
        f = f.with_columns(pl.col("cefr").alias(var))
        f = f.with_columns(
            (pl.col(var) - pl.col(var).max().over("rec_id")).alias(f"{var}_gap"),
            pl.col(var).rank("ordinal", descending=True).over("rec_id").cast(pl.Int32).alias(f"{var}_rank"),
            (pl.col(var) - pl.col(var).filter(pl.col(var) < pl.col(var).max()).max().over("rec_id").fill_null(-20)).alias(f"{var}_margin"))
    X = f.select(cols["cols2"]).to_numpy().astype(np.float32)
    p2 = np.mean([lgb.Booster(model_file=cache(f"{M.MODEL_TAG}stage2_fold{k}.lgb")).predict(X, num_threads=32) for k in range(M.NFOLD)], axis=0)
    f = f.with_columns(pl.Series("p2", p2.astype(np.float32)))
    norm = pl.read_parquet(cache("test_norm.parquet"), columns=["entity_id", "src", "country", "n_core", "a_tok"])
    f = M.twin_features(M.group_features(f, "p2", norm), "p2", norm)
    X = f.select(cols["cols3"]).to_numpy().astype(np.float32)
    p3 = np.mean([lgb.Booster(model_file=cache(f"{M.MODEL_TAG}stage3_fold{k}.lgb")).predict(X, num_threads=32) for k in range(M.NFOLD)], axis=0)
    f = f.with_columns(pl.Series("p3", p3.astype(np.float32)))
    f.select("rec_id", "s1_id", "p2", "p3").write_parquet(out_path)
    print("French pairs re-scored:", f.height, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
