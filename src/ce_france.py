"""Self-training of the cross-encoder on France (a country absent from training).

The US/India cross-encoders carry English word semantics into French ("groupe" reads as the
decoy word "group", "centre" as the noise word "center"). Here the XLM-R large checkpoint is
fine-tuned on French pseudo-labels, cross-fitted by two halves of the French Source-1 entities
(each model scores only the other half), so every French candidate pair gets an out-of-fold score.

Pseudo-labels (label-free, from the current decisions and structural decoy types):
  1  accepted pair of the current submission
  0  rejected pair whose record is matched elsewhere, a descriptor-word swap/add, a country-token
     addition away from the same address, a legal-form change at a near house number, or a pair
     both stacks reject (max score < 0.02) in a type whose training truth is < 0.5
  -  everything else (ambiguous pools) is left out of training

usage: python ce_france.py prep <matching_results.tsv>   -> cache/ce_fr/{train_0,train_1,infer_fr}.parquet
       torchrun ... cross_encoder.py train|infer (CE_TABLES=ce_fr, CE_OUT=ce_fr)
       python ce_france.py merge                         -> cache/ce_scores_france.parquet
"""
import glob
import os
import sys

import polars as pl

from common import cache, read_ground_truth, read_source

CE_DIR = cache(os.environ.get("CEFR_DIR", "ce_fr"))


def _pairs(path):
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "")
            .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("rec_id")))


def _texts(norm):
    return norm.select("entity_id", pl.concat_str([pl.col("business_name").fill_null(""), pl.lit(" | "),
                                                   pl.col("business_address").fill_null("")]).alias("t"))


def prep(sub_path, n_train_keep=150_000, seed=0):
    os.makedirs(CE_DIR, exist_ok=True)
    v = _pairs(sub_path)
    te = pl.read_parquet(cache("typecal_test.parquet")).filter(pl.col("country") == "France")
    tr = pl.read_parquet(cache("typecal_train.parquet"))
    T = tr.group_by("type").agg(pl.col("y").mean().alias("tr_y"))
    from france_rules import french_scores
    sc = french_scores().select("rec_id", "s1_id", "p8", "p10")  # both stacks' scores on French candidates
    te = te.join(T, on="type", how="left").join(sc, on=["rec_id", "s1_id"], how="left")
    te = te.join(v.with_columns(pl.lit(1).alias("acc")), on=["s1_id", "rec_id"], how="left").with_columns(pl.col("acc").fill_null(0))
    te = te.join(v.select("rec_id", pl.lit(1).alias("used")), on="rec_id", how="left").with_columns(pl.col("used").fill_null(0))
    neg = ((pl.col("used") == 1) | pl.col("nm").str.contains("desc")
           | ((pl.col("ctok") == "+C") & (pl.col("addr") != "same-addr"))
           | ((pl.col("addr") == "near") & pl.col("legal").is_in(["diff", "one"]))
           | ((pl.max_horizontal(pl.col("p8").fill_null(0), pl.col("p10").fill_null(0)) < 0.02) & (pl.col("tr_y").fill_null(0) < 0.5)))
    te = te.with_columns(pl.when(pl.col("acc") == 1).then(1).when(neg).then(0).otherwise(None).alias("y"),
                         (pl.col("s1_id").hash(seed=77) % 2).alias("half"))
    cols = ["entity_id", "business_name", "business_address"]
    tx = _texts(pl.read_parquet(cache("test_norm.parquet"), columns=cols))
    te = te.join(tx.rename({"entity_id": "s1_id", "t": "a"}), on="s1_id").join(tx.rename({"entity_id": "rec_id", "t": "b"}), on="rec_id")
    te.select("rec_id", "s1_id", "half", "a", "b").write_parquet(os.path.join(CE_DIR, "infer_fr.parquet"))
    lab = te.filter(pl.col("y").is_not_null())
    print(f"French pairs {te.height}: positives {lab['y'].sum()}, negatives {(lab['y'] == 0).sum()}, unlabeled {te.height - lab.height}", flush=True)
    for m in (0, 1):
        h = lab.filter(pl.col("half") == m)
        pos, ng = h.filter(pl.col("y") == 1), h.filter(pl.col("y") == 0)
        hard = pos.filter((pl.min_horizontal(pl.col("p8").fill_null(0), pl.col("p10").fill_null(0)) < 0.95))
        easy = pos.join(hard.select("rec_id", "s1_id"), on=["rec_id", "s1_id"], how="anti")
        easy = easy.sample(min(easy.height, max(0, int(1.5 * ng.height) - hard.height)), seed=seed + m)
        orig = pl.read_parquet(cache(f"ce/train_{m}.parquet"))
        orig = orig.sample(min(n_train_keep, orig.height), seed=seed + m)
        t = pl.concat([ng.select("a", "b", "y"), hard.select("a", "b", "y"), easy.select("a", "b", "y"),
                       orig.select("a", "b", pl.col("y").cast(pl.Int32))], how="vertical_relaxed")
        t = t.with_columns(pl.col("y").cast(pl.Int32)).sample(fraction=1.0, shuffle=True, seed=seed + m)
        t.write_parquet(os.path.join(CE_DIR, f"train_{m}.parquet"))
        print(f"half {m}: {ng.height} neg, {hard.height} hard + {easy.height} easy pos, {orig.height} original -> {t.height}", flush=True)


def merge(out_dir=os.environ.get("CEFR_DIR", "ce_fr")):
    d = cache(out_dir)
    parts = [pl.read_parquet(p).rename({f"ce{m}": "cefr"}) for m in (0, 1)
             for p in sorted(glob.glob(os.path.join(d, f"scores_fr_m{m}_r*.parquet")))]
    s = pl.concat(parts)
    s.write_parquet(cache(os.environ.get("CEFR_OUT", "ce_scores_france.parquet")))
    print("French out-of-fold scores", s.shape, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(sys.argv[2])
    elif sys.argv[1] == "merge":
        merge()
