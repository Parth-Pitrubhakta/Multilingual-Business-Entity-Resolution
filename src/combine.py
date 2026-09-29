"""Combine a full-path decision set with vocabulary-free additions for countries absent from training.

Countries unseen in training (France) are scored twice: by the full model (target encodings +
cross-encoders; e.g. v8) and by the vocabulary-free path (v10). The vocabulary-free path
recovers true matches that the full model rejects because vocabulary learned on other
languages transfers wrongly ("et Fils" treated like the English decoy suffix "& Sons"), but
it also admits some decoys. We keep the full-path decisions and add a vocabulary-free
match only when it is one of the change types that are ~99% true matches in the training
data: same address (identical house number, address token-set >= 95) and either
  * the record only DROPS name words relative to the S1 name, or
  * it adds / swaps in ONE word whose label-free same-house-number rate is >= 0.7
    (noise words; decoy words sit at a different house number).
Each record stays matched to at most one S1.

Ruleset "v12" additionally (both validated structurally on training data):
  * adds vocabulary-free matches NOT at the same address when the core name is the same
    (or name similarity >= 90) and the house number is identical but the rest of the
    address differs, missing on one side (no legal-form conflict), or has a dropped digit
    (training true-match rates 93-99.7%), record address not empty;
  * drops full-path matches at a neighbouring / one-digit-different house number that the
    vocabulary-free path rejects (the look-alike decoy pattern; 33% true in training).

usage: python combine.py <full_matching.tsv> <vocabfree_matching.tsv> <out_dir> [v11|v12]
"""
import os
import sys

import polars as pl

from common import cache, read_source


def _pairs(path: str) -> pl.DataFrame:
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("rec_id")))


def word_rates(split: str = "test", min_n: int = 20) -> pl.DataFrame:
    """Per (country, word): share of candidate pairs adding that word whose house numbers are identical."""
    f = pl.read_parquet(cache(f"{split}_feats.parquet"), columns=["rec_id", "s1_id", "hnr_eq"])
    n = pl.read_parquet(cache(f"{split}_norm.parquet"), columns=["entity_id", "n_core", "country"])
    t = n.select("entity_id", "country", pl.col("n_core").str.split(" ").alias("t"))
    f = f.join(t.rename({"entity_id": "s1_id", "t": "t1"}), on="s1_id").join(
        t.select(pl.col("entity_id").alias("rec_id"), pl.col("t").alias("t2")), on="rec_id")
    e = (f.with_columns(pl.col("t2").list.set_difference(pl.col("t1")).alias("w"))
         .select("country", "w", "hnr_eq").explode("w").filter(pl.col("w").is_not_null() & (pl.col("hnr_eq") >= 0)))
    return (e.group_by(["country", "w"]).agg(pl.len().alias("n"), (pl.col("hnr_eq") == 1).mean().alias("rate"))
            .filter(pl.col("n") >= min_n))


def combine(full_path: str, free_path: str, out_dir: str, ruleset: str = "v11", split: str = "test", min_rate: float = 0.7):
    seen = set(read_source("train", 1)["country"].unique().to_list())
    s1 = read_source(split, 1).select(pl.col("entity_id").alias("s1_id"), "country")
    full, free = _pairs(full_path), _pairs(free_path)
    add = free.join(full, on=["s1_id", "rec_id"], how="anti").join(s1, on="s1_id")
    add = add.filter(~pl.col("country").is_in(list(seen)))
    add = add.join(full.select("rec_id"), on="rec_id", how="anti")  # record already matched elsewhere
    ft = pl.read_parquet(cache(f"{split}_feats.parquet"), columns=["rec_id", "s1_id", "hnr_eq", "ad_tset", "hn_rel",
                                                                "nm_tsort", "nt_jac", "ad_empty2", "lg_conflict"])
    nm = pl.read_parquet(cache(f"{split}_norm.parquet"), columns=["entity_id", "n_core"])
    t = nm.select("entity_id", pl.col("n_core").str.split(" ").alias("t"))
    add = add.join(ft, on=["rec_id", "s1_id"], how="left").join(t.rename({"entity_id": "s1_id", "t": "t1"}), on="s1_id") \
        .join(t.rename({"entity_id": "rec_id", "t": "t2"}), on="rec_id")
    add = add.with_columns(pl.col("t2").list.set_difference(pl.col("t1")).alias("ex"))
    add = add.with_columns(pl.col("ex").list.first().alias("w")).join(word_rates(split), on=["country", "w"], how="left")
    same = (pl.col("hnr_eq") == 1) & (pl.col("ad_tset") >= 95)
    keep = same & ((pl.col("ex").list.len() == 0) | ((pl.col("ex").list.len() == 1) & (pl.col("rate") >= min_rate)))
    if ruleset == "v12":
        samename = (pl.col("nt_jac") == 1) | (pl.col("nm_tsort") >= 90)
        noaddr = pl.col("ad_empty2") == 1
        keep = keep | (~same & ~noaddr & (
            ((pl.col("hnr_eq") == 1) & samename)
            | ((pl.col("hn_rel") == -1) & samename & (pl.col("lg_conflict") != 1))
            | (pl.col("hn_rel").is_in([1, 2, 3]) & (pl.col("nt_jac") == 1))))
    add = add.filter(keep).select("s1_id", "rec_id")
    base = full
    if ruleset == "v12":
        drop = full.join(free, on=["s1_id", "rec_id"], how="anti").join(s1, on="s1_id")
        drop = drop.filter(~pl.col("country").is_in(list(seen))).join(ft.select("rec_id", "s1_id", "hn_rel"),
                                                                      on=["rec_id", "s1_id"], how="left")
        drop = drop.filter(pl.col("hn_rel").is_in([4, 5])).select("s1_id", "rec_id")
        base = full.join(drop, on=["s1_id", "rec_id"], how="anti")
        print(f"dropped {full.height - base.height} near-house-number full-path matches")
    final = pl.concat([base, add]).unique(["s1_id", "rec_id"])
    assert final.group_by("rec_id").len()["len"].max() == 1
    os.makedirs(out_dir, exist_ok=True)
    ids = read_source(split, 1)["entity_id"].to_list()
    groups = {}
    for s, r in final.iter_rows():
        groups.setdefault(s, []).append(r)
    with open(os.path.join(out_dir, "matching_results.tsv"), "w") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s in ids:
            fh.write(f"{s}\t{','.join(groups.get(s, []))}\n")
    print(f"full-path pairs {full.height}; vocabulary-free additions kept {add.height}; final {final.height}")


if __name__ == "__main__":
    combine(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "v11")
