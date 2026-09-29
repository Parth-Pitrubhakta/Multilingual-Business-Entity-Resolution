"""Write output/candidate_pairs.tsv and output/matching_results.tsv.

candidate_pairs.tsv   = every pair the matching model scored (stage-1 survivors)
matching_results.tsv  = pairs accepted by the final decision rule
Both contain exactly one row per test Source 1 entity (empty list allowed).
"""
import os
import sys

import polars as pl

from common import OUTPUT_DIR, cache, read_source


def _write(path, header, s1_ids, pairs: pl.DataFrame):
    groups = {}
    for s, r in pairs.select("s1_id", "rec_id").iter_rows():
        groups.setdefault(s, []).append(r)
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{header}\n")
        for s in s1_ids:
            ids = groups.get(s, [])
            ids = list(dict.fromkeys(ids))  # de-duplicate, keep order
            f.write(f"{s}\t{','.join(ids)}\n")


def main(split="test", thr=None):
    from match import decide

    tag = os.environ.get("MODEL_TAG", "")
    scores = pl.read_parquet(cache(f"{split}_scores{tag}.parquet"))
    if thr is None:
        thr = float(open(cache(f"{tag}thr.txt")).read())
    s1_ids = read_source(split, 1)["entity_id"].to_list()
    cand = scores.sort(["s1_id", "p3"], descending=[False, True])
    # countries absent from training use the vocabulary-free path and its own threshold
    # the full (v8-configuration) stack has no vocabulary-free path and uses one threshold for every country
    vocab_free = os.environ.get("VOCAB_FREE_UNSEEN", "1") == "1"
    thr_u = float(open(cache(f"{tag}thr_unseen.txt")).read()) if vocab_free and os.path.exists(cache(f"{tag}thr_unseen.txt")) else thr
    seen = set(read_source("train", 1)["country"].unique().to_list())
    s1c = read_source(split, 1).select(pl.col("entity_id").alias("s1_id"), "country")
    sc = scores.join(s1c, on="s1_id", how="left").with_columns(
        pl.when(pl.col("country").is_in(list(seen))).then(pl.lit(thr)).otherwise(pl.lit(thr_u)).alias("_t"))
    best = sc.filter(pl.col("p3") == pl.col("p3").max().over("rec_id")).unique("rec_id", keep="first")
    matches = best.filter(pl.col("p3") >= pl.col("_t")).select("s1_id", "rec_id", "p3")
    matches = matches.sort(["s1_id", "p3"], descending=[False, True])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _write(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), "candidate_entity_ids", s1_ids, cand)
    _write(os.path.join(OUTPUT_DIR, "matching_results.tsv"), "matched_entity_ids", s1_ids, matches)
    n_s1 = len(s1_ids)
    print(f"thr={thr} (unseen countries {thr_u})  S1={n_s1}  candidates={cand.height} ({cand.height / n_s1:.3f}/S1)  "
          f"matches={matches.height} ({matches.height / n_s1:.3f}/S1)  "
          f"empty={n_s1 - matches['s1_id'].n_unique()}")


if __name__ == "__main__":
    main(thr=float(sys.argv[1]) if len(sys.argv) > 1 else None)
