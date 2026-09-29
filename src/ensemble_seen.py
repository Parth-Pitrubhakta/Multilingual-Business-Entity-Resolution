"""Final step: decisions for countries seen in training from a weighted blend of stacks.

Each stack is a MODEL_TAG whose test scores are in test_scores{tag}.parquet (column p3).
Weights and threshold were chosen on the training out-of-fold scores (US/India macro F0.5); the final
blend is full_:1, vf_:1, vfb_:3, vfm_:3 at 0.725 (Documentation_template.md, section 13).
Pairs of countries absent from training (France) are kept exactly as in the input submission.

usage: python ensemble_seen.py <submission.tsv> <out.tsv> <tag:weight,...> <thr>
  e.g. python ensemble_seen.py in.tsv out.tsv full_:0.3,vf_:0.7 0.725
"""
import os
import sys

import polars as pl

from common import cache, read_source
from france_rules import read_pairs, write_pairs


SCORES = os.environ.get("SCORES", "test_scores")  # "w_test_scores" = widened candidate set (widen.py)


def main(sub_path, out_path, spec="full_:0.3,vf_:0.7", thr=0.725):
    weights = {t: float(w) for t, w in (x.split(":") for x in spec.split(","))}
    seen = set(read_source("train", 1)["country"].unique().to_list())
    s1 = read_source("test", 1).select(pl.col("entity_id").alias("s1_id"), "country")
    f = None
    for tag in weights:
        x = pl.read_parquet(cache(f"{SCORES}{tag}.parquet"), columns=["rec_id", "s1_id", "p3"]).rename({"p3": tag})
        f = x if f is None else f.join(x, on=["rec_id", "s1_id"])
    f = f.join(s1, on="s1_id").filter(pl.col("country").is_in(list(seen)))
    f = f.with_columns((sum(w * pl.col(t) for t, w in weights.items()) / sum(weights.values())).alias("pe"))
    best = f.filter(pl.col("pe") == pl.col("pe").max().over("rec_id")).sort(["rec_id", "s1_id"]).unique("rec_id", keep="first", maintain_order=True)
    seen_dec = best.filter(pl.col("pe") >= thr).select("s1_id", "rec_id")
    base = read_pairs(sub_path).join(s1, on="s1_id")
    unseen = base.filter(~pl.col("country").is_in(list(seen))).select("s1_id", "rec_id")
    old = base.filter(pl.col("country").is_in(list(seen))).select("s1_id", "rec_id")
    out = pl.concat([unseen, seen_dec])
    print(f"blend {weights} thr {thr}: seen countries {old.height} -> {seen_dec.height} pairs "
          f"(+{seen_dec.join(old, on=['s1_id', 'rec_id'], how='anti').height} / -{old.join(seen_dec, on=['s1_id', 'rec_id'], how='anti').height}); "
          f"unseen kept {unseen.height}; total {out.height}", flush=True)
    write_pairs(out, out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "full_:0.3,vf_:0.7",
         float(sys.argv[4]) if len(sys.argv) > 4 else 0.725)
