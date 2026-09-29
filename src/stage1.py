"""Stage 2 of candidate generation: learned pruning of retrieved pairs.

The GPU retrieval returns up to 30 (record, S1) pairs per record. A light
LightGBM model using only retrieval scores and cheap context features
(how this pair compares with the record's other retrieved S1 entities and
with the other records that retrieved the same S1) scores every pair, and we
keep pairs above a low probability threshold (plus at most `max_per_rec`
per record). The surviving pairs are the final candidate set that the
matching model runs inference over (= candidate_pairs.tsv).
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import cache, read_ground_truth

S1_FEATS = ["cos_n", "cos_a", "cos_c", "rk_c", "rk_n", "rk_a", "rec_ncand", "rec_max_c", "gap_c", "gap2_c",
            "rec_max_n", "gap_n", "rec_max_a", "gap_a", "rk_rec_c", "s1_nrec", "s1_top1", "s1_rank_c",
            "s1_max_c", "s1_gap_c", "sec_n", "sec_a", "marg_n", "marg_a",
            "q_nm_ts", "q_ad_ts", "q_hn_eq", "q_hn_diff", "q_nm_gap", "q_ad_gap", "src", "a_empty", "nonlatin", "is_web", "d_a_empty", "n_len", "a_len"]


def fold_of(col: str = "s1_id", k: int = 5):
    return pl.col(col).hash(seed=42) % k


def build(split: str) -> pl.DataFrame:
    r = pl.read_parquet(cache(f"{split}_retrieval.parquet"))
    return build_frame(r, pl.read_parquet(cache(f"{split}_norm.parquet"), columns=NORM1))


NORM1 = ["entity_id", "src", "a_tok", "n_nonlatin", "n_isweb", "n_core", "a_comps"]


def _quick_similarity(r: pl.DataFrame, norm: pl.DataFrame, chunk: int = 20_000_000) -> pl.DataFrame:
    """Cheap similarity signals for the pruner (v6): name token-sort ratio,
    address token-set ratio and house-number agreement, per retrieved pair.
    Linear in the number of retrieved pairs, computed in chunks."""
    from rapidfuzz import fuzz, process
    side = norm.select("entity_id", "n_core", "a_tok",
                       pl.col("a_comps").str.extract(r"(\d+)").cast(pl.Int64, strict=False).alias("hn"))
    out = []
    for s in range(0, r.height, chunk):
        c = r.slice(s, chunk).select("rec_id", "s1_id")
        c = c.join(side.rename({"entity_id": "rec_id", "n_core": "n2", "a_tok": "a2", "hn": "h2"}), on="rec_id",
                   how="left", maintain_order="left")
        c = c.join(side.rename({"entity_id": "s1_id", "n_core": "n1", "a_tok": "a1", "hn": "h1"}), on="s1_id",
                   how="left", maintain_order="left")
        nm = process.cpdist(c["n1"].fill_null("").to_list(), c["n2"].fill_null("").to_list(),
                            scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32)
        ad = process.cpdist(c["a1"].fill_null("").to_list(), c["a2"].fill_null("").to_list(),
                            scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        c = c.select(
            pl.Series("q_nm_ts", nm), pl.Series("q_ad_ts", ad),
            pl.when(pl.col("h1").is_null() | pl.col("h2").is_null()).then(-1)
            .otherwise((pl.col("h1") == pl.col("h2")).cast(pl.Int32)).cast(pl.Int8).alias("q_hn_eq"),
            pl.when(pl.col("h1").is_null() | pl.col("h2").is_null()).then(-1.0)
            .otherwise((pl.col("h1") - pl.col("h2")).abs().clip(0, 1_000_000).cast(pl.Float32).log1p())
            .cast(pl.Float32).alias("q_hn_diff"))
        out.append(c)
    return pl.concat([r, pl.concat(out)], how="horizontal")


def build_frame(r: pl.DataFrame, norm: pl.DataFrame) -> pl.DataFrame:
    """Stage-1 features from a retrieval frame (one row per retrieved (rec, S1) pair)."""
    r = _quick_similarity(r, norm)
    flags = norm.select(
        pl.col("entity_id"), pl.col("src"),
        (pl.col("a_tok") == "").cast(pl.Int8).alias("a_empty"),
        pl.col("n_nonlatin").cast(pl.Int8).alias("nonlatin"),
        pl.col("n_isweb").cast(pl.Int8).alias("is_web"),
        pl.col("n_core").str.len_chars().alias("n_len"),
        pl.col("a_tok").str.len_chars().alias("a_len"),
    )
    r = r.join(flags.rename({"entity_id": "rec_id"}), on="rec_id", how="left")
    r = r.join(flags.select(pl.col("entity_id").alias("s1_id"), pl.col("a_empty").alias("d_a_empty")),
               on="s1_id", how="left")
    r = r.with_columns((pl.col("cos_n") + pl.col("cos_a")).alias("cos_c"))
    r = r.with_columns(
        pl.len().over("rec_id").alias("rec_ncand"),
        pl.col("cos_c").max().over("rec_id").alias("rec_max_c"),
        pl.col("cos_n").max().over("rec_id").alias("rec_max_n"),
        pl.col("cos_a").max().over("rec_id").alias("rec_max_a"),
        pl.col("cos_c").rank("ordinal", descending=True).over("rec_id").alias("rk_rec_c"),
    )
    second = r.filter(pl.col("rk_rec_c") == 2).select("rec_id", pl.col("cos_c").alias("sec_c"))
    r = r.join(second, on="rec_id", how="left").with_columns(pl.col("sec_c").fill_null(0))
    # runner-up similarity per view: how unique is the best name / address match?
    r = r.with_columns(
        pl.col("cos_n").rank("ordinal", descending=True).over("rec_id").alias("_rn"),
        pl.col("cos_a").rank("ordinal", descending=True).over("rec_id").alias("_ra"),
    )
    r = r.join(r.filter(pl.col("_rn") == 2).select("rec_id", pl.col("cos_n").alias("sec_n")), on="rec_id", how="left")
    r = r.join(r.filter(pl.col("_ra") == 2).select("rec_id", pl.col("cos_a").alias("sec_a")), on="rec_id", how="left")
    r = r.with_columns(pl.col("sec_n").fill_null(0), pl.col("sec_a").fill_null(0)).with_columns(
        # this pair's similarity minus the best *other* S1's similarity, per view
        pl.when(pl.col("_rn") == 1).then(pl.col("cos_n") - pl.col("sec_n"))
        .otherwise(pl.col("cos_n") - pl.col("rec_max_n")).alias("marg_n"),
        pl.when(pl.col("_ra") == 1).then(pl.col("cos_a") - pl.col("sec_a"))
        .otherwise(pl.col("cos_a") - pl.col("rec_max_a")).alias("marg_a"),
    ).drop("_rn", "_ra")
    r = r.with_columns(
        (pl.col("q_nm_ts") - pl.col("q_nm_ts").max().over("rec_id")).alias("q_nm_gap"),
        (pl.col("q_ad_ts") - pl.col("q_ad_ts").max().over("rec_id")).alias("q_ad_gap"),
    )
    r = r.with_columns(
        (pl.col("cos_c") - pl.col("rec_max_c")).alias("gap_c"),
        # for the top pair: margin over the runner-up; for others: distance to the top
        pl.when(pl.col("rk_rec_c") == 1).then(pl.col("cos_c") - pl.col("sec_c"))
        .otherwise(pl.col("cos_c") - pl.col("rec_max_c")).alias("gap2_c"),
        (pl.col("cos_n") - pl.col("rec_max_n")).alias("gap_n"),
        (pl.col("cos_a") - pl.col("rec_max_a")).alias("gap_a"),
    ).drop("sec_c")
    r = r.with_columns(
        pl.len().over("s1_id").alias("s1_nrec"),
        (pl.col("rk_rec_c") == 1).sum().over("s1_id").alias("s1_top1"),
        pl.col("cos_c").rank("ordinal", descending=True).over("s1_id").alias("s1_rank_c"),
        pl.col("cos_c").max().over("s1_id").alias("s1_max_c"),
    ).with_columns((pl.col("cos_c") - pl.col("s1_max_c")).alias("s1_gap_c"))
    return r


def train(max_rows: int = 60_000_000):
    r = build("train")
    gt = read_ground_truth().with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    r = r.join(gt, on=["rec_id", "s1_id"], how="left").with_columns(pl.col("y").fill_null(0))
    r = r.with_columns(fold_of().alias("fold"))
    tr = r.filter(pl.col("fold") != 0)
    if tr.height > max_rows:
        tr = tr.sample(max_rows, seed=0)
    ds = lgb.Dataset(tr.select(S1_FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy())
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=160, verbose=-1)
    model = lgb.train(params, ds, num_boost_round=400)
    model.save_model(cache("stage1.lgb"))
    r = r.with_columns(pl.Series("p1", model.predict(r.select(S1_FEATS).to_numpy().astype(np.float32),
                                                     num_threads=160)).cast(pl.Float32))
    r.select("rec_id", "s1_id", "p1", "y", "fold", *S1_FEATS).write_parquet(cache("train_stage1.parquet"))
    report(r, gt)


def report(r: pl.DataFrame, gt: pl.DataFrame):
    """Recall / size trade-off on the held-out fold, with and without a per-record cap."""
    v = r.filter(pl.col("fold") == 0).with_columns(pl.col("p1").rank("ordinal", descending=True).over("rec_id").alias("_k"))
    gtv = gt.with_columns(fold_of().alias("fold")).filter(pl.col("fold") == 0)
    n_s1 = gtv["s1_id"].n_unique() / 0.944  # all fold-0 S1 (incl. singletons), approx
    tot = gtv.height
    print(f"val true pairs {tot}, retrieved true {v['y'].sum()} ({v['y'].sum() / tot:.5f})")
    for k in (1, 2, 3, 99):
        for thr in [0.003, 0.01, 0.02, 0.03, 0.05, 0.1]:
            kk = v.filter((pl.col("p1") >= thr) & (pl.col("_k") <= k))
            print(f"top{k} thr {thr}: recall {kk['y'].sum() / tot:.5f}  pairs {kk.height}  per-true {kk.height / tot:.3f}")


def predict(split: str) -> pl.DataFrame:
    r = build(split)
    model = lgb.Booster(model_file=cache("stage1.lgb"))
    r = r.with_columns(pl.Series("p1", model.predict(r.select(S1_FEATS).to_numpy().astype(np.float32),
                                                     num_threads=160)).cast(pl.Float32))
    r.select("rec_id", "s1_id", "p1", *S1_FEATS).write_parquet(cache(f"{split}_stage1.parquet"))
    return r


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        predict(sys.argv[1])
