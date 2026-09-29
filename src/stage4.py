"""Stage 4 (V2): re-decide the ambiguous band with the LLM pair score (llm_ce.py) on top of the four stacks.

Train: LightGBM, 5-fold by S1 fold (hash(s1_id, 42) % 5), on the band pairs of the widened training frame; every
input score is itself out-of-fold. Pairs outside the band keep their stack blend. Decision rule unchanged:
each record goes to its best S1 if the (re-scored) probability >= thr.

  python stage4.py eval            -> out-of-fold macro F0.5 vs the v25 rule, threshold grid
  python stage4.py predict <thr> <v24_matching.tsv> <out.tsv>   -> final US/India decisions (France kept)
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import cache, read_ground_truth, read_source
from llm_ce import blend
from stage1 import fold_of

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=96, seed=4,
              deterministic=True, force_row_wise=True)
ROUNDS = int(os.environ.get("S4_ROUNDS", "400"))
STACKS = ["full_", "vf_", "vfb_", "vfm_"]


LLMS = [x for x in os.environ.get("S4_LLM", "llm").split(",") if x]   # LLM score sets (cache/<name>/llm_scores_<split>)
RAW = os.environ.get("S4_RAW", "1") == "1"                              # + every pairwise feature and the 4 cross-encoder logits
CE_SFX = {"ce": "", "cel": "_large", "cel2": "_large2", "cem": "_mdeberta"}
RAW_COLS = []


def frame(split):
    global RAW_COLS
    b = pl.read_parquet(os.path.join(cache("llm"), f"{split}_band.parquet")).drop(["a", "b"])
    for name in LLMS:
        llm = pl.read_parquet(os.path.join(cache(name), f"llm_scores_{split}.parquet"), columns=["rec_id", "s1_id", "llm"]).rename({"llm": name})
        b = b.join(llm, on=["rec_id", "s1_id"], how="left")
        b = b.with_columns(
            pl.col(name).rank("ordinal", descending=True).over("rec_id").alias(f"{name}_rank"),
            (pl.col(name) - pl.col(name).max().over("rec_id")).alias(f"{name}_gap"),
            (pl.col(name) - pl.col(name).filter(pl.col(name) < pl.col(name).max()).max().over("rec_id").fill_null(-20)).alias(f"{name}_margin"),
            pl.col(name).rank("ordinal", descending=True).over("s1_id").alias(f"{name}_s1_rank"))
    if RAW:
        pf = pl.read_parquet(cache(f"w_{split}_feats.parquet"))
        RAW_COLS = [c for c in pf.columns if c not in ("rec_id", "s1_id", "y", "fold", "src")]
        b = b.join(pf.select(["rec_id", "s1_id"] + RAW_COLS), on=["rec_id", "s1_id"], how="left")
        for v, sfx in CE_SFX.items():
            ce = pl.concat([pl.read_parquet(cache(f"ce_scores_{split}{sfx}.parquet")).select("rec_id", "s1_id", "ce"),
                            pl.read_parquet(cache(f"w_ce_{split}_{v}.parquet")).select("rec_id", "s1_id", "ce")]).unique(["rec_id", "s1_id"], keep="first")
            b = b.join(ce.rename({"ce": f"x_{v}"}), on=["rec_id", "s1_id"], how="left")
        RAW_COLS = RAW_COLS + [f"x_{v}" for v in CE_SFX]
    full = blend(split)  # S1-level context from every candidate pair of the S1 (not only band pairs)
    s1ctx = full.group_by("s1_id").agg((pl.col("pe") >= 0.725).sum().alias("s1_nconf"), pl.col("pe").sum().alias("s1_psum"),
                                       pl.len().alias("s1_ncand"))
    b = b.join(s1ctx, on="s1_id", how="left")
    c = read_source(split, 1).select(pl.col("entity_id").alias("s1_id"), (pl.col("country") == "US").cast(pl.Int8).alias("is_us"))
    b = b.join(c, on="s1_id", how="left")
    b = b.with_columns(
        pl.len().over("rec_id").alias("r_n"),
        pl.col("pe").rank("ordinal", descending=True).over("rec_id").alias("pe_rank"),
        (pl.col("pe") - pl.col("pe").filter(pl.col("pe") < pl.col("pe").max()).max().over("rec_id").fill_null(0)).alias("pe_margin"),
        (pl.col("pe").max().over("rec_id") - pl.col("pe")).alias("pe_gap"),
        (pl.col("s1_psum") - pl.col("pe")).alias("s1_psum_other"),
    )
    return b


def feats():
    return (STACKS + ["pe", "s1_nconf", "s1_psum", "s1_ncand", "is_us", "r_n", "pe_rank", "pe_margin", "pe_gap", "s1_psum_other"]
            + [f"{n}{s}" for n in LLMS for s in ("", "_rank", "_gap", "_margin", "_s1_rank")] + RAW_COLS)


def _macro_setup():
    s1 = read_source("train", 1).select(pl.col("entity_id").alias("s1_id"))
    gt = read_ground_truth().with_columns(pl.lit(1).alias("tt"))
    base = s1.join(gt.group_by("s1_id").len().rename({"len": "nt"}), on="s1_id", how="left").with_columns(pl.col("nt").fill_null(0))
    return gt, base


def macro(pred, gt, base):
    tp = pred.join(gt, on=["s1_id", "rec_id"], how="left").group_by("s1_id").agg(pl.len().alias("np"), pl.col("tt").sum().alias("tp"))
    g = base.join(tp, on="s1_id", how="left").with_columns(pl.col(["np", "tp"]).fill_null(0))
    return g.select(pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0)
                    .otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp")))).mean()).item()


def decide(x, col, thr):
    best = x.filter(pl.col(col) == pl.col(col).max().over("rec_id")).sort(["rec_id", "s1_id"]).unique("rec_id", keep="first")
    return best.filter(pl.col(col) >= thr).select("s1_id", "rec_id")


def evaluate():
    b = frame("train").with_columns(fold_of().alias("fold"))
    FEATS = feats()
    print("stage-4 features:", len(FEATS), "| LLM sets:", LLMS, "| raw:", RAW, flush=True)
    X = b.select(FEATS).to_numpy().astype(np.float32)
    y = b["y"].to_numpy()
    fold = b["fold"].to_numpy()
    p4 = np.zeros(len(b), np.float32)
    for k in range(5):
        tr, va = fold != k, fold == k
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), ROUNDS)
        p4[va] = m.predict(X[va])
        print(f"fold {k} done", flush=True)
    b = b.with_columns(pl.Series("p4", p4))
    tag = "_".join(LLMS) + ("_raw" if RAW else "")
    b.select("rec_id", "s1_id", "y", "fold", "pe", "p4").write_parquet(cache(f"stage4_oof_{tag}.parquet"))
    from sklearn.metrics import roc_auc_score
    print("band AUC: blend %.5f  " % roc_auc_score(y, b["pe"]) + "  ".join(f"{n} {roc_auc_score(y, b[n].fill_null(-20)):.5f}" for n in LLMS)
          + f"  stage4 {roc_auc_score(y, p4):.5f}", flush=True)
    full = blend("train").select("rec_id", "s1_id", "pe")
    out = full.join(b.select("rec_id", "s1_id", "p4"), on=["rec_id", "s1_id"], how="left") \
              .with_columns(pl.coalesce("p4", "pe").alias("pf"))
    gt, base = _macro_setup()
    F0 = macro(decide(out, "pe", 0.725), gt, base)
    print(f"v25 rule (blend, thr 0.725): {F0:.6f}", flush=True)
    for thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.725, 0.75, 0.8]:
        F = macro(decide(out, "pf", thr), gt, base)
        print(f"stage 4 thr {thr}: {F:.6f}  delta {(F - F0) * 1e6:+.0f}e-6", flush=True)


def predict(thr, sub_path, out_path):
    from france_rules import read_pairs, write_pairs
    tr = frame("train")
    FEATS = feats()
    m = lgb.train(PARAMS, lgb.Dataset(tr.select(FEATS).to_numpy().astype(np.float32), tr["y"].to_numpy()), ROUNDS)
    te = frame("test")
    te = te.with_columns(pl.Series("p4", m.predict(te.select(FEATS).to_numpy().astype(np.float32)).astype(np.float32)))
    full = blend("test").select("rec_id", "s1_id", "pe")
    out = full.join(te.select("rec_id", "s1_id", "p4"), on=["rec_id", "s1_id"], how="left").with_columns(pl.coalesce("p4", "pe").alias("pf"))
    dec = decide(out, "pf", thr)
    s1 = read_source("test", 1).select(pl.col("entity_id").alias("s1_id"), "country")
    seen = set(read_source("train", 1)["country"].unique().to_list())
    base = read_pairs(sub_path).join(s1, on="s1_id")
    unseen = base.filter(~pl.col("country").is_in(list(seen))).select("s1_id", "rec_id")
    old = base.filter(pl.col("country").is_in(list(seen))).select("s1_id", "rec_id")
    res = pl.concat([unseen, dec])
    print(f"stage 4 thr {thr}: seen {old.height} -> {dec.height} (+{dec.join(old, on=['s1_id', 'rec_id'], how='anti').height} / "
          f"-{old.join(dec, on=['s1_id', 'rec_id'], how='anti').height}); unseen kept {unseen.height}", flush=True)
    write_pairs(res, out_path)


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate()
    elif sys.argv[1] == "predict":
        predict(float(sys.argv[2]), sys.argv[3], sys.argv[4])
