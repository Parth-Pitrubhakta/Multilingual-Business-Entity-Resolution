"""Matching model: pairwise LightGBM (stage 2) + group-aware stacker (stage 3).

Pipeline per split:
  1. candidates  = per record, its top-3 stage-1 pairs with p1 >= 0.02 (the final candidate set)
  2. features    = pairwise similarity features (features.py) + retrieval
                   context features + p1
  3. stage 2     = LightGBM, 5-fold cross-fitted on train (folds by S1 id)
                   so every train pair has an out-of-fold score p2
  4. stage 3     = LightGBM on stage-2 features + group features computed
                   from p2: competition among the S1 candidates of a record,
                   the S1's other confident records, and the similarity of the
                   record to the S1's most confident sibling record
  5. decision    = each record goes to at most one S1 (its arg-max), kept if
                   p3 >= threshold tuned for macro F0.5 on the held-out fold
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from common import cache, f05_macro, read_ground_truth, read_source
from features import pair_features
from stage1 import S1_FEATS, fold_of

CAND_THR = float(os.environ.get("CAND_THR", "0.02"))
CAND_TOPK = int(os.environ.get("CAND_TOPK", "3"))
MODEL_TAG = os.environ.get("MODEL_TAG", "")        # prefix of model / threshold / score files (e.g. "full_", "vf_")
TOKSTATS = os.environ.get("TOKSTATS", "1") == "1"  # label-free token house-number statistics (v10 features)   # each record matches at most one S1: keep its best few


def select_candidates(st: pl.DataFrame) -> pl.DataFrame:
    """Final candidate set = pairs with p1 >= CAND_THR among the record's CAND_TOPK best by p1."""
    return (st.with_columns(pl.col("p1").rank("ordinal", descending=True).over("rec_id").alias("_k"))
            .filter((pl.col("p1") >= CAND_THR) & (pl.col("_k") <= CAND_TOPK)).drop("_k"))
NFOLD = 5
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=160, verbose=-1)


def candidates(split: str) -> pl.DataFrame:
    return select_candidates(pl.read_parquet(cache(f"{split}_stage1.parquet")))


def build_features(split: str) -> pl.DataFrame:
    """Candidate pairs + pairwise features, cached to <split>_feats.parquet."""
    path = cache(f"{split}_feats.parquet")
    c = candidates(split)
    c = c.with_columns(
        pl.len().over("rec_id").alias("c_rec_n"),
        pl.len().over("s1_id").alias("c_s1_n"),
        pl.col("p1").rank("ordinal", descending=True).over("rec_id").alias("c_rec_rank"),
        (pl.col("p1") - pl.col("p1").max().over("rec_id")).alias("c_rec_gap"),
        pl.col("p1").rank("ordinal", descending=True).over("s1_id").alias("c_s1_rank"),
        pl.col("p1").sum().over("s1_id").alias("c_s1_psum"),
    )
    norm = pl.read_parquet(cache(f"{split}_norm.parquet"))
    keep_cols = [x for x in c.columns if x not in ("src",)]
    prev = cache(f"{split}_feats.parquet")
    if os.path.exists(prev):  # pure speed-up: identical values for pairs already featurised
        old = pl.read_parquet(prev)
        pcols = [x for x in old.columns if x not in set(c.columns) | {"y", "fold"} or x == "src"]
        pcols = [x for x in pcols if not x.startswith("fq")]
        old = old.select(["rec_id", "s1_id"] + [x for x in pcols if x not in ("rec_id", "s1_id")])
        have = c.select("rec_id", "s1_id").join(old, on=["rec_id", "s1_id"], how="inner")
        miss = c.select("rec_id", "s1_id").join(old.select("rec_id", "s1_id"), on=["rec_id", "s1_id"], how="anti")
        del old
        print(f"{split}: reusing {have.height} featurised pairs, computing {miss.height}", flush=True)
        if miss.height:
            have = pl.concat([have, pair_features(miss, norm).select(have.columns)], how="vertical_relaxed")
        f = c.select(keep_cols).join(have, on=["rec_id", "s1_id"], how="left")
    else:
        f = pair_features(c.select(keep_cols), norm)
    f = f.join(name_frequency(norm, "rec_id", "fq2"), on="rec_id", how="left")
    f = f.join(name_frequency(norm, "s1_id", "fq1"), on="s1_id", how="left")
    f.write_parquet(path)
    print(split, "features", f.shape)
    return f


def name_key(norm: pl.DataFrame) -> pl.Expr:
    """Order-insensitive exact key of the core name (web stem if name is only a URL)."""
    return pl.concat_str([pl.col("country"), pl.col("n_core").str.split(" ").list.sort().list.join(" ")], separator="|")


def name_frequency(norm: pl.DataFrame, id_col: str, pref: str) -> pl.DataFrame:
    """How many S1 entities / S2-S3 records (same country) share this record's name key.

    A near-exact name match is strong evidence only when the name is rare.
    """
    k = norm.select("entity_id", "src", name_key(norm).alias("k"))
    cnt = k.group_by("k").agg((pl.col("src") == 1).sum().alias(pref + "_s1"),
                              (pl.col("src") != 1).sum().alias(pref + "_rec"))
    return k.join(cnt, on="k").select(pl.col("entity_id").alias(id_col), pref + "_s1", pref + "_rec")


def twin_features(f: pl.DataFrame, pcol: str, norm: pl.DataFrame) -> pl.DataFrame:
    """Where do the other records with the same name key go?

    For pair (r, e): among records sharing r's name key, how many are
    confidently assigned to e, to some other S1, or to nothing.
    """
    best = f.filter(pl.col(pcol) == pl.col(pcol).max().over("rec_id")).unique("rec_id")
    best = best.select("rec_id", pl.col("s1_id").alias("b_s1"), pl.col(pcol).alias("b_p"))
    k = norm.filter(pl.col("src") != 1).select(pl.col("entity_id").alias("rec_id"), name_key(norm).alias("k"))
    kb = k.join(best, on="rec_id", how="left").with_columns(
        pl.when(pl.col("b_p") >= 0.5).then(pl.col("b_s1")).otherwise(None).alias("a_s1"))
    tot = kb.group_by("k").agg(pl.len().alias("tw_n"), pl.col("a_s1").is_not_null().sum().alias("tw_assigned"))
    per = kb.filter(pl.col("a_s1").is_not_null()).group_by(["k", "a_s1"]).agg(pl.len().alias("tw_to"))
    f = f.join(k, on="rec_id", how="left").join(tot, on="k", how="left")
    f = f.join(per.rename({"a_s1": "s1_id"}), on=["k", "s1_id"], how="left")
    f = f.join(kb.select("rec_id", pl.col("a_s1").alias("_self")), on="rec_id", how="left")
    self_e = (pl.col("_self") == pl.col("s1_id")).fill_null(False).cast(pl.Int32)
    self_any = pl.col("_self").is_not_null().cast(pl.Int32)
    f = f.with_columns(
        (pl.col("tw_n") - 1).alias("tw_n"),
        (pl.col("tw_to").fill_null(0) - self_e).alias("tw_to_e"),
        (pl.col("tw_assigned") - pl.col("tw_to").fill_null(0) - (self_any - self_e)).alias("tw_to_other"),
    ).with_columns((pl.col("tw_n") - pl.col("tw_to_e") - pl.col("tw_to_other")).alias("tw_unassigned"))
    return f.drop("k", "tw_to", "tw_assigned", "_self")


# --------------------------------------------------------------------------
# target encoding of name differences (out-of-fold on train)
# --------------------------------------------------------------------------

TE_PRIOR_W = 20.0
TE_COLS = ("x_extra", "x_missing", "x_legal", "x_swap", "x_aextra", "x_amissing", "x_city")


def te_keys(f: pl.DataFrame, norm: pl.DataFrame) -> pl.DataFrame:
    """Per-pair difference keys that are target-encoded:

    x_extra / x_missing   name tokens only in the record / only in S1
    x_legal               legal-form transition, e.g. 'ltd pvt>ltd'
    x_swap                'a>b' when exactly one S1 name token was replaced by another
    x_aextra / x_amissing non-numeric address tokens only in the record / only in S1
    x_city                'city_S1>city_record' when the guessed cities differ
    """
    n = norm.select(
        "entity_id", "country", pl.col("n_core").str.split(" ").alias("t"), pl.col("n_legal"),
        pl.col("a_tok").str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))).alias("a"),
        pl.col("a_city"))
    k = f.select("rec_id", "s1_id")
    k = k.join(n.select(pl.col("entity_id").alias("s1_id"), "country", pl.col("t").alias("t1"), pl.col("n_legal").alias("l1"),
                        pl.col("a").alias("a1"), pl.col("a_city").alias("c1")),
               on="s1_id", how="left", maintain_order="left")
    k = k.join(n.select(pl.col("entity_id").alias("rec_id"), pl.col("t").alias("t2"), pl.col("n_legal").alias("l2"),
                        pl.col("a").alias("a2"), pl.col("a_city").alias("c2")),
               on="rec_id", how="left", maintain_order="left")
    k = k.with_columns(
        pl.col("t2").list.set_difference(pl.col("t1")).alias("x_extra"),
        pl.col("t1").list.set_difference(pl.col("t2")).alias("x_missing"),
        pl.concat_str([pl.col("l1").fill_null(""), pl.col("l2").fill_null("")], separator=">").alias("x_legal"),
        pl.col("a2").list.set_difference(pl.col("a1")).alias("x_aextra"),
        pl.col("a1").list.set_difference(pl.col("a2")).alias("x_amissing"),
        pl.when((pl.col("c1") != pl.col("c2")) & (pl.col("c1") != "") & (pl.col("c2") != ""))
        .then(pl.concat_str([pl.col("c1"), pl.col("c2")], separator=">")).otherwise(None).alias("x_city"),
    )
    k = k.with_columns(
        pl.when((pl.col("x_extra").list.len() == 1) & (pl.col("x_missing").list.len() == 1))
        .then(pl.concat_str([pl.col("x_missing").list.first(), pl.col("x_extra").list.first()], separator=">"))
        .otherwise(None).alias("x_swap"))
    return k.select("rec_id", "s1_id", "country", *TE_COLS)


def _te_table(keys: pl.DataFrame, y: pl.Series, col: str, prior: float) -> pl.DataFrame:
    d = keys.select(col).with_columns(y.alias("y"))
    if d[col].dtype == pl.List(pl.String):
        d = d.explode(col)
    d = d.filter(pl.col(col).is_not_null() & (pl.col(col) != ""))
    return d.group_by(col).agg(
        ((pl.col("y").sum() + TE_PRIOR_W * prior) / (pl.len() + TE_PRIOR_W)).alias("te"),
        pl.len().alias("te_n"))


def _te_apply(keys: pl.DataFrame, tab: pl.DataFrame, col: str, prior: float) -> pl.DataFrame:
    """Aggregate target-encoded values of a key column per row.

    Tokens never seen in the fitted table stay *missing* (not the prior): in
    train an unseen token is almost always a typo of a true match, while in a
    new country (France) it is an ordinary word, so imputing a match-like
    prior would transfer the wrong meaning. `_n` / `_unseen` count the tokens.
    """
    kk = keys.select(pl.int_range(pl.len()).alias("_i"), col)
    if kk[col].dtype == pl.List(pl.String):
        e = kk.explode(col).join(tab, on=col, how="left")
        present = pl.col(col).is_not_null() & (pl.col(col) != "")
        a = e.group_by("_i").agg(
            pl.col("te").min().alias(f"te_{col}_min"), pl.col("te").max().alias(f"te_{col}_max"),
            pl.col("te").mean().alias(f"te_{col}_mean"), pl.col("te_n").min().alias(f"te_{col}_cnt"),
            present.sum().alias(f"te_{col}_n"),
            (present & pl.col("te").is_null()).sum().alias(f"te_{col}_unseen"))
    else:
        a = kk.join(tab, on=col, how="left").select(
            "_i",
            pl.when(pl.col(col).is_null()).then(-1.0).otherwise(pl.col("te")).alias(f"te_{col}"),
            pl.col("te_n").fill_null(0).alias(f"te_{col}_cnt"))
    return kk.select("_i").join(a, on="_i", how="left").sort("_i").drop("_i")


LIST_TE = ("x_extra", "x_missing", "x_aextra", "x_amissing")
SCALAR_TE = ("x_legal", "x_swap", "x_city")


MASK_CE = os.environ.get("MASK_CE", "1") == "1"   # v10: cross-encoder features are vocabulary-dependent too
VOCAB_FREE_UNSEEN = os.environ.get("VOCAB_FREE_UNSEEN", "1") == "1"  # mask them for countries absent from training


def mask_te(X: np.ndarray, cols: list, rows: np.ndarray) -> None:
    """In place: make vocabulary-dependent features of `rows` look like a country never
    seen in training: every token unseen by the target encodings and (MASK_CE) no
    cross-encoder score. Used for dropout while fitting, for countries absent from
    training at inference, and for the pseudo-new-country evaluation."""
    idx = {c: j for j, c in enumerate(cols)}
    if MASK_CE:
        for c in cols:
            if c.split("_")[0] in ("ce", "cel", "cel2", "cem"):
                X[rows, idx[c]] = np.nan
    for c in LIST_TE:
        for s in ("min", "max", "mean", "cnt"):
            if f"te_{c}_{s}" in idx:
                X[rows, idx[f"te_{c}_{s}"]] = np.nan
        if f"te_{c}_unseen" in idx:
            X[rows, idx[f"te_{c}_unseen"]] = X[rows, idx[f"te_{c}_n"]]
    for c in SCALAR_TE:
        if f"te_{c}" in idx:
            j = idx[f"te_{c}"]
            v = X[rows, j]
            X[rows, j] = np.where(v == -1, -1, np.nan)
            X[rows, idx[f"te_{c}_cnt"]] = 0


TOKSTAT_MIN_N = 5


def token_hn_stats(keys: pl.DataFrame, hnr_eq: pl.Series) -> pl.DataFrame:
    """Label-free, language-independent: for every name token a record ADDS (or DROPS)
    relative to the S1 name, the leave-one-out share of candidate pairs in this split
    (same country) carrying that token whose house numbers are identical.

    Decoy words (group, holding, participations, ...) sit on look-alike businesses at
    a nearby house number (share ~0.05); noise words that the generator adds to true
    records (center, services, fils, associes, ...) sit at the same number (~0.75-0.9).
    """
    kk = keys.select(pl.int_range(pl.len()).alias("_i"), "country", "x_extra", "x_missing").with_columns(
        hnr_eq.alias("_h"))
    out = kk.select("_i")
    for col, pref in (("x_extra", "ts_ex"), ("x_missing", "ts_mi")):
        e = kk.select("_i", "country", "_h", pl.col(col).alias("tok")).explode("tok").filter(
            pl.col("tok").is_not_null() & (pl.col("tok") != ""))
        e = e.with_columns((pl.col("_h") >= 0).cast(pl.Int32).alias("_v"), (pl.col("_h") == 1).cast(pl.Int32).alias("_e"))
        tab = e.group_by(["country", "tok"]).agg(pl.col("_v").sum().alias("_n"), pl.col("_e").sum().alias("_s"))
        e = e.join(tab, on=["country", "tok"], how="left").with_columns(
            (pl.col("_n") - pl.col("_v")).alias("_nl"), (pl.col("_s") - pl.col("_e")).alias("_sl"))
        e = e.with_columns(pl.when(pl.col("_nl") >= TOKSTAT_MIN_N).then(pl.col("_sl") / pl.col("_nl")).otherwise(None).alias("_r"))
        a = e.group_by("_i").agg(pl.col("_r").min().alias(f"{pref}_hn_min"), pl.col("_r").max().alias(f"{pref}_hn_max"),
                                 pl.col("_r").mean().alias(f"{pref}_hn_mean"),
                                 pl.col("_nl").min().log1p().cast(pl.Float32).alias(f"{pref}_hn_logn"))
        out = out.join(a, on="_i", how="left")
    return out.sort("_i").drop("_i")


def token_features(keys: pl.DataFrame, norm: pl.DataFrame) -> pl.DataFrame:
    """Label-free, country-transferable descriptions of the name differences.

    tk_swap_sim        similarity of the two tokens of a single-token swap
                       (typo: high, a different word: low)
    tk_ex_df_min/max   log document frequency (same split & country) of the
    tk_mi_df_min/max   tokens the record adds / drops. A typo or random trade
                       name is rare, an ordinary word ('groupe', 'holdings') common.
    """
    from rapidfuzz.distance import JaroWinkler as JW
    t = norm.select("entity_id", "country", pl.col("n_core").str.split(" ").alias("tok")).explode("tok")
    t = t.filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).unique(["entity_id", "tok"])
    dfreq = t.group_by(["country", "tok"]).agg(pl.len().log1p().cast(pl.Float32).alias("ldf"))
    kk = keys.select(pl.int_range(pl.len()).alias("_i"), "country", "x_extra", "x_missing", "x_swap")
    out = kk.select("_i")
    for col, pref in (("x_extra", "tk_ex"), ("x_missing", "tk_mi")):
        e = kk.select("_i", "country", pl.col(col).alias("tok")).explode("tok").join(dfreq, on=["country", "tok"], how="left")
        e = e.with_columns(pl.when(pl.col("tok").is_null() | (pl.col("tok") == "")).then(None)
                           .otherwise(pl.col("ldf").fill_null(0.0)).alias("ldf"))
        a = e.group_by("_i").agg(pl.col("ldf").min().alias(f"{pref}_df_min"), pl.col("ldf").max().alias(f"{pref}_df_max"))
        out = out.join(a, on="_i", how="left")
    sw = kk["x_swap"].fill_null("").to_list()
    a_, b_ = zip(*[(x.split(">", 1) + [""])[:2] if x else ("", "") for x in sw]) if sw else ([], [])
    sim = process.cpdist(list(a_), list(b_), scorer=JW.normalized_similarity, workers=-1).astype(np.float32)
    sim[np.array([not x for x in sw])] = -1
    out = out.sort("_i").with_columns(pl.Series("tk_swap_sim", sim))
    return out.drop("_i")


def te_features(f: pl.DataFrame, norm: pl.DataFrame, fit: pl.DataFrame | None = None) -> pl.DataFrame:
    """Train (fit=None): out-of-fold encodings by S1 fold. Test: encode with tables
    fitted on all training pairs (fit = labelled train key frame)."""
    keys = te_keys(f, norm)
    if fit is None:
        keys = keys.with_columns(f["y"], f["fold"])
        prior = float(f["y"].mean())
        parts = []
        for k in range(NFOLD):
            trk = keys.filter(pl.col("fold") != k)
            vak = keys.with_row_index("_r").filter(pl.col("fold") == k)
            enc = [_te_apply(vak, _te_table(trk, trk["y"], c, prior), c, prior) for c in TE_COLS]
            parts.append(pl.concat([vak.select("_r")] + enc, how="horizontal"))
        out = pl.concat(parts).sort("_r").drop("_r")
    else:
        prior = float(fit["y"].mean())
        out = pl.concat([_te_apply(keys, _te_table(fit, fit["y"], c, prior), c, prior) for c in TE_COLS],
                        how="horizontal")
    tok = token_features(keys, norm)
    if not TOKSTATS:
        return pl.concat([f, out, tok], how="horizontal")
    hn = f["hnr_eq"] if "hnr_eq" in f.columns else pl.Series("hnr_eq", np.full(f.height, -1.0, np.float32))
    return pl.concat([f, out, tok, token_hn_stats(keys, hn)], how="horizontal")


def feat_cols(df: pl.DataFrame):
    drop = {"rec_id", "s1_id", "y", "fold", "p2", "p3"}
    return [c for c in df.columns if c not in drop and df[c].dtype != pl.String]


def _lgb_train(X, y, rounds):
    return lgb.train(LGB_PARAMS, lgb.Dataset(X, y, free_raw_data=True), num_boost_round=rounds)


def add_labels(f: pl.DataFrame) -> pl.DataFrame:
    gt = read_ground_truth().with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    f = f.drop([c for c in ("y",) if c in f.columns])
    return f.join(gt, on=["rec_id", "s1_id"], how="left").with_columns(pl.col("y").fill_null(0),
                                                                        fold_of().alias("fold"))


# --------------------------------------------------------------------------
# stage 3 group features
# --------------------------------------------------------------------------

def group_features(f: pl.DataFrame, pcol: str, norm: pl.DataFrame) -> pl.DataFrame:
    """Features describing each pair relative to its record's and S1's other pairs."""
    f = f.with_columns(
        pl.col(pcol).max().over("rec_id").alias("g_rec_max"),
        pl.col(pcol).rank("ordinal", descending=True).over("rec_id").alias("g_rec_rank"),
        pl.col(pcol).sum().over("rec_id").alias("g_rec_sum"),
        pl.col(pcol).rank("ordinal", descending=True).over("s1_id").alias("g_s1_rank"),
        pl.col(pcol).sum().over("s1_id").alias("g_s1_sum"),
        (pl.col(pcol) > 0.5).sum().over("s1_id").alias("g_s1_n50"),
        pl.col(pcol).max().over("s1_id").alias("g_s1_max"),
    )
    # best competing S1 for the same record (excluding this pair)
    f = f.with_columns(
        pl.when(pl.col("g_rec_rank") == 1)
        .then(pl.col(pcol).filter(pl.col("g_rec_rank") == 2).first().over("rec_id").fill_null(0))
        .otherwise(pl.col("g_rec_max")).alias("g_rec_other")
    ).with_columns((pl.col(pcol) - pl.col("g_rec_other")).alias("g_rec_margin"))
    # most confident sibling record of the same S1 (excluding this record)
    top = f.filter(pl.col("g_s1_rank") <= 2).select("s1_id", "rec_id", pl.col(pcol).alias("sib_p"),
                                                     pl.col("g_s1_rank").alias("sib_rank"))
    f = f.join(top.filter(pl.col("sib_rank") == 1).select("s1_id", pl.col("rec_id").alias("sib1"),
                                                            pl.col("sib_p").alias("sib1_p")), on="s1_id", how="left")
    f = f.join(top.filter(pl.col("sib_rank") == 2).select("s1_id", pl.col("rec_id").alias("sib2"),
                                                            pl.col("sib_p").alias("sib2_p")), on="s1_id", how="left")
    f = f.with_columns(
        pl.when(pl.col("sib1") == pl.col("rec_id")).then(pl.col("sib2")).otherwise(pl.col("sib1")).alias("sib"),
        pl.when(pl.col("sib1") == pl.col("rec_id")).then(pl.col("sib2_p")).otherwise(pl.col("sib1_p"))
        .fill_null(0).alias("g_sib_p"),
    ).drop("sib1", "sib2", "sib1_p", "sib2_p")
    nm = norm.select("entity_id", "n_core", "a_tok")
    f = f.join(nm.select(pl.col("entity_id").alias("rec_id"), pl.col("n_core").alias("_n"), pl.col("a_tok").alias("_a")),
               on="rec_id", how="left")
    f = f.join(nm.select(pl.col("entity_id").alias("sib"), pl.col("n_core").alias("_sn"), pl.col("a_tok").alias("_sa")),
               on="sib", how="left")
    g = lambda c: f[c].fill_null("").to_list()
    has = f["sib"].is_not_null().to_numpy()
    sn = process.cpdist(g("_n"), g("_sn"), scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    sa = process.cpdist(g("_a"), g("_sa"), scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    sr = process.cpdist(g("_n"), g("_sn"), scorer=fuzz.ratio, workers=-1).astype(np.float32)
    sn[~has], sa[~has], sr[~has] = -1, -1, -1
    f = f.with_columns(pl.Series("g_sib_nm", sn), pl.Series("g_sib_ad", sa), pl.Series("g_sib_nr", sr))
    return f.drop("sib", "_n", "_a", "_sn", "_sa")


# --------------------------------------------------------------------------
# decision + evaluation
# --------------------------------------------------------------------------

def decide(f: pl.DataFrame, pcol: str, thr: float) -> pl.DataFrame:
    """Each record -> its arg-max S1 if p >= thr. Returns (s1_id, rec_id) pairs."""
    best = f.filter(pl.col(pcol) == pl.col(pcol).max().over("rec_id"))
    best = best.unique("rec_id", keep="first")
    return best.filter(pl.col(pcol) >= thr).select("s1_id", "rec_id")


def evaluate(pred: pl.DataFrame, s1_ids, truth: dict) -> float:
    pd = {}
    for s, r in pred.iter_rows():
        pd.setdefault(s, set()).add(r)
    return f05_macro(pd, truth, s1_ids)


def truth_dict(s1_ids):
    gt = read_ground_truth()
    ss = set(s1_ids)
    t = {}
    for s, r in gt.iter_rows():
        if s in ss:
            t.setdefault(s, set()).add(r)
    return t


def tune(f: pl.DataFrame, pcol: str, s1_ids, truth):
    best = (0, 0.5)
    for thr in np.arange(0.2, 0.9, 0.025):
        sc = evaluate(decide(f, pcol, thr), s1_ids, truth)
        if sc > best[0]:
            best = (sc, float(thr))
    print(f"  best {pcol}: F0.5={best[0]:.5f} @ thr={best[1]:.3f}")
    return best


# --------------------------------------------------------------------------
# train / predict
# --------------------------------------------------------------------------

# v5 training configuration
AUG_FRAMES = ["train_sim_f19_s1_feats.parquet",   # orphan simulation (simulate.py)
              "train_dec_r8_s2_feats.parquet"]   # near-miss decoy injection (simulate.py decoys)
AUG_NORMS = ["train_dec_r8_s2_norm.parquet"]     # normalised decoy records
TE_DROP = 0.30                                   # share of S1 entities with target encodings hidden while fitting
FIT_PARAMS = dict(LGB_PARAMS, learning_rate=float(os.environ.get("FIT_LR", "0.1")), num_threads=int(os.environ.get("LGB_THREADS", "160")))
if os.environ.get("LGB_SEED"):  # optional: a differently seeded stack for blending (default keeps LightGBM's seeds)
    FIT_PARAMS["seed"] = int(os.environ["LGB_SEED"])
ROUNDS2, ROUNDS3 = int(os.environ.get("ROUNDS2", "400")), int(os.environ.get("ROUNDS3", "250"))
NC = ["entity_id", "country", "n_core", "n_legal", "a_tok", "a_city"]
NG = ["entity_id", "src", "country", "n_core", "a_tok"]


def cross_fit(X, y, fold, drop, cols, rounds, name):
    """5-fold cross-fitting by S1 fold. TE-dropout rows are masked in the *fitting*
    data only, so every model learns to decide both with and without encodings."""
    oof = np.zeros(len(y), dtype=np.float32)
    resume_after = float(os.environ.get("RESUME_AFTER", "inf"))  # reuse fold models written after this time
    for k in range(NFOLD):
        path = cache(f"{MODEL_TAG}{name}_fold{k}.lgb")
        if os.path.exists(path) and os.path.getmtime(path) > resume_after:
            m = lgb.Booster(model_file=path)
            print(f"  {name} fold {k} reused", flush=True)
            va = fold == k
            oof[va] = m.predict(X[va], num_threads=160)
            continue
        tr = np.where(fold != k)[0]
        Xt = X[tr]
        mask_te(Xt, cols, np.where(drop[tr])[0])
        m = lgb.train(FIT_PARAMS, lgb.Dataset(Xt, y[tr], free_raw_data=True), num_boost_round=rounds)
        del Xt
        va = fold == k
        oof[va] = m.predict(X[va], num_threads=160)
        m.save_model(cache(f"{MODEL_TAG}{name}_fold{k}.lgb"))
        print(f"  {name} fold {k} done", flush=True)
    return oof


USE_CE = os.environ.get("USE_CE", "1") == "1"


CE_VARIANTS = [v for v in os.environ.get("CE_VARIANTS", "ce,cel,cel2").split(",") if v]  # "ce" = base, "cel" = large


def attach_ce(f: pl.DataFrame, split: str) -> pl.DataFrame:
    """Cross-encoder scores (cross_encoder.py; out-of-fold on train) + record-level context.

    Variant "ce" reads ce_scores_{split}.parquet (XLM-R base), "cel" reads
    ce_scores_{split}_large.parquet (XLM-R large)."""
    if not USE_CE:
        return f
    for var in CE_VARIANTS:
        suffix = {"ce": "", "cel": "_large", "cel2": "_large2", "cem": "_mdeberta"}[var]
        path = cache(f"ce_scores_{split}{suffix}.parquet")
        if not os.path.exists(path):
            continue
        ce = pl.read_parquet(path).unique(["rec_id", "s1_id"], keep="first").rename({"ce": var})
        f = f.join(ce, on=["rec_id", "s1_id"], how="left", maintain_order="left")
        f = f.with_columns(
            (pl.col(var) - pl.col(var).max().over("rec_id")).alias(f"{var}_gap"),
            pl.col(var).rank("ordinal", descending=True).over("rec_id").cast(pl.Int32).alias(f"{var}_rank"),
            (pl.col(var) - pl.col(var).filter(pl.col(var) < pl.col(var).max()).max().over("rec_id").fill_null(-20)).alias(f"{var}_margin"),
        )
    return f


def _frame(path: str, normk: pl.DataFrame) -> pl.DataFrame:
    f = add_labels(pl.read_parquet(cache(path)))
    return attach_ce(te_features(f, normk), "train")


def _apply_folds(f: pl.DataFrame, cols, name, masked=False) -> np.ndarray:
    """Out-of-fold scores of a frame with the saved fold models (optionally all-TE-masked)."""
    X = f.select(cols).to_numpy().astype(np.float32)
    if masked:
        mask_te(X, cols, np.arange(len(X)))
    fold = f["fold"].to_numpy()
    p = np.zeros(len(f), dtype=np.float32)
    for k in range(NFOLD):
        va = fold == k
        p[va] = lgb.Booster(model_file=cache(f"{MODEL_TAG}{name}_fold{k}.lgb")).predict(X[va], num_threads=160)
    return p


def _eval_sets():
    """Validation entities (fold 0) of the original data and of each simulation."""
    from simulate import deleted_ids
    s1 = read_source("train", 1).select(pl.col("entity_id").alias("s1_id")).with_columns(fold_of().alias("fold"))
    base = s1.filter(pl.col("fold") == 0)["s1_id"]
    sets = {"orig": base.to_list()}
    for path in AUG_FRAMES:
        tag = path.replace("train_", "").replace("_feats.parquet", "")
        if tag.startswith("sim_f"):
            frac, seed = int(tag.split("_f")[1].split("_")[0]) / 100, int(tag.split("_s")[-1])
            D = deleted_ids(frac, seed)
            sets[tag] = base.filter(~base.is_in(D.implode())).to_list()
        else:  # decoy injection keeps every entity
            sets[tag] = base.to_list()
    return sets


def _norm_with_aug(cols):
    return pl.concat([pl.read_parquet(cache("train_norm.parquet"), columns=cols)]
                     + [pl.read_parquet(cache(p), columns=cols) for p in AUG_NORMS if os.path.exists(cache(p))])


def train():
    normk = _norm_with_aug(NC)
    norm = _norm_with_aug(NG)
    frames = {"orig": _frame("train_feats.parquet", normk)}
    te_keys(frames["orig"], normk).with_columns(frames["orig"]["y"]).write_parquet(cache("train_te_keys.parquet"))
    for path in AUG_FRAMES:
        tag = path.replace("train_", "").replace("_feats.parquet", "")
        frames[tag] = _frame(path, normk)
    del normk
    names = list(frames)
    cols2 = feat_cols(frames["orig"])
    print("stage2 features:", len(cols2), "| frames:", {k: v.height for k, v in frames.items()}, flush=True)
    sizes = [frames[k].height for k in names]
    cat = pl.concat([frames[k].select(["s1_id", "y", "fold"] + cols2) for k in names], how="vertical_relaxed")
    drop = ((cat["s1_id"].hash(seed=7) % 100) < int(TE_DROP * 100)).to_numpy()
    X = cat.select(cols2).to_numpy().astype(np.float32)
    oof2 = cross_fit(X, cat["y"].to_numpy(), cat["fold"].to_numpy(), drop, cols2, ROUNDS2, "stage2")
    del X
    off = np.cumsum([0] + sizes)
    for i, k in enumerate(names):
        g = frames[k].with_columns(pl.Series("p2", oof2[off[i]:off[i + 1]]))
        frames[k] = twin_features(group_features(g, "p2", norm), "p2", norm)
    cols3 = feat_cols(frames["orig"]) + ["p2"]
    cat = pl.concat([frames[k].select(["s1_id", "y", "fold"] + cols3) for k in names], how="vertical_relaxed")
    X = cat.select(cols3).to_numpy().astype(np.float32)
    oof3 = cross_fit(X, cat["y"].to_numpy(), cat["fold"].to_numpy(), drop, cols3, ROUNDS3, "stage3")
    del X, cat
    for i, k in enumerate(names):
        frames[k] = frames[k].with_columns(pl.Series("p3", oof3[off[i]:off[i + 1]]))
    json.dump({"cols2": cols2, "cols3": cols3}, open(cache(f"{MODEL_TAG}cols.json"), "w"))
    # ---- evaluation: original, harder (orphan simulation), pseudo-new-country (all TE unseen)
    sets = _eval_sets()
    truths = {k: truth_dict(v) for k, v in sets.items()}
    grid = np.round(np.arange(0.3, 0.951, 0.025), 3)
    table = {}
    for k in names:
        v = frames[k].filter(pl.col("fold") == 0)
        table[k] = [evaluate(decide(v, "p3", t), sets[k], truths[k]) for t in grid]
    mean = np.mean([table[k] for k in names], axis=0)
    thr = float(grid[int(np.argmax(mean))])
    for k in names:
        print(f"  [{k}] F0.5 @thr={thr}: {table[k][int(np.argmax(mean))]:.5f}  (best own: {max(table[k]):.5f} "
              f"@ {grid[int(np.argmax(table[k]))]})", flush=True)
        if k.startswith("dec"):
            v = frames[k].filter(pl.col("fold") == 0)
            dd = v.filter(pl.col("rec_id").str.slice(3, 1) == "D")
            merged = decide(v, "p3", thr).filter(pl.col("rec_id").str.slice(3, 1) == "D").height
            print(f"      decoy records in fold-0 candidates: {dd['rec_id'].n_unique()}, merged at thr: {merged}", flush=True)
    pf = frames["orig"].drop([c for c in frames["orig"].columns if c.startswith("g_") or c.startswith("tw_")]
                             + ["p2", "p3"])
    pf = pf.with_columns(pl.Series("p2", _apply_folds(pf, cols2, "stage2", masked=True)))
    pf = twin_features(group_features(pf, "p2", norm), "p2", norm)
    pf = pf.with_columns(pl.Series("p3", _apply_folds(pf, cols3, "stage3", masked=True)))
    v = pf.filter(pl.col("fold") == 0)
    pn = [evaluate(decide(v, "p3", t), sets["orig"], truths["orig"]) for t in grid]
    thr_u = float(grid[int(np.argmax(pn))])
    print(f"  [pseudo-new-country: vocabulary features hidden] F0.5 @thr={thr}: {pn[int(np.argmax(mean))]:.5f}"
          f" | own best {max(pn):.5f} @ {thr_u} (threshold used for countries unseen in training)", flush=True)
    with open(cache(f"{MODEL_TAG}thr_unseen.txt"), "w") as fh:
        fh.write(str(thr_u))
    with open(cache(f"{MODEL_TAG}thr.txt"), "w") as fh:
        fh.write(str(thr))
    frames["orig"].select("rec_id", "s1_id", "y", "fold", "p1", "p2", "p3").write_parquet(cache(f"train_oof{MODEL_TAG}.parquet"))


def unseen_country_rows(f: pl.DataFrame, norm: pl.DataFrame) -> np.ndarray:
    """Rows whose S1 country label never occurs among the training entities (open set:
    any new label). Their vocabulary-dependent features are masked at inference."""
    seen = set(pl.read_parquet(cache("train_norm.parquet"), columns=["country", "src"])
               .filter(pl.col("src") == 1)["country"].unique().to_list())
    c = f.select("s1_id").join(norm.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id",
                              how="left", maintain_order="left")["country"]
    return np.where(~c.is_in(list(seen)).to_numpy())[0]


def predict(split="test"):
    cols = json.load(open(cache(f"{MODEL_TAG}cols.json")))
    f = pl.read_parquet(cache(f"{split}_feats.parquet"))
    normk = pl.read_parquet(cache(f"{split}_norm.parquet"), columns=NC)
    f = attach_ce(te_features(f, normk, fit=pl.read_parquet(cache("train_te_keys.parquet"))), split)
    unseen = unseen_country_rows(f, normk) if VOCAB_FREE_UNSEEN else np.array([], dtype=np.int64)
    print(f"{split}: {len(unseen)} pairs from countries unseen in training -> vocabulary-free features", flush=True)
    del normk
    X = f.select(cols["cols2"]).to_numpy().astype(np.float32)
    mask_te(X, cols["cols2"], unseen)
    p2 = np.mean([lgb.Booster(model_file=cache(f"{MODEL_TAG}stage2_fold{k}.lgb")).predict(X, num_threads=160)
                  for k in range(NFOLD)], axis=0)
    f = f.with_columns(pl.Series("p2", p2.astype(np.float32)))
    norm = pl.read_parquet(cache(f"{split}_norm.parquet"), columns=["entity_id", "src", "country", "n_core", "a_tok"])
    f = group_features(f, "p2", norm)
    f = twin_features(f, "p2", norm)
    X = f.select(cols["cols3"]).to_numpy().astype(np.float32)
    mask_te(X, cols["cols3"], unseen)
    p3 = np.mean([lgb.Booster(model_file=cache(f"{MODEL_TAG}stage3_fold{k}.lgb")).predict(X, num_threads=160)
                  for k in range(NFOLD)], axis=0)
    f = f.with_columns(pl.Series("p3", p3.astype(np.float32)))
    f.select("rec_id", "s1_id", "p1", "p2", "p3").write_parquet(cache(f"{split}_scores{MODEL_TAG}.parquet"))
    return f


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        build_features(sys.argv[2])
    elif cmd == "train":
        train()
    elif cmd == "predict":
        predict(sys.argv[2] if len(sys.argv) > 2 else "test")
