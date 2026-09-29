"""Orphan-record simulation ("hard" training/validation data).

The test set has ~5.8 S2/S3 records per S1 entity against ~4.7 in train:
many more records belong to businesses that have *no* Source 1 entry. When
such an orphan is a look-alike of an existing S1 entity (same street, a
nearby house number, a similar name), that entity becomes the orphan's best
candidate with no competitor, a situation that is rare in train.

We reproduce it from the training data alone: delete a deterministic,
hash-selected fraction of the S1 entities, drop them from the retrieval
lists (shifting ranks), and rebuild everything downstream exactly as the
pipeline does (stage-1 context features, p1, the candidate set, context and
name-frequency features). The records of deleted entities become unlabelled
orphans. Pair features are reused from cache and computed only for pairs
that enter the candidate set because a deleted entity left it.
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import cache, read_source
from features import pair_features
from match import add_labels, name_frequency, select_candidates
from stage1 import NORM1, S1_FEATS, build_frame

CTX_COLS = ["c_rec_n", "c_s1_n", "c_rec_rank", "c_rec_gap", "c_s1_rank", "c_s1_psum"]
FQ_COLS = ["fq2_s1", "fq2_rec", "fq1_s1", "fq1_rec"]


def deleted_ids(frac: float, seed: int) -> pl.Series:
    s1 = read_source("train", 1).select("entity_id")
    sel = s1.filter((pl.col("entity_id").hash(seed=seed) % 100000) < int(frac * 100000))
    return sel["entity_id"]


def _shift_ranks(keep: pl.DataFrame, dele: pl.DataFrame) -> pl.DataFrame:
    """Ranks of surviving pairs move up by the number of deleted pairs ranked above them."""
    keep = keep.with_row_index("_r")
    d = dele.select("rec_id", pl.col("rk_c").alias("d_c"), pl.col("rk_n").alias("d_n"), pl.col("rk_a").alias("d_a"))
    j = keep.select("_r", "rec_id", "rk_c", "rk_n", "rk_a").join(d, on="rec_id", how="inner")
    sh = j.group_by("_r").agg(
        ((pl.col("d_c") < pl.col("rk_c")) & (pl.col("rk_c") < 999)).sum().alias("s_c"),
        ((pl.col("d_n") < pl.col("rk_n")) & (pl.col("rk_n") < 999)).sum().alias("s_n"),
        ((pl.col("d_a") < pl.col("rk_a")) & (pl.col("rk_a") < 999)).sum().alias("s_a"),
    )
    keep = keep.join(sh, on="_r", how="left").with_columns(
        (pl.col("rk_c") - pl.col("s_c").fill_null(0)).cast(pl.Int16).alias("rk_c"),
        (pl.col("rk_n") - pl.col("s_n").fill_null(0)).cast(pl.Int16).alias("rk_n"),
        (pl.col("rk_a") - pl.col("s_a").fill_null(0)).cast(pl.Int16).alias("rk_a"),
    )
    return keep.sort("_r").drop("_r", "s_c", "s_n", "s_a")


def simulate(frac: float, seed: int) -> pl.DataFrame:
    """Build a stage-2-ready train frame with `frac` of S1 entities deleted."""
    tag = f"sim_f{int(frac * 100)}_s{seed}"
    D = deleted_ids(frac, seed)
    print(f"[{tag}] deleting {len(D)} S1 entities", flush=True)
    r = pl.read_parquet(cache("train_retrieval.parquet"))
    isdel = pl.col("s1_id").is_in(D.implode())
    keep = _shift_ranks(r.filter(~isdel), r.filter(isdel))
    del r
    norm_s1 = pl.read_parquet(cache("train_norm.parquet"), columns=NORM1)
    st = build_frame(keep, norm_s1)
    del keep
    model = lgb.Booster(model_file=cache("stage1.lgb"))
    st = st.with_columns(pl.Series("p1", model.predict(st.select(S1_FEATS).to_numpy().astype(np.float32),
                                                       num_threads=160)).cast(pl.Float32))
    c = select_candidates(st).select("rec_id", "s1_id", "p1", *S1_FEATS)
    del st
    c = c.with_columns(
        pl.len().over("rec_id").alias("c_rec_n"),
        pl.len().over("s1_id").alias("c_s1_n"),
        pl.col("p1").rank("ordinal", descending=True).over("rec_id").alias("c_rec_rank"),
        (pl.col("p1") - pl.col("p1").max().over("rec_id")).alias("c_rec_gap"),
        pl.col("p1").rank("ordinal", descending=True).over("s1_id").alias("c_s1_rank"),
        pl.col("p1").sum().over("s1_id").alias("c_s1_psum"),
    )
    # pair features: reuse cached ones, compute the rest
    base = pl.read_parquet(cache("train_feats.parquet"))
    drop = set(S1_FEATS) | set(CTX_COLS) | set(FQ_COLS) | {"p1", "y", "fold"}
    pcols = [x for x in base.columns if x not in drop]
    base = base.select(pcols)
    have = c.select("rec_id", "s1_id").join(base, on=["rec_id", "s1_id"], how="inner")
    miss = c.select("rec_id", "s1_id").join(base.select("rec_id", "s1_id"), on=["rec_id", "s1_id"], how="anti")
    del base
    print(f"[{tag}] candidates {c.height}: cached {have.height}, new {miss.height}", flush=True)
    norm = pl.read_parquet(cache("train_norm.parquet"))
    if miss.height:
        newf = pair_features(miss, norm).select(pcols)
        have = pl.concat([have, newf], how="vertical_relaxed")
    f = c.join(have, on=["rec_id", "s1_id"], how="left")
    # name frequencies over the reduced S1 table
    norm_red = norm.filter(~pl.col("entity_id").is_in(D.implode()))
    f = f.join(name_frequency(norm_red, "rec_id", "fq2"), on="rec_id", how="left")
    f = f.join(name_frequency(norm_red, "s1_id", "fq1"), on="s1_id", how="left")
    f = add_labels(f)
    print(f"[{tag}] frame {f.shape}, positives {f['y'].sum()}", flush=True)
    f.write_parquet(cache(f"train_{tag}_feats.parquet"))
    return f




# --------------------------------------------------------------------------
# Decoy injection: test-like density of near-miss look-alike records
# --------------------------------------------------------------------------
# The test has ~1.3-1.8x more high-scoring near-miss competitors per entity
# (same street, house number within 20, same or near-same name) than train.
# We clone a hash-selected share of TRUE records, shift the house number by
# 1..20 and optionally change the legal form or add a business word. The
# clones are unlabelled records (a merge is a false positive) and go through
# the real pipeline: normalisation, TF-IDF cosines with the training IDF,
# stage-1 features and p1, candidate selection and all context features.

import re

from blocking import hash_view
from preprocess import normalize_frame

_LEGAL_POOL = {"US": ["LLC", "Inc", "Corp", "Co", "LP", "Ltd", "PC"],
               "India": ["Pvt Ltd", "Limited", "Private Limited", "LLP", "Ltd"]}
_WORD_POOL = ["Group", "Holdings", "Services", "Partners", "Enterprises", "Center", "Solutions",
              "International", "Trading", "Associates", "Ventures", "Industries", "Global", "Company"]
_LEGAL_RE = re.compile(r"(?i)[\s,]*\(?\b(llc|inc\.?|corp\.?|corporation|co\.?|lp|ltd\.?|limited|pvt\.?\s*ltd\.?|"
                       r"private limited|llp|pc|p\.c\.|incorporated|company)\)?\.?\s*$")
_NUM_RE = re.compile(r"\d+")


_ABBR_SWAP = [("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Drive", "Dr"), ("Lane", "Ln"),
              ("Court", "Ct"), ("Boulevard", "Blvd"), ("Place", "Pl"), ("Circle", "Cir"), ("Nagar", "Ngr")]


def _surface_noise(name: str, addr: str, rng) -> tuple:
    """Independent record-style noise so a decoy is not a verbatim clone of a true record."""
    if rng.random() < 0.5:
        for a, b in _ABBR_SWAP:
            if a in addr:
                addr = addr.replace(a, b, 1)
                break
            if re.search(rf"\b{b}\b", addr):
                addr = re.sub(rf"\b{b}\b", a, addr, count=1)
                break
    comps = [c.strip() for c in addr.split(",") if c.strip()]
    if len(comps) > 2 and rng.random() < 0.2:
        comps = comps[:-1]
    if len(comps) > 2 and rng.random() < 0.2:
        comps = [comps[-1]] + comps[:-1]
    addr = ", ".join(comps)
    if rng.random() < 0.3:
        addr = addr.upper()
    toks = name.split()
    long = [i for i, t in enumerate(toks) if len(t) >= 4 and t.isalpha()]
    if long and rng.random() < 0.2:
        i = long[int(rng.integers(len(long)))]
        t = toks[i]
        j = int(rng.integers(1, len(t) - 1))
        toks[i] = t[:j] + t[j + 1] + t[j] + t[j + 2:]
    name = " ".join(toks)
    if rng.random() < 0.2:
        name = name.upper()
    return name, addr


def _perturb(name: str, addr: str, country: str, rng, noise: bool = True) -> tuple:
    m = _NUM_RE.search(addr or "")
    if not m:
        return None
    n = int(m.group(0))
    d = int(rng.integers(1, 21)) * (1 if rng.random() < 0.5 or n <= 20 else -1)
    new_addr = addr[:m.start()] + str(max(1, n + d)) + addr[m.end():]
    name = name or ""
    u = rng.random()
    if u < 0.3:
        pool = _LEGAL_POOL.get(country, _LEGAL_POOL["US"])
        base = _LEGAL_RE.sub("", name).strip()
        name = f"{base} {pool[int(rng.integers(len(pool)))]}"
    elif u < 0.6:
        base = _LEGAL_RE.sub("", name).strip()
        tail = name[len(base):]
        name = f"{base} {_WORD_POOL[int(rng.integers(len(_WORD_POOL)))]}{tail}"
    return _surface_noise(name, new_addr, rng) if noise else (name, new_addr)


def _tfidf_rows(X, df, n):
    X = X.tocsr(copy=True)
    idf = (np.log((1 + n) / (1 + df)) + 1.0).astype(np.float32)
    X.data = (1.0 + np.log(X.data)) * idf[X.indices]
    nr = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    nr[nr == 0] = 1
    import scipy.sparse as sp
    return (sp.diags((1 / nr).astype(np.float32)) @ X).tocsr()


def simulate_decoys(rate: float, seed: int, style: str = "noisy") -> pl.DataFrame:
    """style='noisy' (training): independent surface noise, half from the clean S1 text.
    style='clone' (held-out check): verbatim clone of a true record except the house number / name tweak."""
    tag = f"dec_r{int(rate * 100)}_s{seed}" + ("c" if style == "clone" else "")
    rng = np.random.default_rng(seed)
    gt = read_ground_truth_pairs()
    pick = gt.filter((pl.col("rec_id").hash(seed=seed + 100) % 100000) < int(rate * 100000))
    raw = pl.concat([read_source("train", 2), read_source("train", 3)]).join(pick.select("rec_id").rename({"rec_id": "entity_id"}), on="entity_id")
    # half of the decoys start from the entity's clean Source-1 text instead of the record's
    s1 = read_source("train", 1).select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"),
                                        pl.col("business_address").alias("a1"))
    raw = raw.join(pick, left_on="entity_id", right_on="rec_id").join(s1, on="s1_id")
    clone = style == "clone"
    raw = raw if clone else raw.with_columns(
        pl.when((pl.col("entity_id").hash(seed=seed + 7) % 2) == 0).then(pl.col("n1")).otherwise(pl.col("business_name")).alias("business_name"),
        pl.when((pl.col("entity_id").hash(seed=seed + 7) % 2) == 0).then(pl.col("a1")).otherwise(pl.col("business_address")).alias("business_address"))
    rows = []
    for eid, nm, ad, c, src in raw.select("entity_id", "business_name", "business_address", "country", "src").iter_rows():
        p = _perturb(nm, ad, c, rng, noise=not clone)
        if p is not None:
            # decoy ids are unique per simulation tag (two simulations may clone the same record)
            rows.append((eid, f"{eid[:3]}D{seed}{'c' if clone else 'n'}_{eid[3:]}", p[0], p[1], c, src))
    dec = pl.DataFrame(rows, schema=["orig_id", "entity_id", "business_name", "business_address", "country", "src"],
                       orient="row").with_columns(pl.col("src").cast(pl.Int8))
    print(f"[{tag}] {dec.height} decoy records from {pick.height} picked true records", flush=True)
    dnorm = normalize_frame(dec.drop("orig_id"), cache("translit.json"))
    norm = pl.read_parquet(cache("train_norm.parquet"))
    dnorm = dnorm.select(norm.columns)
    dnorm.write_parquet(cache(f"train_{tag}_norm.parquet"))
    # retrieval rows of the clone = the original record's retrieval rows, with recomputed cosines
    r = pl.read_parquet(cache("train_retrieval.parquet"))
    dr = r.join(dec.select(pl.col("orig_id").alias("rec_id"), pl.col("entity_id").alias("new_id")), on="rec_id")
    dr = dr.with_columns(pl.col("new_id").alias("rec_id")).drop("new_id")
    allv = pl.concat([norm, dnorm])
    pos = {e: i for i, e in enumerate(allv["entity_id"].to_list())}
    Xn, Xa = hash_view(allv, "name"), hash_view(allv, "addr")
    carr = np.array(allv["country"].to_list())
    n_orig = norm.height
    qi = np.array([pos[x] for x in dr["rec_id"].to_list()])
    di = np.array([pos[x] for x in dr["s1_id"].to_list()])
    cos_n = np.zeros(len(qi), np.float32)
    cos_a = np.zeros(len(qi), np.float32)
    for c in set(carr[qi]):
        rows_c = np.where(carr[:n_orig] == c)[0]
        sel = np.where(carr[qi] == c)[0]
        for X, out in ((Xn, cos_n), (Xa, cos_a)):
            df = np.bincount(X[rows_c].indices, minlength=X.shape[1]).astype(np.float64)
            need = np.unique(np.concatenate([qi[sel], di[sel]]))
            T = _tfidf_rows(X[need], df, len(rows_c))
            loc = {v: i for i, v in enumerate(need)}
            a = T[[loc[v] for v in qi[sel]]]
            b = T[[loc[v] for v in di[sel]]]
            out[sel] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    dr = dr.with_columns(pl.Series("cos_n", cos_n), pl.Series("cos_a", cos_a))
    r = pl.concat([r, dr.select(r.columns)])
    del Xn, Xa, allv
    st = build_frame(r, pl.concat([norm, dnorm]).select(NORM1))
    del r
    model = lgb.Booster(model_file=cache("stage1.lgb"))
    st = st.with_columns(pl.Series("p1", model.predict(st.select(S1_FEATS).to_numpy().astype(np.float32),
                                                       num_threads=160)).cast(pl.Float32))
    f = _finish(st, pl.concat([norm, dnorm]), None, tag)
    return f


def read_ground_truth_pairs():
    from common import read_ground_truth
    return read_ground_truth()


def _finish(st, norm_all, D, tag):
    """Candidates (p1 >= CAND_THR), context, cached/new pair features, name frequencies, labels."""
    c = select_candidates(st).select("rec_id", "s1_id", "p1", *S1_FEATS)
    c = c.with_columns(
        pl.len().over("rec_id").alias("c_rec_n"),
        pl.len().over("s1_id").alias("c_s1_n"),
        pl.col("p1").rank("ordinal", descending=True).over("rec_id").alias("c_rec_rank"),
        (pl.col("p1") - pl.col("p1").max().over("rec_id")).alias("c_rec_gap"),
        pl.col("p1").rank("ordinal", descending=True).over("s1_id").alias("c_s1_rank"),
        pl.col("p1").sum().over("s1_id").alias("c_s1_psum"),
    )
    base = pl.read_parquet(cache("train_feats.parquet"))
    drop = set(S1_FEATS) | set(CTX_COLS) | set(FQ_COLS) | {"p1", "y", "fold"}
    pcols = [x for x in base.columns if x not in drop]
    base = base.select(pcols)
    have = c.select("rec_id", "s1_id").join(base, on=["rec_id", "s1_id"], how="inner")
    miss = c.select("rec_id", "s1_id").join(base.select("rec_id", "s1_id"), on=["rec_id", "s1_id"], how="anti")
    del base
    print(f"[{tag}] candidates {c.height}: cached {have.height}, new {miss.height}", flush=True)
    if miss.height:
        have = pl.concat([have, pair_features(miss, norm_all).select(pcols)], how="vertical_relaxed")
    f = c.join(have, on=["rec_id", "s1_id"], how="left")
    red = norm_all if D is None else norm_all.filter(~pl.col("entity_id").is_in(D.implode()))
    f = f.join(name_frequency(red, "rec_id", "fq2"), on="rec_id", how="left")
    f = f.join(name_frequency(red, "s1_id", "fq1"), on="s1_id", how="left")
    f = add_labels(f)
    print(f"[{tag}] frame {f.shape}, positives {f['y'].sum()}", flush=True)
    f.write_parquet(cache(f"train_{tag}_feats.parquet"))
    return f


if __name__ == "__main__":
    if sys.argv[1] == "decoys":
        simulate_decoys(float(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "noisy")
    else:
        simulate(float(sys.argv[1]), int(sys.argv[2]))
