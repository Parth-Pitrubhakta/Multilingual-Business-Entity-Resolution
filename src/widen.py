"""Widened final candidate set (v25/v28): re-score a looser pruner rule with the ALREADY-TRAINED stacks.

The shipped rule keeps a record's top-3 S1 by pruner probability with p1 >= 0.02. Here the rule is
top-W_TOPK with p1 >= W_THR (default top-4, p1 >= 0.01). Nothing is retrained: new pairs get pairwise
features, cross-encoder scores from the existing (entity-half cross-fitted) models, and the saved fold
models of each stack re-score the whole widened frame (group features recomputed on the wider set).
Train scores stay strictly out-of-fold: stage-2/3 fold-k models score fold-k S1 rows, target encodings
for fold k are fitted on the ORIGINAL training frame's other folds, cross-encoder half m scores half != m.

  python widen.py feats <train|test>           -> cache/w_<split>_feats.parquet, w_<split>_delta.parquet
  CUDA_VISIBLE_DEVICES=g python widen.py ce <ce|cel|cel2|cem> <split>   -> cache/w_ce_<split>_<variant>.parquet
  MODEL_TAG=.. CE_VARIANTS=.. python widen.py score <split>             -> cache/w_<split>_scores<tag>.parquet
  python widen.py cands <candidate_pairs_top3.tsv> <out.tsv>            -> final candidate_pairs.tsv
"""
import os
import sys
import time

import numpy as np
import polars as pl

from common import cache, read_source

W_THR = float(os.environ.get("W_THR", "0.01"))
W_TOPK = int(os.environ.get("W_TOPK", "4"))
BS = int(os.environ.get("W_BS", "2048"))  # cross-encoder inference batch (lower it when the GPUs are shared)
CE_DIRS = {"ce": "ce", "cel": "ce_large", "cel2": "ce_large2", "cem": "ce_mdeberta"}
CE_SUFFIX = {"ce": "", "cel": "_large", "cel2": "_large2", "cem": "_mdeberta"}


def seen_s1(split):
    """Test: widen only S1 entities of countries present in training (France keeps its shipped candidate set)."""
    s1 = read_source(split, 1).select(pl.col("entity_id").alias("s1_id"), "country")
    if split == "train":
        return s1.select("s1_id")
    seen = read_source("train", 1)["country"].unique().to_list()
    return s1.filter(pl.col("country").is_in(seen)).select("s1_id")


def feats(split):
    from features import pair_features
    from match import name_frequency
    st = pl.read_parquet(cache(f"{split}_stage1.parquet"))
    st = st.join(seen_s1(split), on="s1_id")
    c = (st.with_columns(pl.col("p1").rank("ordinal", descending=True).over("rec_id").alias("_k"))
         .filter((pl.col("p1") >= W_THR) & (pl.col("_k") <= W_TOPK)).drop("_k"))
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
    old = pl.read_parquet(cache(f"{split}_feats.parquet"))
    pcols = [x for x in old.columns if x not in set(c.columns) | {"y", "fold"} or x == "src"]
    pcols = [x for x in pcols if not x.startswith("fq")]
    old = old.select(["rec_id", "s1_id"] + [x for x in pcols if x not in ("rec_id", "s1_id")])
    have = c.select("rec_id", "s1_id").join(old, on=["rec_id", "s1_id"], how="inner")
    miss = c.select("rec_id", "s1_id").join(old.select("rec_id", "s1_id"), on=["rec_id", "s1_id"], how="anti")
    del old
    print(f"{split}: widened {c.height} pairs ({c.height / seen_s1(split).height:.3f}/S1), reusing {have.height}, "
          f"computing {miss.height}", flush=True)
    miss.write_parquet(cache(f"w_{split}_delta.parquet"))
    t0 = time.time()
    have = pl.concat([have, pair_features(miss, norm).select(have.columns)], how="vertical_relaxed")
    print(f"{split}: pair features {time.time() - t0:.0f}s", flush=True)
    f = c.select(keep_cols).join(have, on=["rec_id", "s1_id"], how="left")
    f = f.join(name_frequency(norm, "rec_id", "fq2"), on="rec_id", how="left")
    f = f.join(name_frequency(norm, "s1_id", "fq1"), on="s1_id", how="left").sort(["rec_id", "s1_id"])
    f.write_parquet(cache(f"w_{split}_feats.parquet"))
    print(split, "widened features", f.shape, flush=True)


def ce(variant, split):
    """Cross-encoder scores for the new pairs with the existing models (single GPU)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from cross_encoder import MAX_LEN, _texts, ce_half
    d = pl.read_parquet(cache(f"w_{split}_delta.parquet")).with_columns(ce_half().alias("half"))
    cols = ["entity_id", "business_name", "business_address"]
    tx = _texts(pl.read_parquet(cache(f"{split}_norm.parquet"), columns=cols))
    d = d.join(tx.rename({"entity_id": "s1_id", "t": "a"}), on="s1_id").join(tx.rename({"entity_id": "rec_id", "t": "b"}), on="rec_id")
    t0 = time.time()
    scores = {}
    for m in (0, 1):
        sh = d.filter(pl.col("half") != m) if split == "train" else d  # train: out-of-fold half only
        mdir = cache(os.path.join(CE_DIRS[variant], f"model_{m}"))
        tok = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForSequenceClassification.from_pretrained(mdir).cuda().eval()
        enc = tok(sh["a"].to_list(), sh["b"].to_list(), truncation=True, max_length=MAX_LEN)["input_ids"]
        order = np.argsort([len(x) for x in enc])
        out = np.zeros(len(enc), np.float32)
        with torch.no_grad():
            for s in range(0, len(order), BS):
                idx = order[s:s + BS]
                L = max(len(enc[i]) for i in idx)
                ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(idx), L), dtype=torch.long)
                for r, j in enumerate(idx):
                    ids[r, :len(enc[j])] = torch.tensor(enc[j])
                    att[r, :len(enc[j])] = 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = model(input_ids=ids.cuda(non_blocking=True), attention_mask=att.cuda(non_blocking=True)).logits.squeeze(-1)
                out[idx] = lo.float().cpu().numpy()
        scores[m] = sh.select("rec_id", "s1_id").with_columns(pl.Series(f"ce{m}", out))
        del model
        torch.cuda.empty_cache()
        print(f"[{variant} {split}] model {m}: {sh.height} pairs, {time.time() - t0:.0f}s", flush=True)
    if split == "train":
        s = pl.concat([scores[0].rename({"ce0": "ce"}), scores[1].rename({"ce1": "ce"})])
    else:
        s = scores[0].join(scores[1], on=["rec_id", "s1_id"]).select("rec_id", "s1_id", ((pl.col("ce0") + pl.col("ce1")) / 2).alias("ce"))
    s.write_parquet(cache(f"w_ce_{split}_{variant}.parquet"))
    print(f"[{variant} {split}] DONE {s.height} pairs {time.time() - t0:.0f}s", flush=True)


def attach_ce_w(f, split, variants):
    for var in variants:
        base = pl.read_parquet(cache(f"ce_scores_{split}{CE_SUFFIX[var]}.parquet")).unique(["rec_id", "s1_id"], keep="first")
        news = [pl.read_parquet(cache(p)).select("rec_id", "s1_id", "ce") for p in [f"w_ce_{split}_{var}.parquet"] if os.path.exists(cache(p))]
        allce = pl.concat([base.select("rec_id", "s1_id", "ce")] + news, how="vertical_relaxed") \
                  .unique(["rec_id", "s1_id"], keep="first").rename({"ce": var})
        f = f.join(allce, on=["rec_id", "s1_id"], how="left", maintain_order="left")
        f = f.with_columns(
            (pl.col(var) - pl.col(var).max().over("rec_id")).alias(f"{var}_gap"),
            pl.col(var).rank("ordinal", descending=True).over("rec_id").cast(pl.Int32).alias(f"{var}_rank"),
            (pl.col(var) - pl.col(var).filter(pl.col(var) < pl.col(var).max()).max().over("rec_id").fill_null(-20)).alias(f"{var}_margin"),
        )
    return f


def te_oof_from_orig(f, normk):
    """Train target encodings for the widened frame: fold k encoded with tables fitted on the ORIGINAL
    frame's labelled keys of the other folds (exactly the tables a widened inference would see)."""
    import match as M
    from stage1 import fold_of
    keys = M.te_keys(f, normk)
    fit = pl.read_parquet(cache("train_te_keys.parquet")).with_columns(fold_of().alias("fold"))
    prior = float(fit["y"].mean())
    keys = keys.with_columns(f["fold"])
    parts = []
    for k in range(M.NFOLD):
        trk = fit.filter(pl.col("fold") != k)
        vak = keys.with_row_index("_r").filter(pl.col("fold") == k)
        enc = [M._te_apply(vak, M._te_table(trk, trk["y"], c, prior), c, prior) for c in M.TE_COLS]
        parts.append(pl.concat([vak.select("_r")] + enc, how="horizontal"))
    out = pl.concat(parts).sort("_r").drop("_r")
    tok = M.token_features(keys.drop("fold"), normk)
    if not M.TOKSTATS:
        return pl.concat([f, out, tok], how="horizontal")
    hn = f["hnr_eq"] if "hnr_eq" in f.columns else pl.Series("hnr_eq", np.full(f.height, -1.0, np.float32))
    return pl.concat([f, out, tok, M.token_hn_stats(keys.drop("fold"), hn)], how="horizontal")


def score(split):
    import json
    import lightgbm as lgb
    import match as M
    cols = json.load(open(cache(f"{M.MODEL_TAG}cols.json")))
    f = pl.read_parquet(cache(f"w_{split}_feats.parquet"))
    t0 = time.time()
    if split == "train":
        normk = M._norm_with_aug(M.NC)
        f = M.add_labels(f)
        f = te_oof_from_orig(f, normk)
    else:
        normk = pl.read_parquet(cache(f"{split}_norm.parquet"), columns=M.NC)
        f = M.te_features(f, normk, fit=pl.read_parquet(cache("train_te_keys.parquet")))
    del normk
    f = attach_ce_w(f, split, M.CE_VARIANTS)
    print(f"{M.MODEL_TAG} {split}: frame {f.shape} {time.time() - t0:.0f}s", flush=True)
    boosters = lambda name: [lgb.Booster(model_file=cache(f"{M.MODEL_TAG}{name}_fold{k}.lgb")) for k in range(M.NFOLD)]

    def apply(X, name):
        bs = boosters(name)
        if split == "train":  # out-of-fold: fold-k model scores fold-k S1 rows
            fold = f["fold"].to_numpy()
            p = np.zeros(len(X), np.float32)
            for k in range(M.NFOLD):
                va = fold == k
                p[va] = bs[k].predict(X[va], num_threads=int(os.environ.get("LGB_THREADS", "48")))
            return p
        return np.mean([b.predict(X, num_threads=int(os.environ.get("LGB_THREADS", "48"))) for b in bs], axis=0).astype(np.float32)

    X = f.select(cols["cols2"]).to_numpy().astype(np.float32)
    f = f.with_columns(pl.Series("p2", apply(X, "stage2")))
    norm = M._norm_with_aug(M.NG) if split == "train" else pl.read_parquet(cache(f"{split}_norm.parquet"), columns=M.NG)
    f = M.twin_features(M.group_features(f, "p2", norm), "p2", norm)
    X = f.select(cols["cols3"]).to_numpy().astype(np.float32)
    f = f.with_columns(pl.Series("p3", apply(X, "stage3")))
    keep = ["rec_id", "s1_id", "p1", "p2", "p3"] + (["y", "fold"] if split == "train" else [])
    f.select(keep).write_parquet(cache(f"w_{split}_scores{M.MODEL_TAG}.parquet"))
    print(f"{M.MODEL_TAG} {split}: scored {f.height} pairs, {time.time() - t0:.0f}s", flush=True)


def cands(sub_cand, out_cand):
    """Final candidate_pairs.tsv: seen countries = the widened set the stacks scored (w_test_feats),
    other countries (France) = their shipped candidate rows, unchanged."""
    s1 = read_source("test", 1).select(pl.col("entity_id").alias("s1_id"))
    w = pl.read_parquet(cache("w_test_feats.parquet"), columns=["rec_id", "s1_id"])
    old = pl.read_csv(sub_cand, separator="\t", quote_char=None, infer_schema=False)
    old = (old.with_columns(pl.col("candidate_entity_ids").fill_null("").str.split(",")).explode("candidate_entity_ids")
           .filter(pl.col("candidate_entity_ids") != "")
           .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("candidate_entity_ids").alias("rec_id")))
    keep = old.join(seen_s1("test"), on="s1_id", how="anti")
    allc = pl.concat([keep.select("s1_id", "rec_id"), w.select("s1_id", "rec_id")])
    g = allc.group_by("s1_id").agg(pl.col("rec_id").sort().str.join(",").alias("candidate_entity_ids"))
    out = s1.join(g, on="s1_id", how="left").with_columns(pl.col("candidate_entity_ids").fill_null(""))
    out.rename({"s1_id": "source1_entity_id"}).write_csv(out_cand, separator="\t", quote_style="never")
    print(f"candidates: {allc.height} pairs, {allc.height / s1.height:.3f}/S1 -> {out_cand}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "feats":
        feats(sys.argv[2])
    elif cmd == "ce":
        ce(sys.argv[2], sys.argv[3])
    elif cmd == "score":
        score(sys.argv[2])
    elif cmd == "cands":
        cands(sys.argv[2], sys.argv[3])
