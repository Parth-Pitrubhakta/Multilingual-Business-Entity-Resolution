"""France post-processing: from the combined submission (ruleset v12, combine.py) to the final one.

France is absent from training. The steps below correct the two stacks' French decisions using
label-free structure of the data generator (see Documentation_template.md, section 11):

  v13  drop accepted same-address one-word swaps to mid-rate generic words: French descriptor swaps
       (club <-> amicale <-> comite ...) are same-address decoys, as real-word swaps are in training (0-15% true)
  v15  add "groupe" noise-word matches: France's appended noise words are "services" and "groupe"
       (55% of their same-address swaps sit after the legal form, like US center/services/partners);
       the stacks read "groupe" as the US decoy word "group"
  v16  add matches in vocabulary-free pair types with high training truth (typecal.py)
  v17  add/remove using the France-adapted cross-encoder (ce_france.py, round 1) gated by the type prior
  v18  same with the round-2 cross-encoder
  v19  entity-aware gating (break-even precision is lower for entities with <= 1 match) and
       unique-name name-only records
  v20  records whose name was replaced by a random token, at an address held by exactly one candidate
       Source-1 entity (training truth 0.89-0.996; ambiguous when two entities share the address)
  v21  the full stack re-scored on France with French target encodings and the French cross-encoder
       (predict_france.py); its arg-max pairs >= threshold are added only where the type prior agrees
  v24  tightening back to the leaderboard-confirmed v17 state: the leaderboard gain of v23 over v17 (+0.00012)
       is fully explained by the US/India blend (+0.00014 to +0.00028 on the held-out simulations), so the
       post-v17 French additions were net break-even or worse. Keep v17, the post-v17 removals, and only the
       post-v17 additions that both cross-encoder rounds (q >= 0.9) and the type prior (>= 0.9) confirm on the
       same street, excluding name-only and near-house-number records (France's decoy generator places
       look-alikes next door)

usage:
  python france_rules.py rules     <v12.tsv> <v16.tsv>                      (v13, v15, v16)
  python france_rules.py selftrain <base.tsv> <ce_scores.parquet> <out.tsv> (v17 from v16 / v18 from v17)
  python france_rules.py final     <v18.tsv> <ce_scores.parquet> <out.tsv>  (v19, v20)
  python france_rules.py rescored  <v20.tsv> <ce_scores.parquet> <rescored.parquet> <out.tsv>  (v21)
  python france_rules.py tighten   <v17.tsv> <v21.tsv> <ce_scores1.parquet> <ce_scores2.parquet> <out.tsv>  (v24)
"""
import os
import sys

import polars as pl

from combine import word_rates
from common import cache, read_source


def read_pairs(path):
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "")
            .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").alias("rec_id")))


def write_pairs(pairs, path):
    assert pairs.group_by("rec_id").len()["len"].max() == 1, "a record matched to two entities"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    g = {}
    for s, r in pairs.select("s1_id", "rec_id").iter_rows():
        g.setdefault(s, []).append(r)
    with open(path, "w") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s in read_source("test", 1)["entity_id"].to_list():
            fh.write(f"{s}\t{','.join(g.get(s, []))}\n")


def _french_s1():
    return read_source("test", 1).filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1_id"))


def _tokens():
    tn = pl.read_parquet(cache("test_norm.parquet"), columns=["entity_id", "n_core"])
    return tn.select("entity_id", pl.col("n_core").str.split(" ").alias("t"))


def _check_candidates(add):
    cand = pl.read_parquet(cache("test_scoresfull_.parquet"), columns=["rec_id", "s1_id"])
    assert add.join(cand, on=["rec_id", "s1_id"], how="anti").height == 0, "pair outside the candidate set"


def french_scores():
    """French candidate pairs with both stacks' scores (p8 = full stack, p10 = vocabulary-free stack)."""
    s10 = pl.read_parquet(cache("test_scoresvf_.parquet"), columns=["rec_id", "s1_id", "p3"]).rename({"p3": "p10"})
    s8 = pl.read_parquet(cache("test_scoresfull_.parquet"), columns=["rec_id", "s1_id", "p3"]).rename({"p3": "p8"})
    return s8.join(s10, on=["rec_id", "s1_id"]).join(_french_s1(), on="s1_id")


# ---------------------------------------------------------------- v13
def step_v13(base):
    ft = pl.read_parquet(cache("test_feats.parquet"), columns=["rec_id", "s1_id", "hnr_eq", "ad_tset"])
    t = _tokens()
    x = base.join(_french_s1(), on="s1_id").join(ft, on=["rec_id", "s1_id"])
    x = x.join(t.rename({"entity_id": "s1_id", "t": "t1"}), on="s1_id").join(t.rename({"entity_id": "rec_id", "t": "t2"}), on="rec_id")
    x = x.with_columns(pl.col("t2").list.set_difference(pl.col("t1")).alias("ex"), pl.col("t1").list.set_difference(pl.col("t2")).alias("mi"))
    x = x.with_columns(pl.col("ex").list.first().alias("w"), pl.lit("France").alias("country"))
    x = x.join(word_rates().select("country", "w", "rate"), on=["country", "w"], how="left")
    same = (pl.col("hnr_eq") == 1) & (pl.col("ad_tset") >= 95)
    rm = x.filter(same & (pl.col("ex").list.len() == 1) & (pl.col("mi").list.len() == 1) & (pl.col("rate") >= 0.3) & (pl.col("rate") < 0.7))
    out = base.join(rm.select("s1_id", "rec_id"), on=["s1_id", "rec_id"], how="anti")
    print(f"v13 = {base.height} - {rm.height} French same-address mid-rate word swaps = {out.height}", flush=True)
    return out


# ---------------------------------------------------------------- v15
def step_v15(base):
    f = pl.read_parquet(cache("test_feats.parquet"), columns=["rec_id", "s1_id", "hnr_eq", "ad_tset", "hn_rel", "ad_empty2", "lg_conflict"])
    n = pl.read_parquet(cache("test_norm.parquet"), columns=["entity_id", "n_core", "country"])
    t = n.select("entity_id", "country", pl.col("n_core").str.split(" ").alias("t"))
    f = f.join(t.select(pl.col("entity_id").alias("rec_id"), pl.col("t").alias("t2")), on="rec_id").filter(pl.col("t2").list.contains("groupe"))
    f = f.join(t.rename({"entity_id": "s1_id", "t": "t1"}), on="s1_id").filter(pl.col("country") == "France")
    f = f.with_columns(pl.col("t2").list.set_difference(pl.col("t1")).alias("ex"), pl.col("t1").list.set_difference(pl.col("t2")).alias("mi"))
    clean = (pl.col("ex").list.len() == 1) & (pl.col("ex").list.first() == "groupe") & (pl.col("mi").list.len() <= 1)
    same = (pl.col("hnr_eq") == 1) & (pl.col("ad_tset") >= 95)
    f = f.filter(clean & (pl.col("ad_empty2") != 1)).with_columns(
        pl.when(same).then(0).when(pl.col("hnr_eq") == 1).then(1).when(pl.col("hn_rel") == -1).then(2)
        .when(pl.col("hn_rel").is_in([1, 2, 3])).then(3).otherwise(9).alias("arank")).filter(pl.col("arank") <= 3)
    f = f.join(french_scores().select("rec_id", "s1_id", "p10"), on=["rec_id", "s1_id"], how="left")
    _check_candidates(f)
    free = f.join(base.select("rec_id"), on="rec_id", how="anti")
    free = free.sort(["rec_id", "arank", "p10"], descending=[False, False, True]).unique("rec_id", keep="first")
    out = pl.concat([base, free.select("s1_id", "rec_id")])
    print(f"v15 = {base.height} + {free.height} French 'groupe' noise-word matches = {out.height}", flush=True)
    return out


# ---------------------------------------------------------------- type tables
def type_tables():
    if not (os.path.exists(cache("typecal_train.parquet")) and os.path.exists(cache("typecal_test.parquet"))):
        from typecal import typed
        from common import read_ground_truth
        tr = typed("train").join(read_ground_truth().with_columns(pl.lit(1).alias("y")), on=["rec_id", "s1_id"], how="left").with_columns(pl.col("y").fill_null(0))
        tr.write_parquet(cache("typecal_train.parquet"))
        typed("test", countries=["France", "US"]).write_parquet(cache("typecal_test.parquet"))
    return pl.read_parquet(cache("typecal_train.parquet")), pl.read_parquet(cache("typecal_test.parquet"))


def _flags(te, base):
    te = te.join(base.with_columns(pl.lit(1).alias("acc")), on=["s1_id", "rec_id"], how="left").with_columns(pl.col("acc").fill_null(0))
    return te.join(base.select("rec_id", pl.lit(1).alias("used")), on="rec_id", how="left").with_columns(pl.col("used").fill_null(0))


# ---------------------------------------------------------------- v16
def step_v16(base):
    tr, te = type_tables()
    T = tr.group_by("type").agg(pl.len().alias("tr_n"), pl.col("y").mean().alias("tr_y"))
    te = _flags(te, base).join(T, on="type", how="left")
    us = te.filter(pl.col("country") == "US").group_by("type").agg(pl.len().alias("us_n"), pl.col("acc").mean().alias("us_acc"))
    fr = te.filter(pl.col("country") == "France").join(us, on="type", how="left")
    n = pl.read_parquet(cache("test_norm.parquet"), columns=["entity_id", "n_core", "country"]).filter(pl.col("country") == "France")
    tokf = n.select(pl.col("n_core").str.split(" ").alias("t")).explode("t").group_by("t").len().rename({"t": "w", "len": "wfreq"})
    t = n.select("entity_id", pl.col("n_core").str.split(" ").alias("t"))
    fr = fr.join(t.rename({"entity_id": "s1_id", "t": "t1"}), on="s1_id").join(t.rename({"entity_id": "rec_id", "t": "t2"}), on="rec_id")
    fr = fr.with_columns(pl.col("t2").list.set_difference(pl.col("t1")).list.first().alias("w")).join(tokf, on="w", how="left")
    typo_ok = ~pl.col("nm").str.contains("typo") | (pl.col("wfreq").fill_null(0) <= 30)
    rel = ["same", "drop1", "swap-typo", "multi-typo", "add-typo"]
    ok = (pl.col("nm").is_in(rel) & typo_ok & (pl.col("ctok") == "") & (pl.col("tr_y") >= 0.9) & (pl.col("tr_n") >= 1000)
          & (pl.col("us_acc") >= 0.8) & (pl.col("us_n") >= 300) & (pl.col("street") == "st=") & (pl.col("addr") != "name-only"))
    free = fr.filter((pl.col("acc") == 0) & (pl.col("used") == 0))
    top = free.sort(["rec_id", "tr_y", "p1"], descending=[False, True, True]).group_by("rec_id", maintain_order=True).agg(pl.all().first(), pl.col("tr_y").alias("all_y"))
    top = top.with_columns(pl.col("all_y").list.get(1, null_on_oob=True).alias("second_y"))
    add = top.filter(ok).filter(pl.col("second_y").is_null() | (pl.col("second_y") < 0.5))
    _check_candidates(add)
    out = pl.concat([base, add.select("s1_id", "rec_id")])
    print(f"v16 = {base.height} + {add.height} French vocabulary-free type matches = {out.height}", flush=True)
    return out


# ---------------------------------------------------------------- cross-encoder joined table
def cefr_table(base, scores_path):
    s = pl.read_parquet(scores_path).with_columns((1 / (1 + (-pl.col("cefr")).exp())).alias("q"))
    tr, te = type_tables()
    te = te.filter(pl.col("country") == "France")
    T = tr.group_by("type").agg(pl.len().alias("tr_n"), pl.col("y").mean().alias("tr_y"))
    te = te.join(s, on=["rec_id", "s1_id"]).join(T, on="type", how="left").join(french_scores().select("rec_id", "s1_id", "p8", "p10"), on=["rec_id", "s1_id"], how="left")
    te = _flags(te, base)
    T2 = tr.group_by("type").agg(pl.len().alias("tr_n2"), pl.col("y").mean().alias("tr_y2"))
    te = te.with_columns(pl.concat_str([pl.col("addr"), pl.col("legal"), pl.col("street"), pl.col("nm").str.replace("hirate", "noise"), pl.col("ctok")], separator="|").alias("type2"))
    return te.join(T2.rename({"type": "type2"}), on="type2", how="left")


def _top_free(te):
    free = te.filter((pl.col("acc") == 0) & (pl.col("used") == 0))
    top = free.sort(["rec_id", "q"], descending=[False, True]).group_by("rec_id", maintain_order=True).agg(pl.all().first(), pl.col("q").alias("qs"))
    return top.with_columns(pl.col("qs").list.get(1, null_on_oob=True).alias("q2"))


def _name_uniqueness():
    n = pl.read_parquet(cache("test_norm.parquet"), columns=["entity_id", "n_core", "country"])
    return (n.filter(pl.col("entity_id").str.starts_with("S1-")).with_columns(pl.len().over(["country", "n_core"]).alias("n_same_core"))
            .select(pl.col("entity_id").alias("s1_id"), "n_same_core"))


# ---------------------------------------------------------------- v17 / v18
def step_selftrain(base, scores_path):
    te = cefr_table(base, scores_path).join(_name_uniqueness(), on="s1_id", how="left")
    top = _top_free(te)
    nameonly_ok = (pl.col("addr") != "name-only") | ((pl.col("n_same_core") == 1) & pl.col("legal").is_in(["none", "one", "same"]))
    ok = ((pl.col("q") >= 0.9) & (pl.col("q2").is_null() | (pl.col("q2") < 0.5)) & (pl.col("tr_y2") >= 0.8) & (pl.col("tr_n2") >= 200)
          & ((pl.col("ctok") == "") | ((pl.col("ctok") == "+C") & (pl.col("addr") == "same-addr"))) & (pl.col("street") != "st!") & nameonly_ok)
    add = top.filter(ok)
    rm = te.filter((pl.col("acc") == 1) & (pl.col("q") < 0.1) & (pl.col("tr_y2") < 0.5))
    _check_candidates(add)
    out = pl.concat([base.join(rm.select("s1_id", "rec_id"), on=["s1_id", "rec_id"], how="anti"), add.select("s1_id", "rec_id")])
    print(f"self-training step = {base.height} - {rm.height} + {add.height} = {out.height}", flush=True)
    return out


# ---------------------------------------------------------------- v19
def step_v19(base, scores_path):
    te = cefr_table(base, scores_path).join(_name_uniqueness().rename({"n_same_core": "nsc"}), on="s1_id", how="left")
    te = te.join(base.group_by("s1_id").len().rename({"len": "m"}), on="s1_id", how="left").with_columns(pl.col("m").fill_null(0))
    top = _top_free(te)
    base_ok = ((pl.col("q2").is_null() | (pl.col("q2") < 0.5)) & ((pl.col("ctok") == "") | ((pl.col("ctok") == "+C") & (pl.col("addr") == "same-addr")))
               & (pl.col("street") != "st!"))
    typed_ok = ((pl.col("addr") != "name-only") & (pl.col("q") >= 0.9) & (pl.col("tr_n2") >= 200)
                & ((pl.col("tr_y2") >= 0.8) | ((pl.col("m") <= 1) & (pl.col("tr_y2") >= 0.65))))
    uniq = ((pl.col("addr") == "name-only") & (pl.col("nm") == "same") & (pl.col("nsc") == 1) & (pl.col("q") >= 0.95)
            & pl.col("legal").is_in(["none", "one", "same"]))
    add = top.filter(base_ok & (typed_ok | uniq))
    _check_candidates(add)
    out = pl.concat([base, add.select("s1_id", "rec_id")])
    print(f"v19 = {base.height} + {add.height} = {out.height}", flush=True)
    return out


# ---------------------------------------------------------------- v20
def step_v20(base, scores_path):
    te = cefr_table(base, scores_path)
    te = te.with_columns((pl.col("addr") == "same-addr").cast(pl.Int32).sum().over("rec_id").alias("n_sameaddr_s1"))
    add = te.filter((pl.col("acc") == 0) & (pl.col("used") == 0) & (pl.col("addr") == "same-addr")
                    & pl.col("nm").is_in(["multi-rare", "swap-rare", "add-rare"]) & (pl.col("n_sameaddr_s1") == 1)
                    & (pl.col("q") >= 0.9) & (pl.col("ctok") == "") & (pl.col("street") == "st="))
    add = add.sort(["rec_id", "q"], descending=[False, True]).unique("rec_id", keep="first")
    _check_candidates(add)
    out = pl.concat([base, add.select("s1_id", "rec_id")])
    print(f"v20 = {base.height} + {add.height} random-name same-address matches = {out.height}", flush=True)
    return out


# ---------------------------------------------------------------- v21
def step_v21(base, scores_path, rescored_path, thr=0.725):
    te = cefr_table(base, scores_path).join(_name_uniqueness(), on="s1_id", how="left")
    s = pl.read_parquet(rescored_path).select("rec_id", "s1_id", pl.col("p3").alias("pf"))
    te = te.join(s, on=["rec_id", "s1_id"], how="left").with_columns((pl.col("pf") == pl.col("pf").max().over("rec_id")).alias("isbest"))
    nameonly_ok = (pl.col("addr") != "name-only") | ((pl.col("n_same_core") == 1) & pl.col("legal").is_in(["none", "one", "same"]))
    decoy = (pl.col("nm").str.contains("desc") | ((pl.col("ctok") == "+C") & (pl.col("addr") != "same-addr"))
             | ((pl.col("addr") == "near") & pl.col("legal").is_in(["diff", "one"])))
    add = te.filter((pl.col("acc") == 0) & (pl.col("used") == 0) & pl.col("isbest") & (pl.col("pf") >= thr)
                    & (pl.col("tr_y2") >= 0.85) & (pl.col("tr_n2") >= 200) & nameonly_ok & ~decoy & (pl.col("street") != "st!"))
    add = add.unique("rec_id", keep="first")
    rm = te.filter((pl.col("acc") == 1) & (pl.col("pf") < 0.2) & (pl.col("tr_y2") < 0.5))
    _check_candidates(add)
    out = pl.concat([base.join(rm.select("s1_id", "rec_id"), on=["s1_id", "rec_id"], how="anti"), add.select("s1_id", "rec_id")])
    print(f"v21 = {base.height} - {rm.height} + {add.height} (re-scored full stack, gated) = {out.height}", flush=True)
    return out


# ---------------------------------------------------------------- v24
def step_tighten(v17, v21, ce1_path, ce2_path):
    add = v21.join(v17, on=["s1_id", "rec_id"], how="anti")
    rm = v17.join(v21, on=["s1_id", "rec_id"], how="anti")
    q = [pl.read_parquet(p).select("rec_id", "s1_id", (1 / (1 + (-pl.col("cefr")).exp())).alias(f"q{i}"))
         for i, p in enumerate((ce1_path, ce2_path), 1)]
    tr, te = type_tables()
    T = tr.group_by("type").agg(pl.col("y").mean().alias("tr_y2"))
    te = te.filter(pl.col("country") == "France").with_columns(
        pl.concat_str([pl.col("addr"), pl.col("legal"), pl.col("street"), pl.col("nm").str.replace("hirate", "noise"), pl.col("ctok")],
                      separator="|").alias("type2")).join(T.rename({"type": "type2"}), on="type2", how="left")
    a = add.join(te, on=["s1_id", "rec_id"], how="left").join(q[0], on=["rec_id", "s1_id"], how="left").join(q[1], on=["rec_id", "s1_id"], how="left")
    strict = ((pl.col("q1") >= 0.9) & (pl.col("q2") >= 0.9) & (pl.col("tr_y2") >= 0.9) & (pl.col("street") == "st=")
              & (pl.col("ctok") == "") & ~pl.col("addr").is_in(["name-only", "near"])).fill_null(False)
    keep = a.filter(strict).select("s1_id", "rec_id")
    out = pl.concat([v17.join(rm, on=["s1_id", "rec_id"], how="anti"), keep])
    print(f"v24 France = v17 {v17.height} - {rm.height} post-v17 removals + {keep.height} of {add.height} post-v17 additions "
          f"(strict agreement) = {out.height}", flush=True)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rules":
        v = read_pairs(sys.argv[2])
        v = step_v16(step_v15(step_v13(v)))
        write_pairs(v, sys.argv[3])
    elif cmd == "selftrain":
        write_pairs(step_selftrain(read_pairs(sys.argv[2]), sys.argv[3]), sys.argv[4])
    elif cmd == "final":
        v = step_v19(read_pairs(sys.argv[2]), sys.argv[3])
        write_pairs(step_v20(v, sys.argv[3]), sys.argv[4])
    elif cmd == "tighten":
        write_pairs(step_tighten(read_pairs(sys.argv[2]), read_pairs(sys.argv[3]), sys.argv[4], sys.argv[5]), sys.argv[6])
    elif cmd == "rescored":
        write_pairs(step_v21(read_pairs(sys.argv[2]), sys.argv[3], sys.argv[4]), sys.argv[5])
