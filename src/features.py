"""Pairwise features for (Source 1 entity, Source 2/3 record) candidate pairs.

String similarities are computed with rapidfuzz.process.cpdist (C++,
multi-threaded, element-wise over the pair list). Set-style features
(token containment, number matching, legal-form conflicts, acronyms, DBA /
website handling) are computed in a process pool. All features are
country-agnostic; `country` itself is never used as a feature, so the model
transfers to countries absent from training (France).
"""
from multiprocessing import Pool

import re

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

NORM_COLS = ["business_address", "n_full", "n_core", "n_main", "n_alt", "n_legal", "n_web", "n_skel", "n_acr",
             "n_nonlatin", "n_isweb", "a_tok", "a_nums", "a_state", "a_postal", "a_city", "a_comps"]


def attach(pairs: pl.DataFrame, norm: pl.DataFrame) -> pl.DataFrame:
    """Join normalised fields of both sides: suffix _1 for S1, _2 for the record."""
    n1 = norm.select(pl.col("entity_id").alias("s1_id"), *[pl.col(c).alias(c + "_1") for c in NORM_COLS])
    n2 = norm.select(pl.col("entity_id").alias("rec_id"), "src", *[pl.col(c).alias(c + "_2") for c in NORM_COLS])
    return pairs.join(n1, on="s1_id", how="left").join(n2, on="rec_id", how="left")


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw).astype(np.float32)


def string_features(df: pl.DataFrame) -> dict:
    f = {}
    g = lambda c: df[c].fill_null("").to_list()
    c1, c2 = g("n_core_1"), g("n_core_2")
    f1, f2 = g("n_full_1"), g("n_full_2")
    a1, a2 = g("a_tok_1"), g("a_tok_2")
    k1, k2 = g("n_skel_1"), g("n_skel_2")
    m2 = g("n_main_2")
    alt2 = g("n_alt_2")
    web2 = g("n_web_2")
    city2 = g("a_city_2")
    city1 = g("a_city_1")
    nos1 = [s.replace(" ", "") for s in c1]
    nos2 = [s.replace(" ", "") for s in c2]
    f["nm_ratio"] = _cp(c1, c2, fuzz.ratio)
    f["nm_tsort"] = _cp(c1, c2, fuzz.token_sort_ratio)
    f["nm_tset"] = _cp(c1, c2, fuzz.token_set_ratio)
    f["nm_partial"] = _cp(c1, c2, fuzz.partial_ratio)
    f["nm_jw"] = _cp(c1, c2, JaroWinkler.normalized_similarity)
    f["nm_lev"] = _cp(c1, c2, Levenshtein.distance)
    f["nm_full_ratio"] = _cp(f1, f2, fuzz.ratio)
    f["nm_full_tsort"] = _cp(f1, f2, fuzz.token_sort_ratio)
    f["nm_skel_tsort"] = _cp(k1, k2, fuzz.token_sort_ratio)
    f["nm_skel_tset"] = _cp(k1, k2, fuzz.token_set_ratio)
    f["nm_nospace_ratio"] = _cp(nos1, nos2, fuzz.ratio)
    f["nm_nospace_partial"] = _cp(nos1, nos2, fuzz.partial_ratio)
    f["nm_main_tset"] = _cp(c1, m2, fuzz.token_set_ratio)
    f["nm_alt_tset"] = _cp(c1, alt2, fuzz.token_set_ratio)
    f["nm_web_ratio"] = _cp(nos1, web2, fuzz.ratio)
    f["nm_web_partial"] = _cp(nos1, web2, fuzz.partial_ratio)
    f["ad_ratio"] = _cp(a1, a2, fuzz.ratio)
    f["ad_tsort"] = _cp(a1, a2, fuzz.token_sort_ratio)
    f["ad_tset"] = _cp(a1, a2, fuzz.token_set_ratio)
    f["ad_partial_tset"] = _cp(a1, a2, fuzz.partial_token_set_ratio)
    f["ad_city_in"] = _cp(city2, a1, fuzz.partial_ratio)
    f["ad_city_in_r"] = _cp(city1, a2, fuzz.partial_ratio)
    # composite house number ('2-37-2' vs '2-37-6')
    h1 = [house_number_raw(x) for x in g("business_address_1")]
    h2 = [house_number_raw(x) for x in g("business_address_2")]
    both = np.array([bool(a) and bool(b) for a, b in zip(h1, h2)])
    f["hnr_eq"] = np.where(both, np.array([a == b for a, b in zip(h1, h2)], dtype=np.float32), -1).astype(np.float32)
    f["hnr_ratio"] = np.where(both, _cp(h1, h2, fuzz.ratio), -1).astype(np.float32)
    f["hnr_in"] = np.where(both, np.array([(a in b) or (b in a) for a, b in zip(h1, h2)], dtype=np.float32), -1).astype(np.float32)
    for k in ("nm_web_ratio", "nm_web_partial"):
        f[k][np.array([not w for w in web2])] = -1
    f["nm_alt_tset"][np.array([not w for w in alt2])] = -1
    return f


# --------------------------------------------------------------------------
# set-based features (pure python, run in a process pool)
# --------------------------------------------------------------------------

def _fuzzy_contain(a, b, thr=0.88):
    """Fraction of tokens of a that have a close (JW >= thr) token in b."""
    if not a:
        return -1.0
    if not b:
        return 0.0
    hit = 0
    bs = set(b)
    for t in a:
        if t in bs:
            hit += 1
            continue
        best = 0.0
        for u in b:
            s = JaroWinkler.normalized_similarity(t, u)
            if s > best:
                best = s
        if best >= thr:
            hit += 1
    return hit / len(a)


def _num_close(x, y):
    """Numbers equal up to one dropped/added leading digit or one edit."""
    if x == y:
        return True
    if len(x) >= 2 and len(y) >= 2 and (x.endswith(y) or y.endswith(x)):
        return True
    return len(x) == len(y) and len(x) >= 3 and Levenshtein.distance(x, y) <= 1


_REL_ORDER = {"eq": 0, "suffix": 1, "prefix": 2, "indel1": 3, "sub1": 4, "near": 5, "other": 6}


def _num_rel(a: str, b: str) -> str:
    """Relation between two number strings. True matches are mostly equal or
    lose a digit (suffix/prefix); distractor records sit at a *nearby* house
    number (+-20) or differ by one substituted digit."""
    if a == b:
        return "eq"
    if a.endswith(b) or b.endswith(a):
        return "suffix"
    if a.startswith(b) or b.startswith(a):
        return "prefix"
    d = Levenshtein.distance(a, b)
    if d == 1 and len(a) != len(b):
        return "indel1"
    if d == 1:
        return "sub1"
    if abs(int(a[:9]) - int(b[:9])) <= 20:
        return "near"
    return "other"


_HN_RAW = re.compile(r"[#(]*\s*([0-9][0-9a-z]*(?:\s*[-/]\s*[0-9a-z]+)*)", re.I)


def house_number_raw(addr: str) -> str:
    """Full composite house number of the first number-bearing token, e.g.
    'H.No 2-37-2, ...' -> '2-37-2', 'No. #25/23' -> '25/23'. Leading zeros of
    every numeric part are removed so '0402' == '402'."""
    if not addr:
        return ""
    m = _HN_RAW.search(addr.lower())
    if not m:
        return ""
    parts = re.split(r"\s*([-/])\s*", m.group(1))
    return "".join(p.lstrip("0") or "0" if p.isdigit() else p for p in parts)


def _house_number(comps: str) -> str:
    for c in comps.split("|"):
        for t in c.split():
            if t.isdigit():
                return t
    return ""


def _row_feats(r):
    (c1, c2, f1, f2, l1, l2, acr1, acr2, a1, a2, nu1, nu2, st1, st2, pc1, pc2, cm1, cm2, web2, nl2) = r
    t1, t2 = c1.split(), c2.split()
    s1, s2 = set(t1), set(t2)
    out = []
    # name token overlap
    inter = len(s1 & s2)
    uni = len(s1 | s2)
    out.append(inter / uni if uni else -1)
    out.append(inter / len(s1) if s1 else -1)
    out.append(inter / len(s2) if s2 else -1)
    out.append(_fuzzy_contain(t1, t2))
    out.append(_fuzzy_contain(t2, t1))
    out.append(len(t1))
    out.append(len(t2))
    # first token equal (distinctive word usually first)
    out.append(float(bool(t1) and bool(t2) and t1[0] == t2[0]))
    # acronym: record core is the initials of S1 core (e.g. 'HF' for 'Hotel Foods')
    nos2 = c2.replace(" ", "")
    out.append(float(len(acr1) >= 2 and (nos2 == acr1 or acr2 == acr1 and len(t2) == len(t1))))
    out.append(float(len(acr1) >= 2 and nos2 == acr1))
    # legal forms
    L1, L2 = set(l1.split()), set(l2.split())
    out.append(float(bool(L1) and bool(L2) and L1 == L2))
    out.append(float(bool(L1) and bool(L2) and not (L1 & L2)))
    out.append(float(not L1) + 2 * float(not L2))
    # address tokens
    at1, at2 = a1.split(), a2.split()
    A1, A2 = set(at1), set(at2)
    ai = len(A1 & A2)
    au = len(A1 | A2)
    out.append(ai / au if au else -1)
    out.append(ai / len(A1) if A1 else -1)
    out.append(ai / len(A2) if A2 else -1)
    w1 = [t for t in at1 if not t.isdigit()]
    w2 = [t for t in at2 if not t.isdigit()]
    out.append(_fuzzy_contain(w2, w1))
    out.append(_fuzzy_contain(w1, w2))
    out.append(len(at1))
    out.append(len(at2))
    # numbers
    N1, N2 = nu1.split(), nu2.split()
    S1n, S2n = set(N1), set(N2)
    if S1n and S2n:
        ni = len(S1n & S2n)
        out.append(ni / len(S1n | S2n))
        out.append(ni / len(S2n))
        out.append(float(N1[0] == N2[0]))
        out.append(float(any(_num_close(x, y) for x in S2n for y in S1n)))
        out.append(float(sum(1 for x in S2n if not any(_num_close(x, y) for y in S1n))))
        big1 = {x for x in S1n if len(x) >= 3}
        big2 = {x for x in S2n if len(x) >= 3}
        out.append(float(bool(big1) and bool(big2) and not (big1 & big2)))
    else:
        out.extend([-1, -1, -1, -1, -1, -1])
    # house number = first number of the first comma component containing one
    hn1 = _house_number(cm1)
    hn2 = _house_number(cm2)
    out.append(float(hn2 in S1n) if hn2 and S1n else -1)
    out.append(float(hn1 in S2n) if hn1 and S2n else -1)
    if hn1 and hn2:
        a, b = int(hn1[:9]), int(hn2[:9])
        out.append(abs(a - b) / max(a, b, 1))
        out.append(float(hn1 == hn2))
    else:
        out.extend([-1, -1])
    out.append(float(sum(1 for x in S1n if x not in S2n)) if S1n and S2n else -1)
    out.append(float(_REL_ORDER[_num_rel(hn1, hn2)]) if hn1 and hn2 else -1)
    # best relation over all number pairs (0 = some number equal)
    if S1n and S2n:
        out.append(float(min(_REL_ORDER[_num_rel(x, y)] for x in S2n for y in S1n)))
        # a number of the record that is 'near' / 'sub1' to an S1 number but equals none
        out.append(float(any(_num_rel(x, y) in ("near", "sub1") for x in S2n for y in S1n) and not (S1n & S2n)))
    else:
        out.extend([-1, -1])
    out.append(float(len(S1n)))
    out.append(float(len(S2n)))
    # state / postal
    ST1, ST2 = set(st1.split()), set(st2.split())
    out.append(1.0 if (ST1 & ST2) else (0.0 if (ST1 and ST2) else -1.0))
    out.append(1.0 if (pc1 and pc2 and pc1 == pc2) else (0.0 if (pc1 and pc2) else -1.0))
    # comma components: fraction of record components fuzzily present in S1
    C1 = [c for c in cm1.split("|") if c]
    C2 = [c for c in cm2.split("|") if c]
    if C1 and C2:
        hits = 0
        for c in C2:
            if max(fuzz.ratio(c, d) for d in C1) >= 85:
                hits += 1
        out.append(hits / len(C2))
        hits1 = 0
        for d in C1:
            if max(fuzz.ratio(c, d) for c in C2) >= 85:
                hits1 += 1
        out.append(hits1 / len(C1))
    else:
        out.extend([-1, -1])
    out.append(float(len(C2)))
    out.append(float(not a2))
    out.append(float(bool(web2)))
    out.append(float(bool(nl2)))
    return out


SET_NAMES = [
    "nt_jac", "nt_c1", "nt_c2", "nt_fc1", "nt_fc2", "nt_len1", "nt_len2", "nt_first_eq",
    "nm_acr", "nm_acr_strict", "lg_eq", "lg_conflict", "lg_missing",
    "at_jac", "at_c1", "at_c2", "aw_fc2", "aw_fc1", "at_len1", "at_len2",
    "num_jac", "num_c2", "num_first_eq", "num_any_close", "num_unmatched2", "num_big_conflict",
    "hn2_in1", "hn1_in2", "hn_reldiff", "hn_eq", "num_unmatched1", "hn_rel", "num_best_rel", "num_near_only", "num_n1", "num_n2", "st_eq", "pc_eq", "cm_c2", "cm_c1", "cm_n2", "ad_empty2", "is_web2", "nonlatin2",
]
_SET_COLS = ["n_core_1", "n_core_2", "n_full_1", "n_full_2", "n_legal_1", "n_legal_2", "n_acr_1", "n_acr_2",
             "a_tok_1", "a_tok_2", "a_nums_1", "a_nums_2", "a_state_1", "a_state_2", "a_postal_1", "a_postal_2",
             "a_comps_1", "a_comps_2", "n_web_2", "n_nonlatin_2"]


def _set_chunk(rows):
    return np.array([_row_feats(r) for r in rows], dtype=np.float32)


def set_features(df: pl.DataFrame, procs: int = 160, chunk: int = 20000) -> dict:
    cols = []
    for c in _SET_COLS:
        s = df[c]
        cols.append(s.fill_null(False).to_list() if s.dtype == pl.Boolean else s.fill_null("").to_list())
    rows = list(zip(*cols))
    jobs = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with Pool(procs) as pool:
        parts = pool.map(_set_chunk, jobs)
    M = np.vstack(parts) if parts else np.zeros((0, len(SET_NAMES)), np.float32)
    return {n: M[:, i] for i, n in enumerate(SET_NAMES)}


def pair_features(pairs: pl.DataFrame, norm: pl.DataFrame) -> pl.DataFrame:
    """pairs: must contain s1_id, rec_id (+ any retrieval columns). Returns pairs + features."""
    df = attach(pairs, norm)
    f = string_features(df)
    f.update(set_features(df))
    feat = pl.DataFrame(f)
    keep = [c for c in pairs.columns] + ["src"]
    return pl.concat([df.select(keep), feat], how="horizontal")
