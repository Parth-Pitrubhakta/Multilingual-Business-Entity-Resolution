"""Entity-level expected-F0.5 decision rule.

The metric is F0.5 averaged over Source-1 entities, so the best acceptance cut-off depends on
the entity: an extra record is worth more for an entity with few confident matches than for
one with many. Each record is first assigned to its arg-max Source-1 entity (records match at
most one entity); then, per entity, the top-j records (by calibrated probability) are chosen to
maximise the expected F0.5 under independent Bernoulli truth, computed exactly with
Poisson-binomial sums. `lam` adds expected true records outside the candidate set.
"""
import numpy as np
import polars as pl

MAXN = 12


def _pmf_prefix(P):
    """P: (E, n) probabilities -> list of prefix pmfs, pref[j] has shape (E, j+1)."""
    E, n = P.shape
    pref = [np.ones((E, 1))]
    for i in range(n):
        prev = pref[-1]
        nxt = np.zeros((E, i + 2))
        nxt[:, :-1] += prev * (1 - P[:, i:i + 1])
        nxt[:, 1:] += prev * P[:, i:i + 1]
        pref.append(nxt)
    return pref


def _best_j(P, lam=0.0, beta2=0.25):
    E, n = P.shape
    pref = _pmf_prefix(P)
    suf = _pmf_prefix(P[:, ::-1])  # suf[m]: pmf of the last m columns
    # unseen true records ~ Poisson(lam), truncated at 3
    u = np.array([np.exp(-lam) * lam ** k / __import__("math").factorial(k) for k in range(4)]) if lam > 0 else np.array([1.0])
    u = u / u.sum()
    best = np.full(E, -1.0); bj = np.zeros(E, dtype=np.int64)
    for j in range(n + 1):
        A = pref[j]                 # TP among chosen: 0..j
        B = suf[n - j]              # true among rejected: 0..n-j
        B = np.stack([np.convolve(b, u)[:B.shape[1] + len(u) - 1] for b in B]) if len(u) > 1 else B
        a = np.arange(A.shape[1])[:, None]; b = np.arange(B.shape[1])[None, :]
        if j == 0:
            f = (b == 0).astype(float) * np.ones_like(a, dtype=float)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                f = np.where(a > 0, 1.25 * a / (1.25 * a + beta2 * b + (j - a)), 0.0)
        ef = np.einsum('ea,ab,eb->e', A, f, B)
        upd = ef > best
        best[upd] = ef[upd]; bj[upd] = j
    return bj, best


def decide_expected_f(f: pl.DataFrame, pcol: str, floor: float = 0.005, lam: float = 0.0) -> pl.DataFrame:
    """f: candidate pairs (s1_id, rec_id, pcol). Returns chosen (s1_id, rec_id)."""
    best = f.filter(pl.col(pcol) == pl.col(pcol).max().over("rec_id")).unique("rec_id", keep="first")
    best = best.filter(pl.col(pcol) >= floor).select("s1_id", "rec_id", pl.col(pcol).alias("p"))
    best = best.sort(["s1_id", "p"], descending=[False, True]).with_columns(pl.int_range(pl.len()).over("s1_id").alias("r"))
    best = best.filter(pl.col("r") < MAXN)
    g = best.group_by("s1_id", maintain_order=True).agg(pl.col("p"), pl.col("rec_id"))
    g = g.with_columns(pl.col("p").list.len().alias("n"))
    out = []
    for n in sorted(g["n"].unique().to_list()):
        h = g.filter(pl.col("n") == n)
        P = np.clip(np.array(h["p"].to_list(), dtype=np.float64), 1e-6, 1 - 1e-6)
        bj, _ = _best_j(P, lam=lam)
        h = h.with_columns(pl.Series("j", bj))
        out.append(h.select("s1_id", "rec_id", "j").explode("rec_id").with_columns(pl.int_range(pl.len()).over("s1_id").alias("r"))
                   .filter(pl.col("r") < pl.col("j")).select("s1_id", "rec_id"))
    return pl.concat(out)
