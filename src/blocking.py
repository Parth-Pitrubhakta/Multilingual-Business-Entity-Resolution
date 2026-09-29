"""Stage 1 of candidate generation: GPU sparse TF-IDF retrieval.

For every Source 2/3 record we retrieve Source 1 entities *of the same
country* (country is an open set of labels; every label present in the data
gets its own block, so unseen countries such as France are handled
identically) with two sparse TF-IDF views:

  * name view    - char 3-grams + words + consonant skeletons of the core name
                   (legal forms and honorifics removed, website stems added)
  * address view - address words, char 3-grams of words, word bigrams inside
                   a comma component, and the canonical state

Cosine similarities are computed exactly as sparse(S1) x dense(query chunk)
matrix products on the GPU; for each query we keep the top-k S1 by name,
by address and by their sum. Because every S2/S3 record belongs to at most
one S1 entity, retrieving from the record side keeps the candidate lists
tiny: the candidate set of an S1 entity is the set of records that
retrieved it.
"""
import math
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher

from common import cache

N_FEATURES = 1 << 22
NPROC = min(128, os.cpu_count() or 1)


def _grams(tok: str, n: int = 3):
    t = "#" + tok + "#"
    return [t[i:i + n] for i in range(len(t) - n + 1)]


def name_features(n_core: str, n_web: str, n_skel: str):
    feats = []
    for t in n_core.split():
        feats.append("w" + t)
        feats.extend(_grams(t))
    for t in n_web.split():
        feats.append("w" + t)
        feats.extend(_grams(t))
    for t in n_skel.split():
        if len(t) >= 2:
            feats.append("k" + t)
    return feats


def addr_features(a_comps: str, a_state: str):
    feats = []
    for comp in a_comps.split("|"):
        toks = comp.split()
        for i, t in enumerate(toks):
            feats.append("a" + t)
            if len(t) >= 4 and not t.isdigit():
                feats.extend(_grams(t))
            if i:
                feats.append("b" + toks[i - 1] + "_" + t)
    for s in a_state.split():
        feats.append("s" + s)
    return feats


def _hash_chunk(args):
    kind, cols = args
    fh = FeatureHasher(n_features=N_FEATURES, input_type="string", alternate_sign=False)
    if kind == "name":
        rows = [name_features(a, b, c) for a, b, c in zip(*cols)]
    else:
        rows = [addr_features(a, b) for a, b in zip(*cols)]
    return fh.transform(rows).tocsr().astype(np.float32)


def hash_view(df: pl.DataFrame, kind: str, procs: int = NPROC, chunk: int = 50000) -> sp.csr_matrix:
    if kind == "name":
        cols = [df[c].fill_null("").to_list() for c in ("n_core", "n_web", "n_skel")]
    else:
        cols = [df[c].fill_null("").to_list() for c in ("a_comps", "a_state")]
    n = df.height
    jobs = [(kind, [c[i:i + chunk] for c in cols]) for i in range(0, n, chunk)]
    with Pool(procs) as pool:
        parts = pool.map(_hash_chunk, jobs)
    return sp.vstack(parts).tocsr()


def tfidf(X: sp.csr_matrix) -> sp.csr_matrix:
    """Sublinear tf * smooth idf (idf fitted on the given rows), L2-normalised rows."""
    X = X.tocsr(copy=True)
    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float64)
    idf = np.log((1 + X.shape[0]) / (1 + df)) + 1.0
    X.data = (1.0 + np.log(X.data)) * idf[X.indices].astype(np.float32)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    X = sp.diags((1.0 / norms).astype(np.float32)) @ X
    return X.tocsr().astype(np.float32)


# --------------------------------------------------------------------------
# GPU worker
# --------------------------------------------------------------------------

def gpu_worker(job_path: str, out_path: str, k: int, batch: int):
    """Top-k retrieval of queries against docs for one shard on one GPU."""
    import torch

    z = np.load(job_path, allow_pickle=True)
    dev = torch.device("cuda")

    def to_torch_csr(M):
        return torch.sparse_csr_tensor(
            torch.from_numpy(M.indptr.astype(np.int64)), torch.from_numpy(M.indices.astype(np.int64)),
            torch.from_numpy(M.data), size=M.shape, device=dev)

    def dense_T(M):
        c = M.tocoo()
        ind = torch.from_numpy(np.vstack([c.col, c.row]).astype(np.int64)).to(dev)
        val = torch.from_numpy(c.data).to(dev)
        return torch.sparse_coo_tensor(ind, val, size=(M.shape[1], M.shape[0])).to_dense()

    Dn = sp.csr_matrix((z["dn_data"], z["dn_ind"], z["dn_ptr"]), shape=tuple(z["dn_shape"]))
    Da = sp.csr_matrix((z["da_data"], z["da_ind"], z["da_ptr"]), shape=tuple(z["da_shape"]))
    Qn = sp.csr_matrix((z["qn_data"], z["qn_ind"], z["qn_ptr"]), shape=tuple(z["qn_shape"]))
    Qa = sp.csr_matrix((z["qa_data"], z["qa_ind"], z["qa_ptr"]), shape=tuple(z["qa_shape"]))
    tDn, tDa = to_torch_csr(Dn), to_torch_csr(Da)
    nq, nd = Qn.shape[0], Dn.shape[0]
    kk = min(k, nd)
    out_idx = np.zeros((nq, 3 * kk), dtype=np.int32)
    out_sn = np.zeros((nq, 3 * kk), dtype=np.float16)
    out_sa = np.zeros((nq, 3 * kk), dtype=np.float16)
    t0 = time.time()
    for s in range(0, nq, batch):
        e = min(nq, s + batch)
        qn = dense_T(Qn[s:e])  # (H_n, B), densified on the GPU
        qa = dense_T(Qa[s:e])
        Sn = torch.sparse.mm(tDn, qn)  # (nd, B)
        Sa = torch.sparse.mm(tDa, qa)
        del qn, qa
        Sc = Sn + Sa
        i1 = torch.topk(Sn, kk, dim=0).indices
        i2 = torch.topk(Sa, kk, dim=0).indices
        i3 = torch.topk(Sc, kk, dim=0).indices
        idx = torch.cat([i3, i1, i2], dim=0)  # (3k, B), combined first
        sn = torch.gather(Sn, 0, idx)
        sa = torch.gather(Sa, 0, idx)
        out_idx[s:e] = idx.T.cpu().numpy()
        out_sn[s:e] = sn.T.cpu().numpy().astype(np.float16)
        out_sa[s:e] = sa.T.cpu().numpy().astype(np.float16)
        del Sn, Sa, Sc
        if (s // batch) % 200 == 0:
            print(f"[{os.path.basename(job_path)}] {e}/{nq} {time.time() - t0:.0f}s", flush=True)
    np.savez(out_path, idx=out_idx, sn=out_sn, sa=out_sa)


def _compress_cols(D: sp.csr_matrix, Q: sp.csr_matrix):
    """Keep only feature columns present in the docs (others add 0 to dot products)."""
    cols = np.unique(D.indices)
    remap = np.full(D.shape[1], -1, dtype=np.int64)
    remap[cols] = np.arange(len(cols))
    D2 = sp.csr_matrix((D.data, remap[D.indices], D.indptr), shape=(D.shape[0], len(cols)))
    Qc = Q.tocoo()
    keep = remap[Qc.col] >= 0
    Q2 = sp.csr_matrix((Qc.data[keep], (Qc.row[keep], remap[Qc.col[keep]])), shape=(Q.shape[0], len(cols)))
    return D2, Q2


def retrieve(split: str, k: int = 10, batch: int = 1024, gpus=None):
    # physical GPU ids, one retrieval shard per GPU (run_pipeline.py sets BER_GPUS)
    gpus = gpus or [int(g) for g in os.environ.get("BER_GPUS", "0,1,2,3").split(",") if g != ""]
    df = pl.read_parquet(cache(f"{split}_norm.parquet")).with_row_index("row")
    print("hashing views ...", flush=True)
    Xn = hash_view(df, "name")
    Xa = hash_view(df, "addr")
    src = df["src"].to_numpy()
    country = df["country"].to_list()
    ids = df["entity_id"].to_numpy()
    countries = sorted(set(country))
    carr = np.array(country)
    results = []
    tmpdir = cache(f"blk_{split}")
    os.makedirs(tmpdir, exist_ok=True)
    for c in countries:
        rows = np.where(carr == c)[0]
        d_rows = rows[src[rows] == 1]
        q_rows = rows[src[rows] != 1]
        if len(d_rows) == 0 or len(q_rows) == 0:
            continue
        print(f"country={c!r}: {len(d_rows)} S1 docs, {len(q_rows)} queries", flush=True)
        Tn = tfidf(Xn[rows])
        Ta = tfidf(Xa[rows])
        is_doc = src[rows] == 1
        Dn, Qn = _compress_cols(Tn[is_doc], Tn[~is_doc])
        Da, Qa = _compress_cols(Ta[is_doc], Ta[~is_doc])
        shards = np.array_split(np.arange(len(q_rows)), len(gpus))
        procs = []
        for g, sh in zip(gpus, shards):
            jp = os.path.join(tmpdir, f"job_{c}_{g}.npz")
            op = os.path.join(tmpdir, f"out_{c}_{g}.npz")
            qn, qa = Qn[sh], Qa[sh]
            np.savez(jp, dn_data=Dn.data, dn_ind=Dn.indices, dn_ptr=Dn.indptr, dn_shape=Dn.shape,
                     da_data=Da.data, da_ind=Da.indices, da_ptr=Da.indptr, da_shape=Da.shape,
                     qn_data=qn.data, qn_ind=qn.indices, qn_ptr=qn.indptr, qn_shape=qn.shape,
                     qa_data=qa.data, qa_ind=qa.indices, qa_ptr=qa.indptr, qa_shape=qa.shape)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            procs.append((subprocess.Popen(
                [sys.executable, __file__, "worker", jp, op, str(k), str(batch)], env=env), sh, op, jp))
        for p, sh, op, jp in procs:
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"GPU worker failed for {c}")
            z = np.load(op)
            kk = z["idx"].shape[1]
            qi = np.repeat(q_rows[sh], kk)
            view = np.tile(np.repeat(np.arange(3), kk // 3), len(sh))  # 0=comb,1=name,2=addr
            rank = np.tile(np.tile(np.arange(kk // 3), 3), len(sh))
            results.append(pl.DataFrame({
                "rec_id": ids[qi],
                "s1_id": ids[d_rows[z["idx"].ravel()]],
                "view": view.astype(np.int8),
                "rank": rank.astype(np.int16),
                "cos_n": z["sn"].ravel().astype(np.float32),
                "cos_a": z["sa"].ravel().astype(np.float32),
            }))
            os.remove(jp)
    res = pl.concat(results)
    # One row per (rec, s1) with the best rank reached in each view.
    agg = res.group_by(["rec_id", "s1_id"]).agg(
        pl.col("cos_n").max(), pl.col("cos_a").max(),
        pl.when(pl.col("view") == 0).then(pl.col("rank")).min().fill_null(999).alias("rk_c"),
        pl.when(pl.col("view") == 1).then(pl.col("rank")).min().fill_null(999).alias("rk_n"),
        pl.when(pl.col("view") == 2).then(pl.col("rank")).min().fill_null(999).alias("rk_a"),
    )
    agg = agg.sort(["rec_id", "s1_id"])  # deterministic row order for everything downstream
    path = cache(f"{split}_retrieval.parquet")
    agg.write_parquet(path)
    print("retrieval pairs:", agg.shape, "->", path)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        gpu_worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    else:
        retrieve(sys.argv[1], k=int(sys.argv[2]) if len(sys.argv) > 2 else 10)
