"""Normalise every record of a split (all three sources) and cache as parquet.

Output: cache/<split>_norm.parquet with one row per record:
    entity_id, src, country, raw name/address and every normalised field
    produced by text_norm.NameNormalizer / normalize_address.
"""
import json
import os
import sys
from multiprocessing import Pool

import polars as pl

from common import cache, read_source
from text_norm import NameNormalizer, normalize_address

_NORM = None


def _init(translit_path):
    global _NORM
    tl = {}
    if translit_path and os.path.exists(translit_path):
        with open(translit_path) as f:
            tl = json.load(f)
    _NORM = NameNormalizer(tl)


def _work(chunk):
    names, addrs, countries = chunk
    out = []
    for n, a, c in zip(names, addrs, countries):
        d = _NORM.normalize(n)
        d.update(normalize_address(a, c))
        out.append(d)
    return out


def normalize_frame(df: pl.DataFrame, translit_path: str, procs: int = min(128, os.cpu_count() or 1), chunk: int = 20000) -> pl.DataFrame:
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    ctry = df["country"].to_list()
    chunks = [(names[i:i + chunk], addrs[i:i + chunk], ctry[i:i + chunk]) for i in range(0, len(names), chunk)]
    with Pool(procs, initializer=_init, initargs=(translit_path,)) as pool:
        res = pool.map(_work, chunks)
    rows = [r for part in res for r in part]
    norm = pl.DataFrame(rows)
    return pl.concat([df, norm], how="horizontal")


def run(split: str, translit_path: str | None = None, suffix: str = ""):
    translit_path = translit_path or cache("translit.json")
    dfs = [read_source(split, s) for s in (1, 2, 3)]
    df = pl.concat(dfs)
    out = normalize_frame(df, translit_path)
    out = out.with_columns(pl.col("country").fill_null("").alias("country"))
    path = cache(f"{split}_norm{suffix}.parquet")
    out.write_parquet(path)
    print(split, out.shape, "->", path)


if __name__ == "__main__":
    for sp in sys.argv[1:] or ["train", "test"]:
        run(sp)
