"""Shared paths, I/O helpers and the F0.5 metric used throughout the pipeline."""
import os
import polars as pl

ROOT = os.environ.get(
    "BER_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
DATA_DIR = os.environ.get("BER_DATA", os.path.join(ROOT, "student_resource", "dataset"))
CACHE_DIR = os.environ.get("BER_CACHE", os.path.join(ROOT, "cache"))
OUTPUT_DIR = os.environ.get("BER_OUTPUT", os.path.join(ROOT, "output"))
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_tsv(path: str) -> pl.DataFrame:
    """Read a challenge TSV as all-string columns (no quoting, empty -> null)."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def read_source(split: str, src: int) -> pl.DataFrame:
    """Read `<split>_source<src>.tsv` and attach an integer `src` column."""
    df = read_tsv(os.path.join(DATA_DIR, split, f"{split}_source{src}.tsv"))
    return df.with_columns(pl.lit(src, dtype=pl.Int8).alias("src"))


def read_ground_truth() -> pl.DataFrame:
    """Return ground truth exploded to one (s1_id, rec_id) row per true pair."""
    gt = read_tsv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"))
    return (
        gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "rec_id"})
    )


def cache(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def f05_macro(pred: dict, truth: dict, s1_ids) -> float:
    """Macro F0.5 over Source-1 entities, exactly as the challenge defines it.

    pred / truth map s1_id -> set of matched ids. Empty-vs-empty scores 1.0,
    any prediction on a true singleton (or empty prediction on a non-singleton)
    scores 0.0.
    """
    tot = 0.0
    n = 0
    for s in s1_ids:
        p = pred.get(s, set())
        t = truth.get(s, set())
        n += 1
        if not p and not t:
            tot += 1.0
            continue
        if not p or not t:
            continue
        tp = len(p & t)
        if tp == 0:
            continue
        prec = tp / len(p)
        rec = tp / len(t)
        tot += 1.25 * prec * rec / (0.25 * prec + rec)
    return tot / max(n, 1)
