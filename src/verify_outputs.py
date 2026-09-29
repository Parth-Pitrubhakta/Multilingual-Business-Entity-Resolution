"""Check the two output files against the challenge format and, optionally, compare them with reference files.

Checks: exact header, tab delimiter with two columns, exactly one row per test Source-1 entity, no duplicate
rows or IDs within a list, no empty / NaN tokens, every ID is an existing test Source-2/3 record, every record
matched to at most one entity, and matching_results pairs are a subset of candidate_pairs.
With --reference_dir: MD5 of both files and the pair-level difference (+ added / - removed, per country).

usage: python verify_outputs.py --output_dir <dir> --data_dir <dataset dir> [--reference_dir <dir>]
"""
import argparse
import hashlib
import os
import sys
from collections import Counter

HEADERS = {"matching_results.tsv": "source1_entity_id\tmatched_entity_ids",
           "candidate_pairs.tsv": "source1_entity_id\tcandidate_entity_ids"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def read_ids(path, col):
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        i = head.index(col)
        c = head.index("country") if "country" in head else None
        ids, country = set(), {}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ids.add(parts[i])
            if c is not None:
                country[parts[i]] = parts[c]
    return ids, country


def parse(path, name, s1_ids, rec_ids, errors):
    pairs = set()
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n")
        if head != HEADERS[name]:
            errors.append(f"{name}: header {head!r} != {HEADERS[name]!r}")
        seen = set()
        for n, line in enumerate(f, 2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                errors.append(f"{name}:{n}: {len(parts)} tab-separated columns")
                continue
            s, lst = parts
            if s in seen:
                errors.append(f"{name}:{n}: duplicate row for {s}")
            seen.add(s)
            if s not in s1_ids:
                errors.append(f"{name}:{n}: {s} is not a test Source-1 entity")
            toks = lst.split(",") if lst else []
            if len(toks) != len(set(toks)):
                errors.append(f"{name}:{n}: duplicate IDs in the list of {s}")
            for r in toks:
                if r == "" or r.lower() in ("nan", "none", "null") or r not in rec_ids:
                    errors.append(f"{name}:{n}: invalid ID {r!r}")
                pairs.add((s, r))
        missing = len(s1_ids - seen)
        if missing:
            errors.append(f"{name}: {missing} test Source-1 entities have no row")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_dir", required=True, help="dataset folder with train/ and test/")
    ap.add_argument("--reference_dir", help="folder with reference matching_results.tsv / candidate_pairs.tsv")
    a = ap.parse_args()
    test = os.path.join(a.data_dir, "test")
    s1_ids, country = read_ids(os.path.join(test, "test_source1.tsv"), "entity_id")
    rec_ids = read_ids(os.path.join(test, "test_source2.tsv"), "entity_id")[0] | read_ids(os.path.join(test, "test_source3.tsv"), "entity_id")[0]
    errors, P = [], {}
    for name in HEADERS:
        path = os.path.join(a.output_dir, name)
        P[name] = parse(path, name, s1_ids, rec_ids, errors)
        print(f"{name}: {len(P[name]):,} pairs ({len(P[name]) / len(s1_ids):.3f} per Source-1 entity), md5 {md5(path)}")
    m, c = P["matching_results.tsv"], P["candidate_pairs.tsv"]
    outside = m - c
    if outside:
        errors.append(f"{len(outside)} matched pairs are not in candidate_pairs.tsv")
    multi = sum(1 for v in Counter(r for _, r in m).values() if v > 1)
    if multi:
        errors.append(f"{multi} records are matched to more than one Source-1 entity")
    print(f"matches in candidates: {len(m) - len(outside):,}/{len(m):,}; records matched twice: {multi}")
    if a.reference_dir:
        for name in HEADERS:
            ref = os.path.join(a.reference_dir, name)
            same = md5(ref) == md5(os.path.join(a.output_dir, name))
            R = parse(ref, name, s1_ids, rec_ids, [])
            add, rem = P[name] - R, R - P[name]
            by = lambda s: dict(Counter(country.get(x, "?") for x, _ in s))
            print(f"vs reference {name}: byte-identical {same}; +{len(add):,} / -{len(rem):,} pairs "
                  f"(of {len(R):,}); added by country {by(add)}; removed by country {by(rem)}")
    if errors:
        print(f"FAIL: {len(errors)} problem(s)")
        for e in errors[:50]:
            print("  " + e)
        sys.exit(1)
    print("PASS: both files follow the challenge format")


if __name__ == "__main__":
    main()
