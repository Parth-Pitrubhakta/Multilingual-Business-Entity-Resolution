"""LLM pair classifier (Qwen2.5-1.5B, Apache-2.0, 1.54B parameters) for the AMBIGUOUS pairs only (v28, stage 4 input).

Band = every candidate pair of a record that has at least one pair with blended stack score in [0.005, 0.999)
(train: 99.8% of the out-of-fold missed matches and 95.8% of the wrong acceptances live there).
Cross-fitting as for the other cross-encoders: model m is fine-tuned on the band pairs of S1 half m
(hash(s1_id, 99) % 2) and scores the other half (out-of-fold) and every test band pair; a test pair keeps the
score of the model of the OTHER half of its entity, so train and test scores have the same distribution.

  python llm_ce.py prep                                   -> <cache>/llm/{train_band,test_band}.parquet
  torchrun --nproc_per_node=2 llm_ce.py train <m>         -> <cache>/llm/model_<m>
  torchrun --nproc_per_node=2 llm_ce.py infer <m>         -> <cache>/llm/scores_<split>_m<m>_r<rank>.parquet
  python llm_ce.py merge                                  -> <cache>/llm/llm_scores_{train,test}.parquet
"""
import os
import sys
import time

import numpy as np
import polars as pl

from common import cache, read_ground_truth

LLM = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B")  # run_pipeline.py passes the pinned local snapshot
LDIR = cache(os.environ.get("LLM_DIR", "llm"))
MAX_LEN = int(os.environ.get("LLM_MAXLEN", "96"))
BAND = (0.005, 0.999)
W = {"full_": 1, "vf_": 1, "vfb_": 3, "vfm_": 3}  # blend weights of the four stacks (as ensemble_seen.py)
SPLITS = ("train", "test")


def blend(split):
    """Widened-candidate scores of the four stacks (widen.py score) and their weighted blend `pe`."""
    x = None
    for t in W:
        s = pl.read_parquet(cache(f"w_{split}_scores{t}.parquet"), columns=["rec_id", "s1_id", "p3"]).rename({"p3": t})
        x = s if x is None else x.join(s, on=["rec_id", "s1_id"])
    return x.with_columns((sum(w * pl.col(t) for t, w in W.items()) / sum(W.values())).alias("pe"))


def band_pairs(x):
    recs = x.filter((pl.col("pe") >= BAND[0]) & (pl.col("pe") < BAND[1])).select("rec_id").unique()
    return x.join(recs, on="rec_id")


def _text(norm):
    return norm.select("entity_id", pl.concat_str([pl.col("business_name").fill_null(""), pl.lit(" | "),
                                                   pl.col("business_address").fill_null("")]).alias("t"))


def prep():
    from cross_encoder import ce_half
    os.makedirs(LDIR, exist_ok=True)
    gt = read_ground_truth().with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
    for split in SPLITS:
        b = band_pairs(blend(split))
        tx = _text(pl.read_parquet(cache(f"{split}_norm.parquet"), columns=["entity_id", "business_name", "business_address"]))
        b = b.join(tx.rename({"entity_id": "s1_id", "t": "a"}), on="s1_id").join(tx.rename({"entity_id": "rec_id", "t": "b"}), on="rec_id")
        b = b.with_columns(ce_half().alias("half"))
        if split == "train":
            b = b.join(gt, on=["rec_id", "s1_id"], how="left").with_columns(pl.col("y").fill_null(0))
        b = b.sort(["rec_id", "s1_id"])  # deterministic order for the training shuffle
        b.write_parquet(os.path.join(LDIR, f"{split}_band.parquet"))
        print(split, "band pairs", b.height, "records", b["rec_id"].n_unique(),
              ("positives %d" % b["y"].sum()) if split == "train" else "", flush=True)


def _dist():
    import torch
    import torch.distributed as dist
    dist.init_process_group("nccl")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return dist, dist.get_rank(), dist.get_world_size(), local


def _tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _encode(tok, df):
    txt = [f"Entity: {a}\nRecord: {b}\nSame business:" for a, b in zip(df["a"].to_list(), df["b"].to_list())]
    return tok(txt, truncation=True, max_length=MAX_LEN)["input_ids"]


def _pad(batch, pad_id):
    import torch
    L = max(len(x) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, x in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    return ids, att


def _model(path, train=False):
    """Sequence classifier with a 1-logit head on the last token; fp32 weights for training (bf16 autocast), bf16 for inference."""
    import torch
    from transformers import AutoModelForSequenceClassification
    dt = torch.float32 if train else torch.bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=dt,
                                                               attn_implementation=os.environ.get("LLM_ATTN", "sdpa"))
    tok = _tok()
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def train(m, epochs=float(os.environ.get("LLM_EPOCHS", "1.0")), bs=int(os.environ.get("LLM_BS", "64")),
          lr=float(os.environ.get("LLM_LR", "1.5e-5"))):
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import get_linear_schedule_with_warmup
    dist, rank, world, local = _dist()
    torch.manual_seed(2000 + m)  # classification-head initialisation
    df = pl.read_parquet(os.path.join(LDIR, "train_band.parquet")).filter(pl.col("half") == m)
    df = df.sample(fraction=1.0, shuffle=True, seed=m)
    n = (df.height // world) * world
    sh = df.slice(0, n)[rank::world]
    model, tok = _model(LLM, train=True)
    enc = _encode(tok, sh)
    y = sh["y"].to_numpy().astype(np.float32)
    model = DDP(model.cuda(), device_ids=[local])
    steps_per_epoch = len(enc) // bs
    total = int(steps_per_epoch * epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=True)
    sch = get_linear_schedule_with_warmup(opt, int(0.03 * total), total)
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(rank + 17 * m)
    step, t0 = 0, time.time()
    model.train()
    while step < total:
        order = rng.permutation(len(enc))
        for s in range(0, steps_per_epoch * bs, bs):
            if step >= total:
                break
            idx = order[s:s + bs]
            ids, att = _pad([enc[i] for i in idx], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids.cuda(non_blocking=True), attention_mask=att.cuda(non_blocking=True)).logits.squeeze(-1)
            loss = lossf(out.float(), torch.from_numpy(y[idx]).cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
            if rank == 0 and step % 100 == 0:
                print(f"[llm{m}] step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    if rank == 0:
        model.module.save_pretrained(os.path.join(LDIR, f"model_{m}"), safe_serialization=True)
        tok.save_pretrained(os.path.join(LDIR, f"model_{m}"))
        print(f"[llm{m}] saved, {time.time() - t0:.0f}s", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def infer(m, bs=512):
    import torch
    dist, rank, world, local = _dist()
    model, tok = _model(os.path.join(LDIR, f"model_{m}"))
    model = model.to(torch.bfloat16).cuda().eval()
    for split in SPLITS:
        df = pl.read_parquet(os.path.join(LDIR, f"{split}_band.parquet"), columns=["rec_id", "s1_id", "half", "a", "b"])
        if split == "train":
            df = df.filter(pl.col("half") != m)  # out-of-fold only
        sh = df[rank::world]
        enc = _encode(tok, sh)
        order = np.argsort([len(x) for x in enc])
        out = np.zeros(len(enc), np.float32)
        t0 = time.time()
        with torch.no_grad():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                ids, att = _pad([enc[i] for i in idx], tok.pad_token_id)
                out[idx] = model(input_ids=ids.cuda(non_blocking=True), attention_mask=att.cuda(non_blocking=True)).logits.squeeze(-1).float().cpu().numpy()
        sh.select("rec_id", "s1_id").with_columns(pl.Series(f"llm{m}", out)).write_parquet(
            os.path.join(LDIR, f"scores_{split}_m{m}_r{rank}.parquet"))
        print(f"[llm{m} {split}] rank {rank}: {len(enc)} pairs {time.time() - t0:.0f}s", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def merge():
    import glob
    from cross_encoder import ce_half
    for split in SPLITS:
        parts = {m: pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(os.path.join(LDIR, f"scores_{split}_m{m}_r*.parquet")))])
                 for m in (0, 1)}
        if split == "train":
            s = pl.concat([parts[0].rename({"llm0": "llm"}), parts[1].rename({"llm1": "llm"})])
        else:  # same distribution as train: each pair scored by the model of the OTHER S1 half (+ the average, for reference)
            s = parts[0].join(parts[1], on=["rec_id", "s1_id"]).with_columns(ce_half().alias("half"))
            s = s.select("rec_id", "s1_id", pl.when(pl.col("half") == 0).then(pl.col("llm1")).otherwise(pl.col("llm0")).alias("llm"),
                         ((pl.col("llm0") + pl.col("llm1")) / 2).alias("llm_avg"))
        s.write_parquet(os.path.join(LDIR, f"llm_scores_{split}.parquet"))
        print(split, s.shape, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "train":
        train(int(sys.argv[2]))
    elif cmd == "infer":
        infer(int(sys.argv[2]))
    elif cmd == "merge":
        merge()
