"""Cross-encoder pair scorer (multilingual XLM-RoBERTa-base, MIT licence, 278M params).

Reads the *raw* strings of both sides ("name | address" of the Source-1 entity
and of the record) and outputs a match logit. Its score is added as a feature
to the LightGBM stack. Two models are cross-fitted by S1 entity (hash halves),
each trained on one half and scoring only the other half, so every training
pair gets an out-of-fold score; test pairs get the average of both models.

Subcommands
  prep            build pair tables (train halves, inference sets) -> cache/ce_*.parquet
  train  <m>      torchrun: fine-tune model m in {0,1} on half m
  infer  <m> <t>  torchrun: score table t with model m
  merge           collect per-rank score shards -> cache/ce_scores_{train,test}.parquet
"""
import os
import sys
import time

import numpy as np
import polars as pl

from common import cache

MODEL = os.environ.get("CE_MODEL", "xlm-roberta-base")
MAX_LEN = 128
CE_DIR = cache(os.environ.get("CE_TABLES", "ce"))      # pair tables
CE_OUT = cache(os.environ.get("CE_OUT", "ce"))         # models + score shards (per model variant)
CE_SUFFIX = os.environ.get("CE_SUFFIX", "")            # suffix of the merged score files
TRAIN_FRAMES = ["train_feats.parquet", "train_dec_r8_s2_feats.parquet", "train_sim_f19_s1_feats.parquet"]
EXTRA_FRAMES = ["train_dec_r8_s1c_feats.parquet"]  # held-out decoy style: scored, never trained on
EXTRA_NORMS = ["train_dec_r8_s2_norm.parquet", "train_dec_r8_s1c_norm.parquet"]


def _texts(norm: pl.DataFrame) -> pl.DataFrame:
    return norm.select("entity_id", pl.concat_str([pl.col("business_name").fill_null(""), pl.lit(" | "),
                                                   pl.col("business_address").fill_null("")]).alias("t"))


def ce_half():
    return pl.col("s1_id").hash(seed=99) % 2


def prep(pos_per_neg: float = 1.0):
    from match import add_labels
    os.makedirs(CE_DIR, exist_ok=True)
    cols = ["entity_id", "business_name", "business_address"]
    norm = pl.concat([pl.read_parquet(cache("train_norm.parquet"), columns=cols)]
                     + [pl.read_parquet(cache(p), columns=cols) for p in EXTRA_NORMS if os.path.exists(cache(p))])
    tx = _texts(norm)
    # every labelled training pair (deduplicated across the original and simulated frames)
    fr = []
    for p in TRAIN_FRAMES + [x for x in EXTRA_FRAMES if os.path.exists(cache(x))]:  # clone frame: optional check set
        f = pl.read_parquet(cache(p), columns=["rec_id", "s1_id"]).with_columns(pl.lit(p in TRAIN_FRAMES).alias("trainable"))
        fr.append(f)
    pairs = pl.concat(fr).group_by(["rec_id", "s1_id"]).agg(pl.col("trainable").any())
    pairs = add_labels(pairs).with_columns(ce_half().alias("half"))
    # hard positives first: the v6 stage-2 OOF score marks easy ones
    oof = pl.read_parquet(cache("train_oof.parquet"), columns=["rec_id", "s1_id", "p2"])
    pairs = pairs.join(oof, on=["rec_id", "s1_id"], how="left")
    pairs = pairs.join(tx.rename({"entity_id": "s1_id", "t": "a"}), on="s1_id").join(tx.rename({"entity_id": "rec_id", "t": "b"}), on="rec_id")
    pairs = pairs.sort(["rec_id", "s1_id"])  # deterministic order for the sampling below
    pairs.select("rec_id", "s1_id", "y", "half", "a", "b").write_parquet(os.path.join(CE_DIR, "infer_train.parquet"))
    rng = np.random.default_rng(0)
    for m in (0, 1):
        h = pairs.filter((pl.col("half") == m) & pl.col("trainable"))
        neg = h.filter(pl.col("y") == 0)
        pos = h.filter(pl.col("y") == 1)
        hard = pos.filter(pl.col("p2").is_null() | (pl.col("p2") < 0.995))
        easy = pos.filter(pl.col("p2").is_not_null() & (pl.col("p2") >= 0.995))
        n_easy = max(0, int(pos_per_neg * neg.height) - hard.height)
        easy = easy.sample(min(n_easy, easy.height), seed=m)
        t = pl.concat([neg, hard, easy]).select("a", "b", "y").sample(fraction=1.0, shuffle=True, seed=m)
        t.write_parquet(os.path.join(CE_DIR, f"train_{m}.parquet"))
        print(f"half {m}: {neg.height} negatives, {hard.height} hard + {easy.height} easy positives -> {t.height}", flush=True)
    tn = pl.read_parquet(cache("test_norm.parquet"), columns=cols)
    ttx = _texts(tn)
    t = pl.read_parquet(cache("test_feats.parquet"), columns=["rec_id", "s1_id"])
    t = t.join(ttx.rename({"entity_id": "s1_id", "t": "a"}), on="s1_id").join(ttx.rename({"entity_id": "rec_id", "t": "b"}), on="rec_id")
    t.write_parquet(os.path.join(CE_DIR, "infer_test.parquet"))
    print("inference pairs: train", pairs.height, "test", t.height, flush=True)


# --------------------------------------------------------------------------
# distributed training / inference (run under torchrun)
# --------------------------------------------------------------------------

def _dist():
    import torch
    import torch.distributed as dist
    dist.init_process_group("nccl")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return dist, dist.get_rank(), dist.get_world_size(), local


def _pad(batch_ids, pad_id):
    import torch
    L = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch_ids), L), dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    return ids, att


def train(m: int, epochs: float = 1.0, bs: int = 256, lr: float = 4e-5):
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    dist, rank, world, local = _dist()
    torch.manual_seed(1000 + m)  # classification-head initialisation and dropout
    df = pl.read_parquet(os.path.join(CE_DIR, f"train_{m}.parquet"))
    n = (df.height // world) * world
    sh = df.slice(0, n)[rank::world]
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(sh["a"].to_list(), sh["b"].to_list(), truncation=True, max_length=MAX_LEN)["input_ids"]
    y = sh["y"].to_numpy().astype(np.float32)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
    model = DDP(model, device_ids=[local])
    steps_per_epoch = len(enc) // bs
    total = int(steps_per_epoch * epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
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
            if rank == 0 and step % 200 == 0:
                print(f"[ce{m}] step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    if rank == 0:
        os.makedirs(CE_OUT, exist_ok=True)
        model.module.save_pretrained(os.path.join(CE_OUT, f"model_{m}"))
        tok.save_pretrained(os.path.join(CE_OUT, f"model_{m}"))
    dist.barrier()
    dist.destroy_process_group()


def infer(m: int, table: str, bs: int = 1024):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dist, rank, world, local = _dist()
    df = pl.read_parquet(os.path.join(CE_DIR, f"infer_{table}.parquet"))
    if table in ("train", "fr"):  # out-of-fold only: model m scores the other half
        df = df.filter(pl.col("half") != m)
    sh = df[rank::world]
    tok = AutoTokenizer.from_pretrained(os.path.join(CE_OUT, f"model_{m}"))
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(CE_OUT, f"model_{m}")).cuda().eval()
    enc = tok(sh["a"].to_list(), sh["b"].to_list(), truncation=True, max_length=MAX_LEN)["input_ids"]
    lens = np.array([len(x) for x in enc])
    order = np.argsort(lens)
    out = np.zeros(len(enc), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            ids, att = _pad([enc[i] for i in idx], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(input_ids=ids.cuda(non_blocking=True), attention_mask=att.cuda(non_blocking=True)).logits.squeeze(-1)
            out[idx] = lo.float().cpu().numpy()
            if rank == 0 and (s // bs) % 500 == 0:
                print(f"[infer ce{m} {table}] {s}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    sh.select("rec_id", "s1_id").with_columns(pl.Series(f"ce{m}", out)).write_parquet(
        os.path.join(CE_OUT, f"scores_{table}_m{m}_r{rank}.parquet"))
    dist.barrier()
    dist.destroy_process_group()


def merge():
    import glob
    for table in ("train", "test"):
        parts = {m: pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(os.path.join(CE_OUT, f"scores_{table}_m{m}_r*.parquet")))])
                 for m in (0, 1)}
        if table == "train":  # each pair was scored by exactly one (out-of-fold) model
            s = pl.concat([parts[0].rename({"ce0": "ce"}), parts[1].rename({"ce1": "ce"})])
        else:
            s = parts[0].join(parts[1], on=["rec_id", "s1_id"]).with_columns(((pl.col("ce0") + pl.col("ce1")) / 2).alias("ce")).select("rec_id", "s1_id", "ce")
        s.write_parquet(cache(f"ce_scores_{table}{CE_SUFFIX}.parquet"))
        print(table, s.shape, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "train":
        train(int(sys.argv[2]), epochs=float(os.environ.get("CE_EPOCHS", "1")),
              bs=int(os.environ.get("CE_BS", "256")), lr=float(os.environ.get("CE_LR", "4e-5")))
    elif cmd == "infer":
        infer(int(sys.argv[2]), sys.argv[3])
    elif cmd == "merge":
        merge()
