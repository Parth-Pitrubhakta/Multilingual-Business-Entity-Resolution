"""Single entry point: provided data -> normalisation -> blocking -> matching -> output/{candidate_pairs,matching_results}.tsv

Runs every stage of the submitted pipeline (v28) in order, each as a subprocess of the modules in this folder,
with a log per job in <work_dir>/logs and wall-clock times in <work_dir>/logs/timings.tsv.

  python run_pipeline.py --data_dir <student_resource/dataset> --output_dir <out> --work_dir <work> [--gpus 0,1,2,3]
  python run_pipeline.py --list                      # show the steps
  python run_pipeline.py ... --from_step 13          # resume (intermediate files are kept in <work_dir>)

<data_dir> must contain train/{train_source1,2,3,train_ground_truth}.tsv and test/test_source{1,2,3}.tsv.
Intermediate decision files (per stack, France steps v12..v24, v25) are written to <work_dir>/stages.
"""
import argparse
import os
import subprocess
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# the four LightGBM stacks (Documentation_template.md, sections 10-13)
FULL = {"MODEL_TAG": "full_", "CE_VARIANTS": "ce,cel", "TOKSTATS": "0", "MASK_CE": "0", "VOCAB_FREE_UNSEEN": "0"}
VF = {"MODEL_TAG": "vf_", "CE_VARIANTS": "ce,cel,cel2"}
VFB = {"MODEL_TAG": "vfb_", "CE_VARIANTS": "ce,cel,cel2", "LGB_SEED": "7", "FIT_LR": "0.05", "ROUNDS2": "800", "ROUNDS3": "500"}
VFM = {"MODEL_TAG": "vfm_", "CE_VARIANTS": "ce,cel,cel2,cem"}
STACKS = {"full": FULL, "vf": VF, "vfb": VFB, "vfm": VFM}
BLEND, THR = "full_:1,vf_:1,vfb_:3,vfm_:3", "0.725"


class Job:
    def __init__(self, name, args, env=None, gpus=None, nproc=0):
        self.name, self.args, self.env, self.gpus, self.nproc = name, [str(a) for a in args], env or {}, gpus, nproc


class Runner:
    def __init__(self, a):
        self.data = os.path.abspath(a.data_dir)
        self.out = os.path.abspath(a.output_dir)
        self.work = os.path.abspath(a.work_dir)
        self.models = os.path.abspath(a.model_dir) if a.model_dir else os.path.join(self.work, "models")
        self.stages = os.path.join(self.work, "stages")
        self.logs = os.path.join(self.work, "logs")
        self.gpus = [int(g) for g in a.gpus.split(",") if g != ""]
        self.port = 29500 + 100 * (os.getpid() % 50)
        for d in (self.out, self.work, self.stages, self.logs):
            os.makedirs(d, exist_ok=True)

    # ---------------------------------------------------------------- helpers
    def st(self, v, f="matching_results.tsv"):
        return os.path.join(self.stages, v, f)

    def m(self, name):
        return os.path.join(self.models, name)

    def w(self, name):
        return os.path.join(self.work, name)

    def halves(self, make):
        """One job per S1 half m in {0, 1}, each with DDP on 2 GPUs; both halves at once when 4 GPUs are given."""
        pairs = [self.gpus[0:2], self.gpus[2:4]] if len(self.gpus) >= 4 else [self.gpus[0:2]] * 2
        jobs = [make(m, pairs[m]) for m in (0, 1)]
        return [jobs] if len(self.gpus) >= 4 else [[j] for j in jobs]

    def ce_variant(self, tag, common, train_env):
        """Cross-encoder variant: two half models (train), out-of-fold train + test scores (infer), merge."""
        g = self.halves(lambda m, p: Job(f"{tag}_train{m}", ["cross_encoder.py", "train", m], {**common, **train_env(m)}, p, 2))
        for t in ("train", "test"):
            g += self.halves(lambda m, p, t=t: Job(f"{tag}_infer{m}_{t}", ["cross_encoder.py", "infer", m, t], common, p, 2))
        return g + [[Job(f"{tag}_merge", ["cross_encoder.py", "merge"], common)]]

    def france_round(self, r):
        d, sc = ("ce_fr", "ce_scores_france.parquet") if r == 1 else ("ce_fr2", "ce_scores_france2.parquet")
        base, nxt = ("v16", "v17") if r == 1 else ("v17", "v18")
        tr = lambda m: {"CE_MODEL": self.w(f"ce_large2/model_{m}"), "CE_BS": "128", "CE_LR": "8e-6"}
        common = {"CE_TABLES": d, "CE_OUT": d}
        g = [[Job(f"cefr{r}_prep", ["ce_france.py", "prep", self.st(base)], {"CEFR_DIR": d})]]
        g += self.halves(lambda m, p: Job(f"cefr{r}_train{m}", ["cross_encoder.py", "train", m], {**common, **tr(m)}, p, 2))
        g += self.halves(lambda m, p: Job(f"cefr{r}_infer{m}", ["cross_encoder.py", "infer", m, "fr"], common, p, 2))
        g += [[Job(f"cefr{r}_merge", ["ce_france.py", "merge"], {"CEFR_DIR": d, "CEFR_OUT": sc})]]
        g += [[Job(f"{nxt}_selftrain", ["france_rules.py", "selftrain", self.st(base), self.w(sc), self.st(nxt)])]]
        return g

    # ---------------------------------------------------------------- the pipeline
    def steps(self):
        S = []
        add = lambda name, groups: S.append((name, groups))
        one = lambda *jobs: [list(jobs)]
        add("models: download the 4 pretrained checkpoints (pinned revisions)",
            one(Job("download", ["download_models.py", self.models], {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})))
        add("normalise: transliteration dictionary (train pairs) + normalise all records",
            [[Job("translit", ["translit.py"])], [Job("preprocess", ["preprocess.py", "train", "test"])]])
        add("blocking A: GPU sparse TF-IDF retrieval, top-10 per view",
            [[Job("blocking_train", ["blocking.py", "train", 10])], [Job("blocking_test", ["blocking.py", "test", 10])]])
        add("blocking B: LightGBM pruner p1 (top-3, p1 >= 0.02 candidates)",
            [[Job("stage1_train", ["stage1.py", "train"])], [Job("stage1_test", ["stage1.py", "test"])]])
        add("pairwise features on the candidates",
            [[Job("features_train", ["match.py", "features", "train"])], [Job("features_test", ["match.py", "features", "test"])]])
        add("training simulations: orphan entities (19%) + near-miss decoy records (8%)",
            [[Job("sim_orphans", ["simulate.py", 0.19, 1])], [Job("sim_decoys", ["simulate.py", "decoys", 0.08, 2])]])
        add("stack without cross-encoders (its out-of-fold p2 selects hard positives for the cross-encoders)",
            one(Job("stack0", ["match.py", "train"], {"USE_CE": "0", "MASK_CE": "0", "TOKSTATS": "0"})))
        add("cross-encoder XLM-R base", [[Job("ce_prep", ["cross_encoder.py", "prep"])]]
            + self.ce_variant("ce", {}, lambda m: {"CE_MODEL": self.m("xlm-roberta-base")}))
        add("cross-encoder XLM-R large", self.ce_variant(
            "cel", {"CE_OUT": "ce_large", "CE_SUFFIX": "_large"},
            lambda m: {"CE_MODEL": self.m("xlm-roberta-large"), "CE_BS": "128", "CE_LR": "1.5e-5"}))
        add("cross-encoder XLM-R large, second epoch per half", self.ce_variant(
            "cel2", {"CE_OUT": "ce_large2", "CE_SUFFIX": "_large2"},
            lambda m: {"CE_MODEL": self.w(f"ce_large/model_{m}"), "CE_BS": "128", "CE_LR": "8e-6"}))
        add("cross-encoder mDeBERTa-v3 base", self.ce_variant(
            "cem", {"CE_OUT": "ce_mdeberta", "CE_SUFFIX": "_mdeberta"},
            lambda m: {"CE_MODEL": self.m("mdeberta-v3-base"), "CE_BS": "256", "CE_LR": "3e-5"}))
        add("four LightGBM stage-2/3 stacks (5-fold by S1)",
            [[Job(f"train_{k}", ["match.py", "train"], v)] for k, v in STACKS.items()])
        add("score the test candidates with every stack",
            [[Job(f"predict_{k}", ["match.py", "predict", "test"], v)] for k, v in STACKS.items()])
        add("per-stack decisions + combine (ruleset v12)", [
            [Job("decide_full", ["make_submission.py"], {**FULL, "BER_OUTPUT": os.path.join(self.stages, "full")})],
            [Job("decide_vf", ["make_submission.py"], {**VF, "BER_OUTPUT": os.path.join(self.stages, "vf")})],
            [Job("combine_v12", ["combine.py", self.st("full"), self.st("vf"), os.path.join(self.stages, "v12"), "v12"])]])
        add("France: structural corrections v13, v15, v16",
            one(Job("france_rules", ["france_rules.py", "rules", self.st("v12"), self.st("v16")])))
        add("France: self-trained French cross-encoder, round 1 (v17)", self.france_round(1))
        add("France: self-trained French cross-encoder, round 2 (v18)", self.france_round(2))
        add("France: v19/v20, re-scored full stack (v21), tighten to v24", [
            [Job("france_final", ["france_rules.py", "final", self.st("v18"), self.w("ce_scores_france2.parquet"), self.st("v20")])],
            [Job("predict_france", ["predict_france.py", self.st("v20"), self.w("ce_scores_france2.parquet"),
                                    self.w("test_scores_frfull.parquet")], FULL)],
            [Job("france_v21", ["france_rules.py", "rescored", self.st("v20"), self.w("ce_scores_france2.parquet"),
                                self.w("test_scores_frfull.parquet"), self.st("v21")])],
            [Job("france_v24", ["france_rules.py", "tighten", self.st("v17"), self.st("v21"), self.w("ce_scores_france.parquet"),
                                self.w("ce_scores_france2.parquet"), self.st("v24")])],
            [Job("blend_v24", ["ensemble_seen.py", self.st("v24"), self.st("v24_blend"), BLEND, THR])]])
        add("widened candidates (US/India top-4, p1 >= 0.01): pairwise features",
            [[Job("wfeats_train", ["widen.py", "feats", "train"])], [Job("wfeats_test", ["widen.py", "feats", "test"])]])
        add("widened candidates: 4 cross-encoders on the new pairs (one GPU per model)",
            [[Job(f"wce_{v}_{s}", ["widen.py", "ce", v, s], {"W_BS": "512", "POLARS_MAX_THREADS": "8"},
                  [self.gpus[i % len(self.gpus)]]) for i, v in enumerate(("ce", "cel", "cel2", "cem")) for s in ("train", "test")]])
        add("widened candidates: re-score with the 4 trained stacks (no retraining)",
            [[Job(f"wscore_{k}_{s}", ["widen.py", "score", s], {**v, "POLARS_MAX_THREADS": "32", "LGB_THREADS": "40"})
              for k, v in STACKS.items()] for s in ("train", "test")])
        add("v25 decisions + final candidate_pairs.tsv", [
            [Job("blend_v25", ["ensemble_seen.py", self.st("v24"), self.st("v25"), BLEND, THR], {"SCORES": "w_test_scores"})],
            [Job("candidates", ["widen.py", "cands", self.st("full", "candidate_pairs.tsv"), os.path.join(self.out, "candidate_pairs.tsv")])]])
        llm = {"LLM_MODEL": self.m("Qwen2.5-1.5B")}
        add("LLM pair classifier (Qwen2.5-1.5B) on the ambiguous band",
            [[Job("llm_prep", ["llm_ce.py", "prep"], llm)]]
            + self.halves(lambda m, p: Job(f"llm_train{m}", ["llm_ce.py", "train", m], llm, p, 2))
            + self.halves(lambda m, p: Job(f"llm_infer{m}", ["llm_ce.py", "infer", m], llm, p, 2))
            + [[Job("llm_merge", ["llm_ce.py", "merge"], llm)]])
        add("stage 4: LightGBM re-decides the band -> final matching_results.tsv",
            one(Job("stage4", ["stage4.py", "predict", THR, self.st("v24"), os.path.join(self.out, "matching_results.tsv")],
                    {"S4_LLM": "llm", "S4_RAW": "1"})))
        add("verify both output files (format, IDs, matches within candidates)",
            one(Job("verify", ["verify_outputs.py", "--output_dir", self.out, "--data_dir", self.data])))
        return S

    # ---------------------------------------------------------------- execution
    def env(self, job):
        e = dict(os.environ)
        e.update({"BER_DATA": self.data, "BER_CACHE": self.work, "BER_OUTPUT": self.out,
                  "BER_GPUS": ",".join(map(str, self.gpus)), "PYTHONHASHSEED": "0", "TOKENIZERS_PARALLELISM": "true",
                  "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                  "PYTHONUNBUFFERED": "1"})
        e.update(job.env)
        if job.gpus is not None:
            e["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, job.gpus))
        elif "CUDA_VISIBLE_DEVICES" not in os.environ:
            e["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))
        return e

    def cmd(self, job):
        if job.nproc:
            self.port += 1
            return [PY, "-m", "torch.distributed.run", f"--nproc_per_node={job.nproc}", f"--master_port={self.port}"] + job.args
        return [PY] + job.args

    def run(self, i, name, groups):
        print(f"[{time.strftime('%H:%M:%S')}] step {i:2d}: {name}", flush=True)
        t_step = time.time()
        for group in groups:
            procs = []
            for job in group:
                log = os.path.join(self.logs, f"{i:02d}_{job.name}.log")
                fh = open(log, "w")
                procs.append((job, subprocess.Popen(self.cmd(job), cwd=SRC, env=self.env(job), stdout=fh, stderr=subprocess.STDOUT),
                              fh, log, time.time()))
            for job, p, fh, log, t0 in procs:
                rc = p.wait()
                fh.close()
                dt = time.time() - t0
                with open(os.path.join(self.logs, "timings.tsv"), "a") as tf:
                    tf.write(f"{i}\t{job.name}\t{dt:.0f}\t{rc}\n")
                print(f"           {job.name}: {'ok' if rc == 0 else 'FAILED'} ({dt / 60:.1f} min)", flush=True)
                if rc != 0:
                    for q in procs:
                        if q[1].poll() is None:
                            q[1].terminate()
                    print(open(log).read()[-4000:], file=sys.stderr)
                    sys.exit(f"step {i} failed: {job.name} (log {log})")
        print(f"           step {i} done in {(time.time() - t_step) / 60:.1f} min", flush=True)


class _ListOnly(Runner):
    """Paths only (no directories created) so that --list works anywhere."""
    def __init__(self, a):
        self.data, self.out, self.work = a.data_dir, a.output_dir, a.work_dir
        self.models, self.stages, self.logs = os.path.join(a.work_dir, "models"), os.path.join(a.work_dir, "stages"), ""
        self.gpus = [int(g) for g in a.gpus.split(",") if g != ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", help="folder with train/ and test/ of the provided dataset")
    ap.add_argument("--output_dir", default="output", help="where candidate_pairs.tsv and matching_results.tsv are written")
    ap.add_argument("--work_dir", default="work", help="intermediate files, models and logs (~250 GB)")
    ap.add_argument("--model_dir", help="pretrained checkpoints (default <work_dir>/models; downloaded in step 1)")
    ap.add_argument("--gpus", default="0,1,2,3", help="physical GPU ids (4 recommended; >= 2 required)")
    ap.add_argument("--from_step", type=int, default=1)
    ap.add_argument("--to_step", type=int, default=99)
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    a = ap.parse_args()
    if a.list:
        a.data_dir = a.data_dir or "."
        for i, (name, _) in enumerate(_ListOnly(a).steps(), 1):
            print(f"{i:2d}  {name}")
        return
    if not a.data_dir:
        ap.error("--data_dir is required")
    for f in ("train/train_source1.tsv", "train/train_ground_truth.tsv", "test/test_source1.tsv"):
        if not os.path.exists(os.path.join(a.data_dir, f)):
            ap.error(f"{f} not found under --data_dir {a.data_dir}")
    r = Runner(a)
    t0 = time.time()
    for i, (name, groups) in enumerate(r.steps(), 1):
        if a.from_step <= i <= a.to_step:
            r.run(i, name, groups)
    print(f"done in {(time.time() - t0) / 3600:.2f} h -> {r.out}", flush=True)


if __name__ == "__main__":
    main()
