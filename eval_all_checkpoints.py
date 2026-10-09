import argparse, glob, os, re, subprocess, sys
import pandas as pd

RUNS = {
    "no_condition":  "checkpoints_diffusion_ablation_nocondition",
    "3T_only":       "checkpoints_diffusion_ablation_3t",
    "SPADE_only":    "checkpoints_diffusion_ablation_spade",
    "all_condition": "checkpoints_diffusion_resshift_perchkappa",
    "y0_3T":         "checkpoints_diffusion_ablation_nocondition_y03t",
    "3T_only_k05":   "checkpoints_diffusion_ablation_3t_k05",
}
ap = argparse.ArgumentParser()
ap.add_argument("--test_dir", default=os.path.expanduser("~/Desktop/test"))
ap.add_argument("--subjects", nargs="+", default=["212730", "212731", "216393", "216394"])
ap.add_argument("--autoencoder_script_dir", default=".")
ap.add_argument("--out_dir", default="eval_epochs")
ap.add_argument("--runs", nargs="+", default=list(RUNS))
ap.add_argument("--hist_match", choices=["none", "3t", "both"], default="none")
ap.add_argument("--dry_run", action="store_true")
args = ap.parse_args()
SUFFIX = "" if args.hist_match == "none" else f"_hm{args.hist_match}"
if SUFFIX and args.out_dir == "eval_epochs":
    args.out_dir = "eval_epochs" + SUFFIX   # keep results of different modes apart
print(f"hist_match={args.hist_match}, per-checkpoint csv -> {args.out_dir}/")
os.makedirs(args.out_dir, exist_ok=True)

jobs = []
for run in args.runs:
    d = RUNS[run]
    cks = []
    for p in glob.glob(os.path.join(d, "diffusion_epoch*.pth")):
        m = re.search(r"diffusion_epoch(\d+)\.pth$", p)
        if m:
            cks.append((int(m.group(1)), p))
    if not cks:
        print(f"[warn] {run}: no diffusion_epoch*.pth in {d}")
    for ep, p in sorted(cks):
        jobs.append((run, ep, p, os.path.join(args.out_dir, f"{run}_ep{ep}.csv")))

todo = [j for j in jobs if not os.path.exists(j[3])]
print(f"{len(jobs)} checkpoints total, {len(jobs) - len(todo)} already done, {len(todo)} to run")
for run in args.runs:
    print(f"  {run:14s} epochs: {[ep for r, ep, _, _ in jobs if r == run]}")
if args.dry_run:
    sys.exit(0)

for k, (run, ep, ckpt, out) in enumerate(todo, 1):
    cfg = subprocess.check_output([sys.executable, "-c",
        f"import torch; print(torch.load({ckpt!r}, map_location='cpu', weights_only=False)['args']['config'])"],
        text=True).strip()
    print(f"\n[{k}/{len(todo)}] {run} epoch {ep}  ({cfg})", flush=True)
    tmp = out + ".tmp"
    r = subprocess.run([sys.executable, "evaluate_diffusion_resshift_v2.py",
                        "--config", cfg, "--checkpoint", ckpt,
                        "--test_dir", args.test_dir, "--subjects", *args.subjects,
                        "--autoencoder_script_dir", args.autoencoder_script_dir,
                        "--no_ema", "--hist_match", args.hist_match, "--out_csv", tmp])
    if r.returncode != 0:
        print(f"[error] {run} epoch {ep} failed (exit {r.returncode}) -- skipped, rerun later")
        continue
    os.replace(tmp, out)

rows, subj_rows = [], []
M = ("ssim", "psnr", "mae", "mae_brain", "lpips")
for run, ep, ckpt, out in jobs:
    if not os.path.exists(out):
        continue
    df = pd.read_csv(out)
    ov = df[df["subject"].astype(str) == "overall"].iloc[0]
    rows.append({"run": run, "epoch": ep, "n_slices": int(ov["n_slices"]), **{m: float(ov[m]) for m in M}})
    for _, s in df[df["subject"].astype(str) != "overall"].iterrows():
        subj_rows.append({"run": run, "epoch": ep, "subject": s["subject"], "n_slices": int(s["n_slices"]),
                          **{m: float(s[m]) for m in M}})
if rows:
    pd.DataFrame(rows).to_csv(f"eval_all_epochs{SUFFIX}.csv", index=False)
    pd.DataFrame(subj_rows).to_csv(f"eval_all_epochs_per_subject{SUFFIX}.csv", index=False)
    print(f"\nsaved eval_all_epochs{SUFFIX}.csv ({len(rows)} rows) and eval_all_epochs_per_subject{SUFFIX}.csv")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
