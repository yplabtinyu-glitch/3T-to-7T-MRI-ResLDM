import argparse
import csv
import glob
import os
import re
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_fn

from ldm.util import instantiate_from_config
from evaluate_diffusion_resshift_v2 import load_and_normalize_volume, slice_to_tensor, load_autoencoder_utils
from hist_match import match_volume

KEYS = ("ssim", "psnr", "mae", "mae_brain", "lpips")


def load_spade_weights(model, path, retries=6):
    for i in range(retries):
        try:
            sd = torch.load(path, map_location="cpu", weights_only=False)
            break
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"  [retry] could not read {os.path.basename(path)} yet ({type(e).__name__}); waiting 10 s")
            time.sleep(10)
    for key in ("spade", "model", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    sd = {k[len("first_stage_model.spade."):] if k.startswith("first_stage_model.spade.") else k: v for k, v in sd.items()}
    model.first_stage_model.spade.load_state_dict(sd, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spade_dir", required=True)
    ap.add_argument("--config", default="configs/latent-diffusion/mri-3t7t-resshift-ablation-nocondition.yaml")
    ap.add_argument("--test_dir", default=os.path.expanduser("~/Desktop/test"))
    ap.add_argument("--test_subjects", nargs="+", default=["212730", "212731", "216393", "216394"])
    ap.add_argument("--valid_dir", default=None)
    ap.add_argument("--valid_subjects", nargs="+", default=None)
    ap.add_argument("--hist_match", choices=["none", "3t", "both"], default="both")
    ap.add_argument("--hist_ref", default="hist_ref_train.npz")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--target_class", type=int, default=1)
    ap.add_argument("--brain_mask_threshold", type=float, default=0.02)
    ap.add_argument("--autoencoder_script_dir", default=".")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--stop_after", type=int, default=None)
    args = ap.parse_args()
    spade_dir = os.path.expanduser(args.spade_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae_mod = load_autoencoder_utils(args.autoencoder_script_dir)
    config = OmegaConf.load(args.config)
    ae_ckpt = os.path.expanduser(config.model.params.first_stage_config.params.ae_checkpoint_path)
    ae_args = torch.load(ae_ckpt, map_location="cpu", weights_only=False).get("args", {})
    img_size = ae_args.get("img_size", 128)
    kernel = ae_mod.auto_ssim_kernel(img_size, ae_args.get("num_patches", 24))
    model = instantiate_from_config(config.model).to(device).eval()
    import lpips
    lpips_fn = lpips.LPIPS(net="vgg", spatial=True).to(device).eval()
    HM = None
    if args.hist_match != "none":
        r = np.load(args.hist_ref)
        HM = {"lm_3t": r["lm_3t"], "lm_7t": r["lm_7t"], "thr": float(r["thr"])}
    print(f"SSIM kernel={kernel}, hist_match={args.hist_match}, watching {spade_dir}")

    def load_set(test_dir, subjects):
        out = []
        for s in subjects:
            src = load_and_normalize_volume(os.path.join(test_dir, "reg_3T", f"{s}.nii.gz"))
            tgt = load_and_normalize_volume(os.path.join(test_dir, "7T", f"{s}.nii.gz"))
            if HM is not None:
                src, _ = match_volume(src, HM["lm_3t"], HM["thr"])
                if args.hist_match == "both":
                    tgt, _ = match_volume(tgt, HM["lm_7t"], HM["thr"])
            n = src.shape[2]
            idx = list(range(int(n * 0.2), int(n * 0.8)))
            for b in range(0, len(idx), args.batch_size):
                ii = idx[b:b + args.batch_size]
                out.append((torch.cat([slice_to_tensor(src[:, :, i], img_size) for i in ii]),
                            torch.cat([slice_to_tensor(tgt[:, :, i], img_size) for i in ii])))
        return out

    sets = {"test": load_set(args.test_dir, args.test_subjects)}
    if args.valid_dir and args.valid_subjects:
        sets["valid"] = load_set(args.valid_dir, args.valid_subjects)
    print("eval slices: " + ", ".join(f"{k}={sum(len(b[0]) for b in v)}" for k, v in sets.items()))

    @torch.no_grad()
    def evaluate(batches):
        g = {k: 0.0 for k in KEYS} | {k + "_n": 0 for k in KEYS}
        for xs, xt in batches:
            batch = {"source": xs.to(device), "target": xt.to(device),
                     "target_class": torch.full((len(xs),), args.target_class, dtype=torch.long)}
            _, cond = model.get_input(batch, model.first_stage_key)
            gen01 = (model.decode_first_stage(cond["c_concat"][:, 4:8]).clamp(-1, 1).float() + 1) / 2
            tgt01 = (xt.to(device).float().clamp(-1, 1) + 1) / 2
            mask = ae_mod.build_brain_mask(tgt01, threshold=args.brain_mask_threshold)
            ad = (gen01 - tgt01).abs()
            lm = lpips_fn((gen01 * 2 - 1).repeat(1, 3, 1, 1), (tgt01 * 2 - 1).repeat(1, 3, 1, 1))
            r = {"ssim": ssim_fn(gen01, tgt01, data_range=1.0, kernel_size=kernel[0], sigma=kernel[1]).item(),
                 "psnr": psnr_fn(gen01, tgt01, data_range=1.0, dim=(1, 2, 3), reduction="elementwise_mean").item(),
                 "mae": ad.mean().item(), "mae_brain": ae_mod.masked_avg(ad, mask).item(),
                 "lpips": ae_mod.masked_avg(lm, mask).item()}
            for k in KEYS:
                if r[k] == r[k]:
                    g[k] += r[k] * len(xs)
                    g[k + "_n"] += len(xs)
        return {k: g[k] / g[k + "_n"] if g[k + "_n"] else float("nan") for k in KEYS}

    met_path = os.path.join(spade_dir, "test_metrics.csv")
    cols = ["epoch"] + [f"{s}_{k}" for s in sets for k in KEYS]
    done = {}
    if os.path.exists(met_path):
        with open(met_path) as f:
            for row in csv.DictReader(f):
                done[int(row["epoch"])] = row

    def write_outputs():
        with open(met_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for ep in sorted(done):
                w.writerow(done[ep])
        log_path = os.path.join(spade_dir, "log.csv")
        if not os.path.exists(log_path):
            return
        with open(log_path) as f:
            log_rows = list(csv.DictReader(f))
        if not log_rows:
            return
        fields = list(log_rows[0].keys()) + [c for c in cols if c != "epoch"]
        with open(os.path.join(spade_dir, "log_with_test.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in log_rows:
                ep = int(float(row["epoch"]))
                w.writerow({**row, **{c: done[ep][c] for c in cols if c != "epoch" and ep in done}})

    while True:
        found = {}
        for p in glob.glob(os.path.join(spade_dir, "spade_epoch*.pth")):
            m = re.search(r"spade_epoch(\d+)\.pth$", p)
            if m:
                found[int(m.group(1))] = p
        new = sorted(ep for ep in found if ep not in done)
        for ep in new:
            if time.time() - os.path.getmtime(found[ep]) < 15:
                time.sleep(15)
            t0 = time.time()
            load_spade_weights(model, found[ep])
            row = {"epoch": ep}
            for s, batches in sets.items():
                for k, v in evaluate(batches).items():
                    row[f"{s}_{k}"] = v
            done[ep] = row
            write_outputs()
            print(f"[epoch {ep}] " + "  ".join(f"{c}={row[c]:.4f}" for c in cols if c != "epoch")
                  + f"  ({time.time() - t0:.0f} s)", flush=True)
        if not new:
            write_outputs()
        if args.stop_after is not None and args.stop_after in done:
            print(f"evaluated epoch {args.stop_after} -- stopping")
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
