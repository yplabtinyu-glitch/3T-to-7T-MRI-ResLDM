import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_fn

from ldm.util import instantiate_from_config
from evaluate_diffusion_resshift_v2 import load_and_normalize_volume, slice_to_tensor, load_autoencoder_utils
from hist_match import match_volume

KEYS = ("ssim", "psnr", "mae", "mae_brain", "lpips")


def load_spade_weights(model, path):
    sd = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
    for key in ("spade", "model", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    sd = {k[len("first_stage_model.spade."):] if k.startswith("first_stage_model.spade.") else k: v for k, v in sd.items()}
    model.first_stage_model.spade.load_state_dict(sd, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/latent-diffusion/mri-3t7t-resshift-ablation-nocondition.yaml",
                    help="only its AE / SPADE / scale settings are used")
    ap.add_argument("--spade", action="append", required=True, help='"label=path", repeatable')
    ap.add_argument("--test_dir", default=os.path.expanduser("~/Desktop/test"))
    ap.add_argument("--subjects", nargs="+", default=["212730", "212731", "216393", "216394"])
    ap.add_argument("--hist_match", choices=["none", "3t", "both"], default="both")
    ap.add_argument("--hist_ref", default="hist_ref_train.npz")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--vis_subject", default=None)
    ap.add_argument("--vis_slices", type=int, nargs="*", default=None, help="default: 3 slices around the middle")
    ap.add_argument("--rot", type=int, default=0, help="np.rot90 k, display only")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--target_class", type=int, default=1)
    ap.add_argument("--brain_mask_threshold", type=float, default=0.02)
    ap.add_argument("--autoencoder_script_dir", default=".")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae_mod = load_autoencoder_utils(args.autoencoder_script_dir)
    config = OmegaConf.load(args.config)
    ae_ckpt = os.path.expanduser(config.model.params.first_stage_config.params.ae_checkpoint_path)
    ae_args = torch.load(ae_ckpt, map_location="cpu", weights_only=False).get("args", {})
    img_size = ae_args.get("img_size", 128)
    kernel = ae_mod.auto_ssim_kernel(img_size, ae_args.get("num_patches", 24))
    model = instantiate_from_config(config.model).to(device).eval()
    print(f"img_size={img_size}, SSIM kernel={kernel}, hist_match={args.hist_match}")

    HM = None
    if args.hist_match != "none":
        r = np.load(args.hist_ref)
        HM = {"lm_3t": r["lm_3t"], "lm_7t": r["lm_7t"], "thr": float(r["thr"])}

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg", spatial=True).to(device).eval()
    except ImportError:
        lpips_fn = None
        print("[warn] lpips not installed")

    spades = [s.split("=", 1) for s in args.spade]
    for lab, p in spades:
        if not os.path.exists(os.path.expanduser(p)):
            raise SystemExit(f"not found: {lab} = {p}")
    panel_names = ["3T (AE recon)"] + [lab for lab, _ in spades] + ["7T (AE recon)"]

    def load_vols(subject):
        src = load_and_normalize_volume(os.path.join(args.test_dir, "reg_3T", f"{subject}.nii.gz"))
        tgt = load_and_normalize_volume(os.path.join(args.test_dir, "7T", f"{subject}.nii.gz"))
        if HM is not None:
            src, _ = match_volume(src, HM["lm_3t"], HM["thr"])
            if args.hist_match == "both":
                tgt, _ = match_volume(tgt, HM["lm_7t"], HM["thr"])
        return src, tgt

    @torch.no_grad()
    def predict(x_src, x_tgt):
        batch = {"source": x_src.to(device), "target": x_tgt.to(device),
                 "target_class": torch.full((x_src.shape[0],), args.target_class, dtype=torch.long)}
        out = {}
        for lab, path in spades:
            load_spade_weights(model, path)
            x0, cond = model.get_input(batch, model.first_stage_key)
            c = cond["c_concat"]
            out[lab] = (model.decode_first_stage(c[:, 4:8]).clamp(-1, 1).float() + 1) / 2
        out["3T (AE recon)"] = (model.decode_first_stage(c[:, 0:4]).clamp(-1, 1).float() + 1) / 2
        out["7T (AE recon)"] = (model.decode_first_stage(x0).clamp(-1, 1).float() + 1) / 2
        return out

    def batch_metrics(gen01, tgt01):
        mask = ae_mod.build_brain_mask(tgt01, threshold=args.brain_mask_threshold)
        ad = (gen01 - tgt01).abs()
        r = {"ssim": ssim_fn(gen01, tgt01, data_range=1.0, kernel_size=kernel[0], sigma=kernel[1]).item(),
             "psnr": psnr_fn(gen01, tgt01, data_range=1.0, dim=(1, 2, 3), reduction="elementwise_mean").item(),
             "mae": ad.mean().item(), "mae_brain": ae_mod.masked_avg(ad, mask).item(), "lpips": float("nan")}
        if lpips_fn is not None:
            with torch.no_grad():
                lm = lpips_fn((gen01 * 2 - 1).repeat(1, 3, 1, 1), (tgt01 * 2 - 1).repeat(1, 3, 1, 1))
            r["lpips"] = ae_mod.masked_avg(lm, mask).item()
        return r

    if args.full:
        grand = {p: {k: 0.0 for k in KEYS} | {k + "_n": 0 for k in KEYS} | {"n": 0} for p in panel_names}
        for subject in args.subjects:
            src, tgt = load_vols(subject)
            n = src.shape[2]
            idx = list(range(int(n * 0.2), int(n * 0.8)))
            for b in range(0, len(idx), args.batch_size):
                ii = idx[b:b + args.batch_size]
                xs = torch.cat([slice_to_tensor(src[:, :, i], img_size) for i in ii])
                xt = torch.cat([slice_to_tensor(tgt[:, :, i], img_size) for i in ii])
                tgt01 = (xt.to(device).float().clamp(-1, 1) + 1) / 2
                for p, img in predict(xs, xt).items():
                    r = batch_metrics(img, tgt01)
                    for k in KEYS:
                        if r[k] == r[k]:  # skip NaN batches, same as evaluate_diffusion_resshift_v2.py
                            grand[p][k] += r[k] * len(ii)
                            grand[p][k + "_n"] += len(ii)
                    grand[p]["n"] += len(ii)
            print(f"  done {subject}")
        out_csv = f"compare_spade_hm{args.hist_match}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["panel", "n_slices", *KEYS])
            print(f"\n=== overall vs 7T (hist_match={args.hist_match}) ===")
            print(f"{'panel':24s} " + " ".join(f"{k:>9s}" for k in KEYS))
            for p in panel_names:
                g = grand[p]
                vals = [g[k] / g[k + "_n"] if g[k + "_n"] else float("nan") for k in KEYS]
                w.writerow([p, g["n"], *vals])
                print(f"{p:24s} " + " ".join(f"{v:9.4f}" for v in vals))
        print(f"saved {out_csv}")

    if args.vis_subject:
        src, tgt = load_vols(args.vis_subject)
        if not args.vis_slices:
            n = src.shape[2]
            args.vis_slices = [n // 2 - n // 8, n // 2, n // 2 + n // 8]
        xs = torch.cat([slice_to_tensor(src[:, :, i], img_size) for i in args.vis_slices])
        xt = torch.cat([slice_to_tensor(tgt[:, :, i], img_size) for i in args.vis_slices])
        preds = predict(xs, xt)
        src01 = (xs.clamp(-1, 1) + 1) / 2
        tgt01 = (xt.to(device).float().clamp(-1, 1) + 1) / 2
        cols = [("3T input", None)] + [(p, preds[p]) for p in panel_names] + [("7T (ground truth)", None)]
        nr, nc = 2 * len(args.vis_slices), len(cols)
        fig, ax = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.5 * nr), dpi=150, squeeze=False)
        to_np = lambda t: np.rot90(t.float().cpu().numpy(), k=args.rot)
        im = None
        for r, s in enumerate(args.vis_slices):
            t01 = tgt01[r:r + 1]
            for j, (title, imgs) in enumerate(cols):
                a, e = ax[2 * r, j], ax[2 * r + 1, j]
                if imgs is None:
                    img = src01[r, 0] if title == "3T input" else t01[0, 0]
                    a.imshow(to_np(img), cmap="gray", vmin=0, vmax=1)
                    e.set_visible(False)
                else:
                    g = imgs[r:r + 1]
                    m = batch_metrics(g, t01)
                    a.imshow(to_np(g[0, 0]), cmap="gray", vmin=0, vmax=1)
                    a.set_xlabel(f"SSIM {m['ssim']:.3f}  MAE(b) {m['mae_brain']:.3f}\nLPIPS {m['lpips']:.3f}", fontsize=7)
                    im = e.imshow(to_np((g - t01).abs()[0, 0]), cmap="magma", vmin=0, vmax=0.3)
                a.set_title(title if r == 0 else "", fontsize=8.5)
                if j == 0:
                    a.set_ylabel(f"slice {s}", fontsize=8.5)
                for x in (a, e):
                    x.set_xticks([]); x.set_yticks([])
        if im is not None:
            fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.012, pad=0.01, label="|pred - 7T|")
        fig.suptitle(f"subject {args.vis_subject}  (hist_match={args.hist_match}; all panels = AE decode)", fontsize=10)
        out = f"compare_spade_{args.vis_subject}_hm{args.hist_match}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
