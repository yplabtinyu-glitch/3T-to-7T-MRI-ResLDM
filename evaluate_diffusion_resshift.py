"""
evaluate_diffusion_resshift.py -- 完整 T 步 ResShift reverse process 評估，同subject的3T/7T
held-out test pair，複用model.get_input()拿真正的conditioning，
從 c_concat 切出 SPADE-transformed 3T latent 當作 shift anchor y0，
model.p_sample_loop_resshift() 生成，decode後跟真實7T算SSIM/PSNR/LPIPS(brain-masked)/MAE
（SSIM kernel/brain mask沿用train_autoencoder.py的auto_ssim_kernel/build_brain_mask/masked_avg）。

跟 evaluate_diffusion.py 的差異只有三處，用 `diff evaluate_diffusion.py evaluate_diffusion_resshift.py` 就能核對：
  1. --config 預設值換成 resshift 的 yaml
  2. eval_batch() 裡從 cond['c_concat'] 切出 y0，改呼叫 model.p_sample_loop_resshift(y0, cond, shape)
  3. 印出的文字/欄位說明改成 ResShift，並印出 T/kappa 方便核對排程
"""
import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_fn

from ldm.util import instantiate_from_config


def load_autoencoder_utils(script_dir):
    script_dir = os.path.expanduser(script_dir)
    sys.path.insert(0, script_dir)
    import train_autoencoder as ae_mod
    return ae_mod


def load_and_normalize_volume(path):
    vol = nib.load(str(path)).get_fdata(dtype=np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin > 1e-6:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol)
    return vol


def slice_to_tensor(sl_2d, img_size):
    t = torch.from_numpy(np.ascontiguousarray(sl_2d)).float().unsqueeze(0).unsqueeze(0)
    if t.shape[-2:] != (img_size, img_size):
        t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return t * 2.0 - 1.0


def iter_test_slices(test_dir, subject, img_size):
    src_path = os.path.join(test_dir, "reg_3T", f"{subject}.nii.gz")
    tgt_path = os.path.join(test_dir, "7T", f"{subject}.nii.gz")
    src_vol = load_and_normalize_volume(src_path)
    tgt_vol = load_and_normalize_volume(tgt_path)

    if src_vol.shape != tgt_vol.shape:
        raise RuntimeError(
            f"subject {subject}: 3T shape {src_vol.shape} != 7T shape {tgt_vol.shape}"
        )

    n_slices = src_vol.shape[2]
    lo = int(n_slices * 0.2)
    hi = int(n_slices * 0.8)
    for s in range(lo, hi):
        yield slice_to_tensor(src_vol[:, :, s], img_size), slice_to_tensor(tgt_vol[:, :, s], img_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/latent-diffusion/mri-3t7t-resshift.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--target_class", type=int, default=1)
    ap.add_argument("--autoencoder_script_dir", default=".")
    ap.add_argument("--num_patches", type=int, default=None)
    ap.add_argument("--brain_mask_threshold", type=float, default=0.02)
    ap.add_argument("--out_csv", default="diffusion_resshift_eval_results.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae_mod = load_autoencoder_utils(args.autoencoder_script_dir)

    config = OmegaConf.load(args.config)

    ae_ckpt_path = os.path.expanduser(config.model.params.first_stage_config.params.ae_checkpoint_path)
    ae_meta = torch.load(ae_ckpt_path, map_location="cpu", weights_only=False)
    ae_args = ae_meta.get("args", {})
    img_size = args.img_size if args.img_size is not None else ae_args.get("img_size", 128)
    num_patches = args.num_patches if args.num_patches is not None else ae_args.get("num_patches", 24)
    print(f"img_size={img_size}, num_patches={num_patches} (from {ae_ckpt_path}'s saved args)")

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg", spatial=True).to(device)
        lpips_fn.eval()
        for p_ in lpips_fn.parameters():
            p_.requires_grad_(False)
    except ImportError:
        print("[warn] lpips not installed -- LPIPS will be NaN. pip install lpips")
        lpips_fn = None

    ssim_kernel = ae_mod.auto_ssim_kernel(img_size, num_patches)

    model = instantiate_from_config(config.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', 'n/a')})")

    latent_channels = config.model.params.channels
    latent_size = config.model.params.image_size
    n_timesteps = model.num_timesteps
    kappa = getattr(model, "kappa", None)
    print(f"sampling with the full {n_timesteps}-step ResShift reverse process "
          f"(model.p_sample_loop_resshift, kappa={kappa}) -- anchored at the SPADE-transformed "
          f"3T latent y0 (same y0 used as the ResShift shift target during training), no DDPM/DDIM")

    def eval_batch(x_src, x_tgt):
        b = x_src.shape[0]
        y = torch.full((b,), args.target_class, dtype=torch.long)
        batch = {"source": x_src.to(device), "target": x_tgt.to(device), "target_class": y}

        with torch.no_grad():
            _, cond = model.get_input(batch, model.first_stage_key)
            y0 = cond['c_concat'][:, 4:8, ...].clone()  # SPADE-transformed 3T latent, same slice as training's get_input_resshift
            shape = (b, latent_channels, latent_size, latent_size)
            z_pred = model.p_sample_loop_resshift(y0, cond, shape, verbose=False)
            gen = model.decode_first_stage(z_pred).clamp(-1, 1)

        gen_f = gen.float()
        tgt_f = x_tgt.to(device).float()
        gen_01 = (gen_f + 1) / 2
        tgt_01 = (tgt_f.clamp(-1, 1) + 1) / 2

        ssim_val = ssim_fn(gen_01, tgt_01, data_range=1.0,
                            kernel_size=ssim_kernel[0], sigma=ssim_kernel[1]).item()
        psnr_val = psnr_fn(gen_01, tgt_01, data_range=1.0).item()
        mae_val = (gen_f - tgt_f).abs().mean().item()

        lpips_val = float("nan")
        if lpips_fn is not None:
            brain_mask = ae_mod.build_brain_mask(tgt_01, threshold=args.brain_mask_threshold)
            gen_rgb = gen_f.repeat(1, 3, 1, 1)
            tgt_rgb = tgt_f.repeat(1, 3, 1, 1)
            with torch.no_grad():
                lpips_map = lpips_fn(gen_rgb, tgt_rgb)
            lpips_t = ae_mod.masked_avg(lpips_map, brain_mask)
            if torch.isfinite(lpips_t):
                lpips_val = lpips_t.item()

        return {"ssim": ssim_val, "psnr": psnr_val, "mae": mae_val, "lpips": lpips_val, "n": b}

    per_subject_rows = []
    grand_sums = {"ssim": 0.0, "psnr": 0.0, "mae": 0.0, "lpips": 0.0, "lpips_n": 0, "n": 0}

    for subject in args.subjects:
        src_buf, tgt_buf = [], []
        sums = {"ssim": 0.0, "psnr": 0.0, "mae": 0.0, "lpips": 0.0, "lpips_n": 0, "n": 0}

        def flush():
            if not src_buf:
                return
            x_src = torch.cat(src_buf, dim=0)
            x_tgt = torch.cat(tgt_buf, dim=0)
            r = eval_batch(x_src, x_tgt)
            n = r["n"]
            sums["ssim"] += r["ssim"] * n
            sums["psnr"] += r["psnr"] * n
            sums["mae"] += r["mae"] * n
            if r["lpips"] == r["lpips"]:
                sums["lpips"] += r["lpips"] * n
                sums["lpips_n"] += n
            sums["n"] += n

        for src_t, tgt_t in iter_test_slices(args.test_dir, subject, img_size):
            src_buf.append(src_t); tgt_buf.append(tgt_t)
            if len(src_buf) == args.batch_size:
                flush()
                src_buf, tgt_buf = [], []
        flush()

        n = sums["n"]
        row = {
            "subject": subject, "n_slices": n,
            "ssim": sums["ssim"] / n if n else float("nan"),
            "psnr": sums["psnr"] / n if n else float("nan"),
            "mae": sums["mae"] / n if n else float("nan"),
            "lpips": (sums["lpips"] / sums["lpips_n"]) if sums["lpips_n"] else float("nan"),
        }
        per_subject_rows.append(row)
        print(f"[{subject}] n={n}  SSIM={row['ssim']:.4f}  PSNR={row['psnr']:.2f}  "
              f"MAE={row['mae']:.4f}  LPIPS(brain)={row['lpips']:.4f}")

        for k in ("ssim", "psnr", "mae"):
            grand_sums[k] += sums[k]
        grand_sums["lpips"] += sums["lpips"]
        grand_sums["lpips_n"] += sums["lpips_n"]
        grand_sums["n"] += n

    gn = grand_sums["n"]
    print("\n=== overall (all subjects, slice-weighted) ===")
    print(f"SSIM={grand_sums['ssim']/gn:.4f}  PSNR={grand_sums['psnr']/gn:.2f}  "
          f"MAE={grand_sums['mae']/gn:.4f}  "
          f"LPIPS(brain)={(grand_sums['lpips']/grand_sums['lpips_n']) if grand_sums['lpips_n'] else float('nan'):.4f}"
          f"  (n={gn} slices, {n_timesteps}-step ResShift sampling)")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "n_slices", "ssim", "psnr", "mae", "lpips"])
        w.writeheader()
        for row in per_subject_rows:
            w.writerow(row)
    print(f"\nsaved per-subject results to {args.out_csv}")


if __name__ == "__main__":
    main()
