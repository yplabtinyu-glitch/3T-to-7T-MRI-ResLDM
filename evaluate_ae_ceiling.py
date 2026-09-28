"""
evaluate_ae_ceiling.py -- 診斷腳本(1/2): AE本身的重建天花板。

完全跳過 SPADE 跟 diffusion:拿真正的7T測試影像 -> frozen AE encode -> 立刻
decode(不經過任何domain轉換、任何去噪) -> 跟原本的7T影像比SSIM/PSNR/MAE/LPIPS。

這是「任何方法透過這個latent space能達到的理論上限」-- 不管SPADE轉得多準、
diffusion訓練得多好,最終輸出都要先被AE的encoder->decoder這條路徑過濾一次,
所以AE自己的重建品質就是整個pipeline的天花板,不可能有下游步驟超過它。

跟 evaluate_diffusion_resshift.py 用完全一樣的:
  - load_and_normalize_volume / slice_to_tensor / iter_test_slices (同樣的
    per-volume min-max normalize、同樣的20%-80% slice range)
  - train_autoencoder.py 的 auto_ssim_kernel / build_brain_mask / masked_avg
  - 同樣的4個測試subject、同樣的test_dir結構(test_dir/3T/{subject}.nii.gz,
    test_dir/7T/{subject}.nii.gz -- 這裡只用得到7T那份)
確保這個天花板數字跟diffusion的評估結果是在同一套方法論下算出來的,可以直接
放在同一張表裡比較,不是各算各的。

不需要 --checkpoint(不涉及diffusion UNet,也不load diffusion checkpoint),
只需要 config yaml 裡 first_stage_config.params.ae_checkpoint_path 這一個路徑。
直接建構 FrozenAEWithSPADE 並且只呼叫它的 .encode()/.decode(),完全不碰
.spade()、也不牽涉diffusion的scale_factor(那是給diffusion的noise process用的
正規化,對AE自己的encode->decode round trip沒有影響:decode吃的就是encode吐出
來的同一個raw latent scale)。
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

from ldm.models.frozen_ae_spade import FrozenAEWithSPADE


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
    # 這支腳本是AE的重建天花板，只用得到7T，完全不碰3T -- 不像
    # evaluate_diffusion_resshift.py/evaluate_spade_ceiling.py需要3T/7T同一個
    # slice index對齊，所以這裡故意不讀3T、也不做shape比對(3T raw跟7T本來就
    # shape不同，之前已經確認過，那是3T/7T之間registration的問題，跟AE自己
    # 重建7T的能力無關，不該讓這支腳本被那個問題擋住)。
    tgt_path = os.path.join(test_dir, "7T", f"{subject}.nii.gz")
    tgt_vol = load_and_normalize_volume(tgt_path)

    n_slices = tgt_vol.shape[2]
    lo = int(n_slices * 0.2)
    hi = int(n_slices * 0.8)
    for s in range(lo, hi):
        yield slice_to_tensor(tgt_vol[:, :, s], img_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/latent-diffusion/mri-3t7t-resshift.yaml",
                     help="只用來讀 first_stage_config.params 裡的 ae_checkpoint_path")
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--autoencoder_script_dir", default=".")
    ap.add_argument("--num_patches", type=int, default=None)
    ap.add_argument("--brain_mask_threshold", type=float, default=0.02)
    ap.add_argument("--out_csv", default="ae_ceiling_eval_results.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae_mod = load_autoencoder_utils(args.autoencoder_script_dir)

    config = OmegaConf.load(args.config)
    fsc_params = config.model.params.first_stage_config.params
    ae_ckpt_path = os.path.expanduser(fsc_params.ae_checkpoint_path)

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

    # 只需要frozen AE的encode/decode，不碰spade()、不建diffusion UNet。
    # spade_checkpoint_path留空:反正這支腳本完全不呼叫.spade()。
    ae_wrap = FrozenAEWithSPADE(ae_checkpoint_path=fsc_params.ae_checkpoint_path,
                                 num_classes=fsc_params.get("num_classes", 2)).to(device)
    ae_wrap.eval()
    print(f"loaded frozen AE from {ae_ckpt_path} (SPADE weights NOT loaded -- unused in this script)")

    def eval_batch(x_tgt):
        with torch.no_grad():
            z, _, _ = ae_wrap.encode(x_tgt.to(device))
            recon = ae_wrap.decode(z).clamp(-1, 1)

        gen_f = recon.float()
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

        return {"ssim": ssim_val, "psnr": psnr_val, "mae": mae_val, "lpips": lpips_val, "n": x_tgt.shape[0]}

    per_subject_rows = []
    grand_sums = {"ssim": 0.0, "psnr": 0.0, "mae": 0.0, "lpips": 0.0, "lpips_n": 0, "n": 0}

    for subject in args.subjects:
        tgt_buf = []
        sums = {"ssim": 0.0, "psnr": 0.0, "mae": 0.0, "lpips": 0.0, "lpips_n": 0, "n": 0}

        def flush():
            if not tgt_buf:
                return
            x_tgt = torch.cat(tgt_buf, dim=0)
            r = eval_batch(x_tgt)
            n = r["n"]
            sums["ssim"] += r["ssim"] * n
            sums["psnr"] += r["psnr"] * n
            sums["mae"] += r["mae"] * n
            if r["lpips"] == r["lpips"]:
                sums["lpips"] += r["lpips"] * n
                sums["lpips_n"] += n
            sums["n"] += n

        for tgt_t in iter_test_slices(args.test_dir, subject, img_size):
            tgt_buf.append(tgt_t)
            if len(tgt_buf) == args.batch_size:
                flush()
                tgt_buf = []
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
    print("\n=== overall (all subjects, slice-weighted) -- AE reconstruction ceiling (no SPADE, no diffusion) ===")
    print(f"SSIM={grand_sums['ssim']/gn:.4f}  PSNR={grand_sums['psnr']/gn:.2f}  "
          f"MAE={grand_sums['mae']/gn:.4f}  "
          f"LPIPS(brain)={(grand_sums['lpips']/grand_sums['lpips_n']) if grand_sums['lpips_n'] else float('nan'):.4f}"
          f"  (n={gn} slices, 7T->AE encode->decode, real vs. self round-trip)")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "n_slices", "ssim", "psnr", "mae", "lpips"])
        w.writeheader()
        for row in per_subject_rows:
            w.writerow(row)
    print(f"\nsaved per-subject results to {args.out_csv}")


if __name__ == "__main__":
    main()
