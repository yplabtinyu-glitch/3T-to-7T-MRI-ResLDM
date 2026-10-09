"""
evaluate_diffusion_resshift_v2.py -- ResShift 完整 T 步 reverse process 評估（支援 ablation_no_condition）

跟 evaluate_diffusion_resshift.py 的差異：
  1. ablation_no_condition / 非 class-conditional 模型：
     包一層 UNet forward wrapper ——
       (a) UNet 沒有 num_classes 時，把 y 丟掉（修 "must specify y if and only if..." assert）
       (b) 輸入通道數 > UNet.in_channels 時，只保留前 in_channels 個通道
           （DiffusionWrapper 的 concat 是 cat([x_t] + c_concat)，前 4 ch = x_t，等於不吃 condition）
     第一次呼叫會印出實際做了什麼，方便核對。
  2. y0 仍從 cond['c_concat'][:, 4:8] 切（SPADE-transformed 3T latent），跟主模型相同。
     ※ 若 resshift.py 第 171 行的 ablation 分支改用別的 y0，這裡要跟著改 ——
       核對：sed -n 160,200p ldm/models/diffusion/resshift.py
  3. PSNR：每張 slice 各算再平均（dim=(1,2,3)），不再受 batch_size 影響
  4. MAE：改在 [0,1] 算（跟 SSIM/PSNR 同尺度）；另外輸出 brain-masked MAE
  5. 有 use_ema 時用 model.ema_scope() 取樣
  6. 固定 seed（--seed，預設 0）
  ※ 2–4 會讓數字跟舊腳本不同（PSNR 會變高、MAE 約減半），舊 checkpoint 要用本檔重跑才能比。
"""
import argparse
import contextlib
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


def iter_test_slices(test_dir, subject, img_size, hm=None):
    src_vol = load_and_normalize_volume(os.path.join(test_dir, "reg_3T", f"{subject}.nii.gz"))
    tgt_vol = load_and_normalize_volume(os.path.join(test_dir, "7T", f"{subject}.nii.gz"))
    if src_vol.shape != tgt_vol.shape:
        raise RuntimeError(f"subject {subject}: 3T shape {src_vol.shape} != 7T shape {tgt_vol.shape}")
    if hm is not None:  # Nyul histogram matching of brain voxels to the training-set landmarks
        from hist_match import match_volume
        src_vol, lm = match_volume(src_vol, hm["lm_3t"], hm["thr"])
        msg = f"[hist_match] {subject} 3T p50 {lm[5]:.3f}->{hm['lm_3t'][5]:.3f}"
        if hm["mode"] == "both":
            tgt_vol, lm = match_volume(tgt_vol, hm["lm_7t"], hm["thr"])
            msg += f" | 7T p50 {lm[5]:.3f}->{hm['lm_7t'][5]:.3f}"
        print(msg)
    n_slices = src_vol.shape[2]
    for s in range(int(n_slices * 0.2), int(n_slices * 0.8)):
        yield slice_to_tensor(src_vol[:, :, s], img_size), slice_to_tensor(tgt_vol[:, :, s], img_size)


def wrap_unet_for_ablation(unet):
    """Safety check only: cond now comes from get_input_resshift (same path as training),
    so the UNet input must already match. Fail loudly instead of silently trimming channels."""
    orig_forward = unet.forward
    in_ch = unet.in_channels
    class_cond = getattr(unet, "num_classes", None) is not None
    state = {"logged": False}

    def forward(x, *args, **kwargs):
        if x.shape[1] != in_ch:
            raise RuntimeError(f"UNet got {x.shape[1]} input channels, expects {in_ch} -- cond/config mismatch")
        y = kwargs.get("y", args[2] if len(args) >= 3 else None)
        if (y is not None) != class_cond:
            raise RuntimeError(f"y passed={y is not None} but UNet class-conditional={class_cond}")
        if not state["logged"]:
            print(f"[check] first UNet call OK: input {x.shape[1]} ch, y passed={y is not None}")
            state["logged"] = True
        return orig_forward(x, *args, **kwargs)

    unet.forward = forward
    return class_cond, in_ch


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ema", action="store_true", help="force raw (non-EMA) weights")
    ap.add_argument("--spade_only", action="store_true", help="skip diffusion: decode y0 (SPADE latent) directly")
    ap.add_argument("--out_csv", default="diffusion_resshift_eval_results.csv")
    ap.add_argument("--hist_match", choices=["none", "3t", "both"], default="none",
                    help="3t: match test 3T (model input) to training 3T; both: also match test 7T (ground truth) to training 7T")
    ap.add_argument("--hist_ref", default="hist_ref_train.npz", help="from: python hist_match.py build")
    ap.add_argument("--spade_ckpt", default=None, help="override SPADE weights AFTER loading the diffusion ckpt")
    args = ap.parse_args()
    HM = None
    if args.hist_match != 'none':
        _r = np.load(args.hist_ref)
        HM = {'lm_3t': _r['lm_3t'], 'lm_7t': _r['lm_7t'], 'thr': float(_r['thr']), 'mode': args.hist_match}
        print(f"[hist_match] mode={args.hist_match}, reference={args.hist_ref}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
        lpips_fn = lpips.LPIPS(net="vgg", spatial=True).to(device).eval()
        for p_ in lpips_fn.parameters():
            p_.requires_grad_(False)
    except ImportError:
        print("[warn] lpips not installed -- LPIPS will be NaN. pip install lpips")
        lpips_fn = None

    ssim_kernel = ae_mod.auto_ssim_kernel(img_size, num_patches)

    model = instantiate_from_config(config.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if args.spade_ckpt:
        sd = torch.load(os.path.expanduser(args.spade_ckpt), map_location="cpu", weights_only=False)
        for key in ("spade", "model", "state_dict"):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        sd = {k[len("first_stage_model.spade."):] if k.startswith("first_stage_model.spade.") else k: v for k, v in sd.items()}
        spade = model.first_stage_model.spade
        before = {k: v.clone() for k, v in spade.state_dict().items()}
        spade.load_state_dict(sd, strict=True)  # strict: every key must match, else error
        n_diff = sum(not torch.equal(before[k], v.to(before[k].device)) for k, v in spade.state_dict().items())
        print(f"[spade_ckpt] loaded {args.spade_ckpt}: {len(sd)} tensors, {n_diff}/{len(before)} differ from diffusion ckpt's SPADE")
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', 'n/a')})")

    ablation = bool(getattr(model, "ablation_no_condition", False))
    class_cond, unet_in_ch = wrap_unet_for_ablation(model.model.diffusion_model)
    print(f"ablation_no_condition={ablation}, UNet class-conditional={class_cond}, UNet in_channels={unet_in_ch}")

    use_ema = getattr(model, "use_ema", False) and hasattr(model, "ema_scope") and not args.no_ema
    ema_ctx = model.ema_scope if use_ema else contextlib.nullcontext
    print(f"weights used for sampling: {'EMA' if use_ema else 'raw (non-EMA)'}")

    latent_channels = config.model.params.channels
    latent_size = config.model.params.image_size
    n_timesteps = model.num_timesteps
    print(f"sampling: full {n_timesteps}-step ResShift reverse process, kappa={getattr(model, 'kappa', None)}, seed={args.seed}")

    def eval_batch(x_src, x_tgt):
        b = x_src.shape[0]
        y = torch.full((b,), args.target_class, dtype=torch.long)
        batch = {"source": x_src.to(device), "target": x_tgt.to(device), "target_class": y}

        with torch.no_grad(), ema_ctx():
            # 跟訓練完全同一條路徑：y0 = SPADE latent；cond 依 ablation_no_condition / condition_select 裁切
            _, y0, cond = model.get_input_resshift(batch, model.first_stage_key)
            shape = (b, latent_channels, latent_size, latent_size)
            z_pred = y0 if args.spade_only else model.p_sample_loop_resshift(y0, cond, shape, verbose=False)
            gen = model.decode_first_stage(z_pred).clamp(-1, 1)

        gen_01 = (gen.float() + 1) / 2
        tgt_01 = (x_tgt.to(device).float().clamp(-1, 1) + 1) / 2
        brain_mask = ae_mod.build_brain_mask(tgt_01, threshold=args.brain_mask_threshold)
        absdiff = (gen_01 - tgt_01).abs()

        ssim_val = ssim_fn(gen_01, tgt_01, data_range=1.0,
                           kernel_size=ssim_kernel[0], sigma=ssim_kernel[1]).item()
        psnr_val = psnr_fn(gen_01, tgt_01, data_range=1.0,
                           dim=(1, 2, 3), reduction="elementwise_mean").item()
        mae_val = absdiff.mean().item()
        mae_brain_t = ae_mod.masked_avg(absdiff, brain_mask)
        mae_brain = mae_brain_t.item() if torch.isfinite(mae_brain_t) else float("nan")

        lpips_val = float("nan")
        if lpips_fn is not None:
            with torch.no_grad():
                lpips_map = lpips_fn((gen_01 * 2 - 1).repeat(1, 3, 1, 1), (tgt_01 * 2 - 1).repeat(1, 3, 1, 1))
            lpips_t = ae_mod.masked_avg(lpips_map, brain_mask)
            if torch.isfinite(lpips_t):
                lpips_val = lpips_t.item()

        return {"ssim": ssim_val, "psnr": psnr_val, "mae": mae_val,
                "mae_brain": mae_brain, "lpips": lpips_val, "n": b}

    keys = ("ssim", "psnr", "mae", "mae_brain", "lpips")
    per_subject_rows = []
    grand = {k: 0.0 for k in keys} | {f"{k}_n": 0 for k in keys}

    for subject in args.subjects:
        sums = {k: 0.0 for k in keys} | {f"{k}_n": 0 for k in keys}
        src_buf, tgt_buf = [], []

        def flush():
            if not src_buf:
                return
            r = eval_batch(torch.cat(src_buf), torch.cat(tgt_buf))
            for k in keys:
                if r[k] == r[k]:  # skip NaN
                    sums[k] += r[k] * r["n"]
                    sums[f"{k}_n"] += r["n"]

        for src_t, tgt_t in iter_test_slices(args.test_dir, subject, img_size, hm=HM):
            src_buf.append(src_t); tgt_buf.append(tgt_t)
            if len(src_buf) == args.batch_size:
                flush()
                src_buf, tgt_buf = [], []
        flush()

        row = {"subject": subject, "n_slices": sums["ssim_n"]}
        for k in keys:
            row[k] = sums[k] / sums[f"{k}_n"] if sums[f"{k}_n"] else float("nan")
            grand[k] += sums[k]
            grand[f"{k}_n"] += sums[f"{k}_n"]
        per_subject_rows.append(row)
        print(f"[{subject}] n={row['n_slices']}  SSIM={row['ssim']:.4f}  PSNR={row['psnr']:.2f}  "
              f"MAE={row['mae']:.4f}  MAE(brain)={row['mae_brain']:.4f}  LPIPS(brain)={row['lpips']:.4f}")

    overall = {k: (grand[k] / grand[f"{k}_n"]) if grand[f"{k}_n"] else float("nan") for k in keys}
    print("\n=== overall (all subjects, slice-weighted) ===")
    print(f"SSIM={overall['ssim']:.4f}  PSNR={overall['psnr']:.2f}  MAE={overall['mae']:.4f}  "
          f"MAE(brain)={overall['mae_brain']:.4f}  LPIPS(brain)={overall['lpips']:.4f}  "
          f"(n={grand['ssim_n']} slices, epoch {ckpt.get('epoch', '?')})")

    fields = ["subject", "n_slices", *keys]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_subject_rows)
        w.writerow({"subject": "overall", "n_slices": grand["ssim_n"], **overall})
    print(f"\nsaved to {args.out_csv}")


if __name__ == "__main__":
    main()
