"""
visualize_slice_compare.py -- 抽一張 slice，並排比較：
  3T 輸入 | 只經過 SPADE (decode(y0)) | 各 checkpoint 的生成結果 | 真實 7T
第二列是 |pred - 7T| 誤差圖（同一色階），每格標 SSIM / PSNR / MAE(brain)。

前處理、cond、y0、取樣都跟 evaluate_diffusion_resshift_v2.py 同一條路徑
（get_input_resshift + p_sample_loop_resshift，raw weights，固定 seed）。
config 自動讀 checkpoint 裡存的 args['config']，不用手填。

用法（放在 ~/Desktop/ALDM/LDM，跟 evaluate_diffusion_resshift_v2.py 同資料夾）：
  python visualize_slice_compare.py --subject 212730 --autoencoder_script_dir "$AE_DIR"
  可選：--slice 120（預設取中間）、--rot 1（顯示時轉 90° 的次數）、--seed 0
  checkpoint 預設四組，不存在的會自動跳過；要改用 --ckpt "標籤=路徑" 重複指定。
輸出：slice_compare_<subject>_s<slice>.png 和同名 .csv（每格的數值）
"""
import argparse
import csv
import gc
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

DEFAULT_CKPTS = [
    "No condition=checkpoints_diffusion_ablation_nocondition/diffusion_epoch99.pth",
    "3T latent=checkpoints_diffusion_ablation_3t/diffusion_epoch99.pth",
    "3T SPADE=checkpoints_diffusion_ablation_spade/diffusion_epoch99.pth",
    "3T latent + 3T SPADE=checkpoints_diffusion_resshift_perchkappa/diffusion_epoch99.pth",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--slice", type=int, default=None, help="axial index (last axis); default = middle")
    ap.add_argument("--test_dir", default=os.path.expanduser("~/Desktop/test"))
    ap.add_argument("--ckpt", action="append", default=None, help='"label=path", repeatable')
    ap.add_argument("--autoencoder_script_dir", default=".")
    ap.add_argument("--target_class", type=int, default=1)
    ap.add_argument("--brain_mask_threshold", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hist_match", choices=["none", "3t", "both"], default="both")
    ap.add_argument("--hist_ref", default="hist_ref_train.npz")
    ap.add_argument("--rot", type=int, default=0, help="np.rot90 k for display only")
    ap.add_argument("--hide_y0", action="store_true", help="do not show the decoded-y0 (SPADE) column")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae_mod = load_autoencoder_utils(args.autoencoder_script_dir)

    ckpts = []
    for item in (args.ckpt or DEFAULT_CKPTS):
        label, path = item.split("=", 1)
        if os.path.exists(path):
            ckpts.append((label, path))
        else:
            print(f"[skip] {label}: {path} not found")
    if not ckpts:
        raise SystemExit("no checkpoints found")

    src_vol = load_and_normalize_volume(os.path.join(args.test_dir, "reg_3T", f"{args.subject}.nii.gz"))
    tgt_vol = load_and_normalize_volume(os.path.join(args.test_dir, "7T", f"{args.subject}.nii.gz"))
    if args.hist_match != "none":
        import numpy as _np
        from hist_match import match_volume
        _r = _np.load(args.hist_ref)
        src_vol, _ = match_volume(src_vol, _r["lm_3t"], float(_r["thr"]))
        if args.hist_match == "both":
            tgt_vol, _ = match_volume(tgt_vol, _r["lm_7t"], float(_r["thr"]))
        print(f"[hist_match] {args.hist_match}")
    s = args.slice if args.slice is not None else src_vol.shape[2] // 2
    lo, hi = int(src_vol.shape[2] * 0.2), int(src_vol.shape[2] * 0.8)
    if not lo <= s < hi:
        print(f"[warn] slice {s} is outside the evaluated range [{lo}, {hi})")

    panels = []          # (title, img01, metrics or None)
    spade_ref = None     # y0 from the first model, to check all models share the same SPADE
    x_src = x_tgt = None
    tgt_01 = mask = None

    def metrics(img01):
        ssim = ssim_fn(img01, tgt_01, data_range=1.0, kernel_size=kernel[0], sigma=kernel[1]).item()
        psnr = psnr_fn(img01, tgt_01, data_range=1.0, dim=(1, 2, 3), reduction="elementwise_mean").item()
        mae_b = ae_mod.masked_avg((img01 - tgt_01).abs(), mask).item()
        return {"ssim": ssim, "psnr": psnr, "mae_brain": mae_b}

    for i, (label, path) in enumerate(ckpts):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg_path = ck["args"]["config"]
        config = OmegaConf.load(cfg_path)

        if i == 0:
            ae_ckpt = os.path.expanduser(config.model.params.first_stage_config.params.ae_checkpoint_path)
            ae_args = torch.load(ae_ckpt, map_location="cpu", weights_only=False).get("args", {})
            img_size = ae_args.get("img_size", 128)
            kernel = ae_mod.auto_ssim_kernel(img_size, ae_args.get("num_patches", 24))
            x_src = slice_to_tensor(src_vol[:, :, s], img_size).to(device)
            x_tgt = slice_to_tensor(tgt_vol[:, :, s], img_size).to(device)
            tgt_01 = (x_tgt.clamp(-1, 1) + 1) / 2
            mask = ae_mod.build_brain_mask(tgt_01, threshold=args.brain_mask_threshold)

        model = instantiate_from_config(config.model).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        print(f"[{label}] epoch {ck.get('epoch')} val_loss={ck.get('val_loss')} config={cfg_path}")

        batch = {"source": x_src, "target": x_tgt,
                 "target_class": torch.full((1,), args.target_class, dtype=torch.long)}
        with torch.no_grad():
            _, y0, cond = model.get_input_resshift(batch, model.first_stage_key)
            if spade_ref is None:
                spade_ref = y0.clone()
                spade_img = (model.decode_first_stage(y0).clamp(-1, 1) + 1) / 2
                if not args.hide_y0:
                        if not args.hide_y0:
                            panels.append(("Only SPADE", spade_img, metrics(spade_img)))
            else:
                d = (y0 - spade_ref).abs().max().item()
                if d > 1e-5:
                    print(f"  [warn] {label}: SPADE latent differs from '{ckpts[0][0]}' (max |diff|={d:.4g}) "
                          f"-- this model uses a different SPADE checkpoint")
            torch.manual_seed(args.seed)
            shape = (1, config.model.params.channels, config.model.params.image_size, config.model.params.image_size)
            z = model.p_sample_loop_resshift(y0, cond, shape, verbose=False)
            gen = (model.decode_first_stage(z).clamp(-1, 1) + 1) / 2
        panels.append((label, gen, metrics(gen)))

        del model, ck
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    src_01 = (x_src.clamp(-1, 1) + 1) / 2
    cols = [("3T input", src_01, None)] + panels + [("7T (ground truth)", tgt_01, None)]

    def to_np(t):
        return np.rot90(t[0, 0].float().cpu().numpy(), k=args.rot)

    n = len(cols)
    fig, ax = plt.subplots(2, n, figsize=(2.6 * n, 6.2), dpi=150)
    err_max = 0.3
    for j, (title, img, m) in enumerate(cols):
        ax[0, j].imshow(to_np(img), cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(title, fontsize=9)
        if m is not None:
            ax[0, j].set_xlabel(f"SSIM {m['ssim']:.3f}\nPSNR {m['psnr']:.2f}  MAE(b) {m['mae_brain']:.3f}", fontsize=7.5)
            im = ax[1, j].imshow(to_np((img - tgt_01).abs()), cmap="magma", vmin=0, vmax=err_max)
        else:
            ax[1, j].set_visible(False)
        for r in (0, 1):
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
    fig.colorbar(im, ax=ax[1, :].tolist(), fraction=0.015, pad=0.01, label="|pred − 7T|")
    fig.suptitle(f"subject {args.subject}, axial slice {s}  (seed {args.seed}, raw weights)", fontsize=10)

    out = f"slice_compare_{args.subject}_s{s}"
    fig.savefig(out + ".png", bbox_inches="tight")
    with open(out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "ssim", "psnr", "mae_brain"])
        for title, _, m in panels:
            w.writerow([title, m["ssim"], m["psnr"], m["mae_brain"]])
    print(f"saved {out}.png and {out}.csv")
    for title, _, m in panels:
        print(f"  {title:24s} SSIM={m['ssim']:.4f}  PSNR={m['psnr']:.2f}  MAE(brain)={m['mae_brain']:.4f}")


if __name__ == "__main__":
    main()
