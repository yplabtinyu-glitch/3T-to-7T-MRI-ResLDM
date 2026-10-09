"""
evaluate_spade_ceiling_valset.py -- 用SPADE訓練時實際使用的cache val split
(而不是4個held-out test subjects)，把多個checkpoint(預設epoch10~90+latest)
的SPADE-only decode品質(SSIM/PSNR/MAE/LPIPS)一次全部跑完。

跟 evaluate_spade_ceiling.py 的 eval_batch (encode->spade->decode->
SSIM/PSNR/MAE/LPIPS，同一套SSIM kernel、同一套brain mask、同一顆LPIPS
network) 邏輯完全一致，唯一差異是資料來源：

  evaluate_spade_ceiling.py       : ~/Desktop/test/{3T,7T}/{subject}.nii.gz
                                     (4個獨立held-out subject，完整3D volume)
  evaluate_spade_ceiling_valset.py: cache_dir/{split}_src.npy / {split}_tgt.npy
                                     (pretrain_spade.py訓練時實際用來算val_loss
                                     的同一份cache val split，已經是切好、
                                     normalize到[-1,1]的2D slice)

這樣decode品質(SSIM等)才是跟log.csv裡val_loss(latent-space L1)同一份資料
算出來的，可以直接對照「哪個epoch的latent val_loss最低」跟「哪個epoch的
decode影像品質最好」是不是同一個答案。

AE只在一開始load一次，之後每個checkpoint只替換SPADE的state_dict，不用
每個checkpoint都重新load一次AE。

Run from ~/Desktop/ALDM/LDM :
    python evaluate_spade_ceiling_valset.py \
        --config configs/latent-diffusion/mri-3t7t-resshift.yaml \
        --cache_dir ~/Desktop/data/spade_cache \
        --checkpoints_dir ~/Desktop/LSC/checkpoints_spade
"""
import argparse
import csv
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_fn

from ldm.models.frozen_ae_spade import FrozenAEWithSPADE
from pretrain_spade import CachedSpadePairDataset


def load_autoencoder_utils(script_dir):
    script_dir = os.path.expanduser(script_dir)
    sys.path.insert(0, script_dir)
    import train_autoencoder as ae_mod
    return ae_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/latent-diffusion/mri-3t7t-resshift.yaml",
                     help="只用來讀 first_stage_config.params.ae_checkpoint_path；"
                          "config裡的spade_checkpoint_path會被忽略，SPADE權重改由"
                          "--checkpoints_dir/--checkpoint_epochs逐一指定")
    ap.add_argument("--cache_dir", default=os.path.expanduser("~/Desktop/data/spade_cache"))
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--checkpoints_dir", default=os.path.expanduser("~/Desktop/LSC/checkpoints_spade"))
    ap.add_argument("--checkpoint_epochs", nargs="+",
                     default=["10", "20", "30", "40", "50", "60", "70", "80", "90", "latest"],
                     help="數字N對應spade_epoch{N}.pth，'latest'對應spade_latest.pth")
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--target_class", type=int, default=1,
                     help="跟訓練pretrain_spade.py一致，1=7T方向")
    ap.add_argument("--autoencoder_script_dir", default=".")
    ap.add_argument("--num_patches", type=int, default=None)
    ap.add_argument("--brain_mask_threshold", type=float, default=0.02)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out_csv", default="spade_ceiling_valset_sweep.csv")
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

    def ckpt_path_for(epoch_tag):
        name = "spade_latest.pth" if epoch_tag == "latest" else f"spade_epoch{epoch_tag}.pth"
        return os.path.join(args.checkpoints_dir, name)

    valid_epochs = [e for e in args.checkpoint_epochs if os.path.exists(ckpt_path_for(e))]
    missing = [e for e in args.checkpoint_epochs if e not in valid_epochs]
    if missing:
        print(f"[warn] 找不到這些checkpoint，會跳過: {missing}")
    if not valid_epochs:
        raise SystemExit("沒有任何可用的checkpoint，檢查 --checkpoints_dir")

    # AE只load一次；之後每個checkpoint只換SPADE的state_dict
    ae_wrap = FrozenAEWithSPADE(ae_checkpoint_path=fsc_params.ae_checkpoint_path,
                                 num_classes=fsc_params.get("num_classes", 2),
                                 spade_checkpoint_path=ckpt_path_for(valid_epochs[0])).to(device)
    ae_wrap.eval()

    ds = CachedSpadePairDataset(args.cache_dir, args.split, target_class=args.target_class)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"{args.split} slices (cache, 跟pretrain_spade.py訓練時算val_loss用的同一份split): {len(ds)}")

    def eval_batch(x_src, x_tgt):
        b = x_src.shape[0]
        y = torch.full((b,), args.target_class, dtype=torch.long, device=device)
        with torch.no_grad():
            z_src, _, _ = ae_wrap.encode(x_src.to(device))
            z_srctgt = ae_wrap.spade(z_src, y)
            gen = ae_wrap.decode(z_srctgt).clamp(-1, 1)

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

    summary_rows = []
    for epoch_tag in valid_epochs:
        ckpt_path = ckpt_path_for(epoch_tag)
        ae_wrap.spade.load_state_dict(torch.load(ckpt_path, map_location=device))
        ae_wrap.spade.eval()

        sums = {"ssim": 0.0, "psnr": 0.0, "mae": 0.0, "lpips": 0.0, "lpips_n": 0, "n": 0}
        for src, tgt, _y in loader:
            r = eval_batch(src, tgt)
            n = r["n"]
            sums["ssim"] += r["ssim"] * n
            sums["psnr"] += r["psnr"] * n
            sums["mae"] += r["mae"] * n
            if r["lpips"] == r["lpips"]:
                sums["lpips"] += r["lpips"] * n
                sums["lpips_n"] += n
            sums["n"] += n

        n = sums["n"]
        row = {
            "checkpoint": epoch_tag, "n_slices": n,
            "ssim": sums["ssim"] / n if n else float("nan"),
            "psnr": sums["psnr"] / n if n else float("nan"),
            "mae": sums["mae"] / n if n else float("nan"),
            "lpips": (sums["lpips"] / sums["lpips_n"]) if sums["lpips_n"] else float("nan"),
        }
        summary_rows.append(row)
        print(f"[spade_epoch{epoch_tag}] n={n}  SSIM={row['ssim']:.4f}  PSNR={row['psnr']:.2f}  "
              f"MAE={row['mae']:.4f}  LPIPS(brain)={row['lpips']:.4f}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint", "n_slices", "ssim", "psnr", "mae", "lpips"])
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    print(f"\nsaved checkpoint-sweep-on-{args.split}-set results to {args.out_csv}")


if __name__ == "__main__":
    main()
