"""
measure_residual_stats.py -- 對整個訓練集(不是smoke test的一個batch)量測
ResShift的shift residual: z_srctgt(SPADE轉換後的3T latent, ResShift的y0) 減去
z_tgt(真正的7T latent, ResShift的x0)，逐個latent channel分別算 mean/var/std，
外加所有channel合併(pooled)的整體統計量。

跟目前 config(mri-3t7t-resshift.yaml)裡單一全域 kappa=0.5 的假設對照：
如果4個channel的std差很多，代表應該把kappa拆成per-channel的向量，而不是
維持單一scalar乘在所有channel上(目前resshift.py是 kappa**2 * eta_t * I，
I是純量乘單位矩陣，沒有per-channel差異)。

複用跟 train_diffusion.py 完全一樣的資料管線(CachedSpadePairDataset +
DiffusionPairDataset)，不重新發明normalize/資料載入邏輯，確保跟訓練當下
看到的東西是同一份資料、同一種前處理。
"""
import argparse

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from ldm.util import instantiate_from_config
from pretrain_spade import CachedSpadePairDataset


class DiffusionPairDataset(Dataset):
    """跟 train_diffusion.py 裡一模一樣的wrapper。"""
    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        src, tgt, y = self.inner[idx]
        return {"source": src, "target": tgt, "target_class": y}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/latent-diffusion/mri-3t7t-resshift.yaml")
    ap.add_argument("--cache_dir", required=True,
                     help="跟 train_diffusion.py 一樣的 spade_cache 目錄")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--target_class", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model).to(device)
    model.eval()

    inner = CachedSpadePairDataset(args.cache_dir, args.split, target_class=args.target_class)
    ds = DiffusionPairDataset(inner)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, drop_last=False)
    print(f"{args.split} slices: {len(ds)} -> {len(loader)} batches @ batch_size={args.batch_size}")

    n_channels = config.model.params.channels  # 4

    # 用 running sum / sumsq 累計統計量(Welford簡化版，用float64避免精度問題)，
    # 不把12萬張的residual全部留在記憶體裡。
    count = 0
    ch_sum = torch.zeros(n_channels, dtype=torch.float64)
    ch_sumsq = torch.zeros(n_channels, dtype=torch.float64)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x_start, cond = model.get_input(batch, model.first_stage_key)
            z_src = cond["c_concat"][:, :n_channels, ...]
            z_srctgt = cond["c_concat"][:, n_channels:2 * n_channels, ...]
            residual = (z_srctgt - x_start).double()  # [B, C, H, W] -- 這就是ResShift的 y0 - x0

            b, c, h, w = residual.shape
            per_ch = residual.permute(1, 0, 2, 3).reshape(c, -1)  # [C, B*H*W]
            ch_sum += per_ch.sum(dim=1).cpu()
            ch_sumsq += (per_ch ** 2).sum(dim=1).cpu()
            count += per_ch.shape[1]

            if (i + 1) % 200 == 0 or (i + 1) == len(loader):
                print(f"  ...processed {i + 1}/{len(loader)} batches")

    ch_mean = ch_sum / count
    ch_var = ch_sumsq / count - ch_mean ** 2
    ch_std = ch_var.sqrt()

    print(f"\n=== residual (z_srctgt - z_tgt) over {count} pixel-samples/channel, split={args.split} ===")
    for c in range(n_channels):
        print(f"channel {c}: mean={ch_mean[c].item():+.4f}  var={ch_var[c].item():.4f}  std={ch_std[c].item():.4f}")

    overall_mean = ch_sum.sum() / (count * n_channels)
    overall_var = ch_sumsq.sum() / (count * n_channels) - overall_mean ** 2
    print(f"\noverall (4 channels pooled): mean={overall_mean.item():+.4f}  "
          f"var={overall_var.item():.4f}  std={overall_var.sqrt().item():.4f}")
    print(f"\n(目前 config 裡的 kappa=0.5 是對照這個pooled std用的 --"
          f" 如果各channel的std差距明顯，代表應該改成per-channel的kappa向量)")


if __name__ == "__main__":
    main()
