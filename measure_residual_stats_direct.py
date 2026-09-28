"""
measure_residual_stats_direct.py -- 量測「拿掉SPADE、直接3T->7T」這個假設性
ResShift變體的shift residual: z_tgt(真實7T latent) 減去 z_src(原始3T latent，
沒有經過SPADE轉換)，逐個latent channel分別算mean/var/std，外加pooled整體統計量。

跟 measure_residual_stats.py 的差異只有一處：那支腳本算的是
z_srctgt(SPADE轉換後) - z_tgt，這支腳本算的是 z_tgt - z_src(完全跳過SPADE)。
兩者是同一個訓練集、同一支AE，唯一差別就是anchor有沒有先經過SPADE粗略對齊過
domain。把這兩組std放在一起比，就能看出SPADE到底幫diffusion把要修正的殘差
縮小了多少量級。

不需要呼叫.spade()，也不需要ddpm.py的get_input() -- 直接用
model.encode_first_stage()/model.get_first_stage_encoding()對source跟target
各自encode，跟get_input()內部用的是同一組函式、同一個scale_factor，確保量出來
的殘差就是"拿掉SPADE版本"的ResShift實際會看到的殘差，不是憑空猜的。

仍然複用跟train_diffusion.py一樣的CachedSpadePairDataset資料管線(不重新發明
normalize/資料載入邏輯)，只是不再理會batch裡的target_class(這支變體用不到)。
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
    ap.add_argument("--config", default="configs/latent-diffusion/mri-3t7t-resshift.yaml",
                     help="只用來讀ae_checkpoint_path跟scale_factor -- spade_checkpoint_path在這支腳本裡不會被用到")
    ap.add_argument("--cache_dir", required=True,
                     help="跟 train_diffusion.py 一樣的 spade_cache 目錄")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--target_class", type=int, default=1,
                     help="只影響CachedSpadePairDataset怎麼配對，不影響這支腳本算的殘差本身")
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

    count = 0
    ch_sum = torch.zeros(n_channels, dtype=torch.float64)
    ch_sumsq = torch.zeros(n_channels, dtype=torch.float64)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x_src = batch["source"].to(device)
            x_tgt = batch["target"].to(device)

            z_src = model.get_first_stage_encoding(model.encode_first_stage(x_src))
            z_tgt = model.get_first_stage_encoding(model.encode_first_stage(x_tgt))
            residual = (z_tgt - z_src).double()  # 完全跳過SPADE的殘差: z_tgt - z_src

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

    print(f"\n=== residual (z_tgt - z_src, NO SPADE) over {count} pixel-samples/channel, split={args.split} ===")
    for c in range(n_channels):
        print(f"channel {c}: mean={ch_mean[c].item():+.4f}  var={ch_var[c].item():.4f}  std={ch_std[c].item():.4f}")

    overall_mean = ch_sum.sum() / (count * n_channels)
    overall_var = ch_sumsq.sum() / (count * n_channels) - overall_mean ** 2
    print(f"\noverall (4 channels pooled): mean={overall_mean.item():+.4f}  "
          f"var={overall_var.item():.4f}  std={overall_var.sqrt().item():.4f}")
    print(f"\n(對照組 -- SPADE轉換後的殘差 z_srctgt-z_tgt，2026-09-26量到的:"
          f" channel std=[0.7646, 0.3341, 0.4287, 0.7411], pooled std=0.5978。"
          f" 上面這組沒有SPADE的std如果明顯更大，代表SPADE確實有先把域落差縮小一部分，"
          f" 拿掉它之後diffusion要單獨扛起完整的3T->7T落差。)")


if __name__ == "__main__":
    main()
