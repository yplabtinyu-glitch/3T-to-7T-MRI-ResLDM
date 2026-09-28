"""
check_loss_shapes.py
獨立診斷腳本 -- 不會碰到正在跑的訓練 process (PID 2808605)，可以並行跑。
用跟 train_diffusion.py 完全一樣的 config/model/data，抓「一個」真實 batch，
把 ddpm.py 裡 p_losses() (line 636) 每一步的 tensor shape 印出來，直接驗證
x_start/model_output 到底是 4D (B,C,H,W，我們的 2D AE+UNet 應該產生的形狀)
還是 5D (.mean([1,2,3,4])，line 651，若不是5D理論上該報錯)。
"""
import argparse
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from ldm.util import instantiate_from_config
from pretrain_spade import CachedSpadePairDataset


class DiffusionPairDataset(Dataset):
    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        src, tgt, y = self.inner[idx]
        return {"source": src, "target": tgt, "target_class": y}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/latent-diffusion/mri-3t7t-ldm.yaml")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model).to(device)
    model.eval()

    inner = CachedSpadePairDataset(args.cache_dir, "train", target_class=1)
    ds = DiffusionPairDataset(inner)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))

    x, cond = model.get_input(batch, model.first_stage_key)
    print(f"x_start (z_tgt) shape: {tuple(x.shape)}   ndim={x.ndim}")
    if isinstance(cond, dict):
        for k, v in cond.items():
            if torch.is_tensor(v):
                shp = tuple(v.shape)
            elif isinstance(v, list):
                shp = [tuple(vv.shape) if torch.is_tensor(vv) else type(vv) for vv in v]
            else:
                shp = type(v)
            print(f"cond[{k!r}] -> {shp}")

    t = torch.randint(0, model.num_timesteps, (x.shape[0],), device=device).long()
    noise = torch.randn_like(x)
    x_noisy = model.q_sample(x_start=x, t=t, noise=noise)
    print(f"x_noisy shape: {tuple(x_noisy.shape)}")

    model_output = model.apply_model(x_noisy, t, cond)
    print(f"model_output (UNet raw output) shape: {tuple(model_output.shape)}   ndim={model_output.ndim}")

    target = noise if model.parameterization == "eps" else x
    print(f"target shape: {tuple(target.shape)}")

    elementwise = model.get_loss(model_output, target, mean=False)
    print(f"elementwise loss shape: {tuple(elementwise.shape)}   ndim={elementwise.ndim}")

    try:
        reduced = elementwise.mean([1, 2, 3, 4])
        print(f"elementwise.mean([1,2,3,4]) SUCCEEDED -> shape {tuple(reduced.shape)}")
        print(">>> tensors are 5D，跟 p_losses() 預期的一樣，沒有bug -- 已經實測確認。")
    except (IndexError, RuntimeError) as e:
        print(f"elementwise.mean([1,2,3,4]) FAILED: {e}")
        print(">>> 這證實了維度不合的問題 -- 但這樣的話真正訓練時 p_losses() 應該也會用同樣方式"
              "失敗，跟你目前訓練沒crash矛盾。如果你看到這行，跟我說 -- "
              "代表這支script跟真正訓練路徑之間還有沒抓到的差異，我要先找出來才能信任loss數字。")

    try:
        loss, loss_dict = model.p_losses(x, cond, t, noise=noise)
        print(f"\nmodel.p_losses() 直接呼叫 -- 成功。loss={loss.item():.5f}")
        for k, v in loss_dict.items():
            print(f"  {k} = {v.item() if torch.is_tensor(v) else v}")
    except Exception as e:
        print(f"\nmodel.p_losses() 直接呼叫 -- 失敗: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
