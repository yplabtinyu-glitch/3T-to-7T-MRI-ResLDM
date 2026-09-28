import os, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from ldm.models.diffusion.resshift import LatentResShift
from pretrain_spade import CachedSpadePairDataset
from train_diffusion import DiffusionPairDataset

config = OmegaConf.load("configs/latent-diffusion/mri-3t7t-resshift.yaml")
model = LatentResShift(**config.model.params).to("cuda")
model.eval()

cache_dir = os.path.expanduser("~/Desktop/data/spade_cache")  # 跟你nohup指令裡用的一樣
train_inner = CachedSpadePairDataset(cache_dir, "train", target_class=1)
loader = DataLoader(DiffusionPairDataset(train_inner), batch_size=64, shuffle=True)
batch = next(iter(loader))
batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}

x_start, y0, cond = model.get_input_resshift(batch, model.first_stage_key)
resid_std = (y0 - x_start).std().item()
print(f"real residual std = {resid_std:.4f}   (目前kappa={model.kappa})")
