import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/latent-diffusion/mri-3t7t-resshift-ablation-nocondition.yaml")
ap.add_argument("--cache_dir", default=os.path.expanduser("~/Desktop/data/spade_cache"))
ap.add_argument("--max_slices", type=int, default=20000)
ap.add_argument("--batch_size", type=int, default=128)
ap.add_argument("--target_class", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
config = OmegaConf.load(args.config)
print("SPADE:", config.model.params.first_stage_config.params.get("spade_checkpoint_path"))
model = instantiate_from_config(config.model).to(device).eval()

src = np.load(os.path.join(args.cache_dir, "train_src.npy"), mmap_mode="r")
tgt = np.load(os.path.join(args.cache_dir, "train_tgt.npy"), mmap_mode="r")
idx = np.arange(len(src))
if args.max_slices and args.max_slices < len(idx):
    idx = np.sort(np.random.default_rng(args.seed).choice(idx, args.max_slices, replace=False))
print(f"using {len(idx)} / {len(src)} training slices")

S = {k: torch.zeros(4, dtype=torch.float64) for k in ("r3", "rs")}
SS = {k: torch.zeros(4, dtype=torch.float64) for k in ("r3", "rs")}
N = 0
with torch.no_grad():
    for b in range(0, len(idx), args.batch_size):
        ii = idx[b:b + args.batch_size]
        x_src = torch.from_numpy(np.ascontiguousarray(src[ii])).float().unsqueeze(1).to(device)
        x_tgt = torch.from_numpy(np.ascontiguousarray(tgt[ii])).float().unsqueeze(1).to(device)
        batch = {"source": x_src, "target": x_tgt,
                 "target_class": torch.full((len(ii),), args.target_class, dtype=torch.long)}
        x0, cond = model.get_input(batch, model.first_stage_key)
        c = cond["c_concat"]
        for k, r in (("r3", c[:, 0:4] - x0), ("rs", c[:, 4:8] - x0)):
            r = r.double().transpose(0, 1).reshape(4, -1).cpu()
            S[k] += r.sum(1); SS[k] += (r ** 2).sum(1)
        N += x0[:, 0].numel()
        if (b // args.batch_size) % 20 == 0:
            print(f"  {b + len(ii)}/{len(idx)}")

std = {k: ((SS[k] / N) - (S[k] / N) ** 2).sqrt().numpy() for k in S}
mean = {k: (S[k] / N).numpy() for k in S}
print(f"\n{'ch':>3} {'std(3T-7T)':>11} {'std(SPADE-7T)':>14} {'mean(3T-7T)':>12} {'mean(SPADE-7T)':>15}")
for c in range(4):
    print(f"{c:>3} {std['r3'][c]:>11.3f} {std['rs'][c]:>14.3f} {mean['r3'][c]:>12.3f} {mean['rs'][c]:>15.3f}")
print(f"\nkappa for y0 = SPADE ->  {','.join(f'{v:.3f}' for v in std['rs'])}")
print(f"kappa for y0 = 3T    ->  {','.join(f'{v:.3f}' for v in std['r3'])}")
