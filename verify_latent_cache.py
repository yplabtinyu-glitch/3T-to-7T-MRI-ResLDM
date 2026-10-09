import argparse
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from pretrain_spade import CachedSpadePairDataset


def _img(t):
    t = torch.as_tensor(t).float()
    return t.unsqueeze(0) if t.dim() == 2 else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache_dir", default=os.path.expanduser("~/Desktop/data/spade_cache"))
    ap.add_argument("--latent_dir", default=os.path.expanduser("~/Desktop/data/latent_cache"))
    ap.add_argument("--target_class", type=int, default=1)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = instantiate_from_config(OmegaConf.load(args.config).model).to(device).eval()
    ds = CachedSpadePairDataset(args.cache_dir, "train", target_class=args.target_class)
    zs = np.load(os.path.join(args.latent_dir, "train_zsrc.npy"), mmap_mode="r")
    zt = np.load(os.path.join(args.latent_dir, "train_ztgt.npy"), mmap_mode="r")
    print(f"cache {zs.shape} {zs.dtype};  TF32 conv={torch.backends.cudnn.allow_tf32}, matmul={torch.backends.cuda.matmul.allow_tf32}")

    x256 = torch.stack([_img(ds[i][0]) for i in range(256)]).to(device)
    lat = torch.from_numpy(np.array(zs[:64])).float().to(device)

    def jitter(tag):
        with torch.no_grad():
            A = model.encode_first_stage(x256[:64])
            B = model.encode_first_stage(x256)[:64]
        dAB = (A - B).abs().max().item()
        dAC = (A - lat).abs().max().item()
        print(f"[{tag}] max|z|={A.abs().max().item():.2f}   live b64 vs live b256: {dAB:.2e}   live b64 vs cache: {dAC:.2e}")
        return dAB, dAC

    dAB, dAC = jitter("TF32 as configured")
    prev = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    jitter("TF32 off")
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev

    ok = dAC <= 2 * dAB + 1e-5
    print(f"\n=> cache error {dAC:.2e} vs live-encode jitter {dAB:.2e}: "
          + ("OK -- same order as the jitter live encoding already has; cache is usable"
             if ok else "LARGER than live jitter -- something else differs, do not use the cache"))

    if args.bench:
        model.train()
        orig = model.encode_first_stage
        lat_ch = zs.shape[1]
        y = torch.full((64,), args.target_class, dtype=torch.long)
        img_b = {"source": x256[:64].cpu(), "target": torch.stack([_img(ds[i][1]) for i in range(64)]), "target_class": y}
        lat_b = {"source": torch.from_numpy(np.array(zs[:64])).float(), "target": torch.from_numpy(np.array(zt[:64])).float(), "target_class": y}

        def bench(batch, n=30):
            for i in range(n + 5):
                if i == 5:
                    torch.cuda.synchronize() if device == "cuda" else None
                    t0 = time.time()
                loss, _ = model.shared_step(batch)
                loss.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize() if device == "cuda" else None
            return (time.time() - t0) / n * 1000

        t_img = bench(img_b)
        model.encode_first_stage = lambda x: x if x.shape[1] == lat_ch else orig(x)
        t_lat = bench(lat_b)
        model.encode_first_stage = orig
        print(f"[bench] batch 64 shared_step+backward: images {t_img:.1f} ms  vs  cache {t_lat:.1f} ms  "
              f"({(1 - t_lat / t_img) * 100:.0f}% faster, data loading excluded)")


if __name__ == "__main__":
    main()
