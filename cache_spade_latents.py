import argparse
import json
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
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--target_class", type=int, default=1)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.load(args.config)
    sp_path = os.path.realpath(os.path.expanduser(str(config.model.params.first_stage_config.params.spade_checkpoint_path)))
    model = instantiate_from_config(config.model).to(device).eval()
    meta_path = os.path.join(args.latent_dir, "meta.json")
    meta = json.load(open(meta_path))
    print(f"SPADE to cache: {sp_path}")

    for split in ("train", "val"):
        zs = np.load(os.path.join(args.latent_dir, f"{split}_zsrc.npy"), mmap_mode="r")
        out = np.lib.format.open_memmap(os.path.join(args.latent_dir, f"{split}_zspade.npy"), mode="w+",
                                        dtype=zs.dtype, shape=zs.shape)
        t0 = time.time()
        with torch.no_grad():
            for b in range(0, len(zs), args.batch_size):
                z = torch.from_numpy(np.array(zs[b:b + args.batch_size])).float().to(device)
                y = torch.full((len(z),), args.target_class, dtype=torch.long)
                out[b:b + len(z)] = model.first_stage_model.spade(z, y).float().cpu().numpy().astype(zs.dtype)
        out.flush(); del out
        print(f"{split}: {len(zs)} SPADE latents, {time.time() - t0:.0f} s")
    meta["spade"] = {"checkpoint": sp_path, "bytes": os.path.getsize(sp_path), "mtime": os.path.getmtime(sp_path),
                     "target_class": args.target_class}
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"meta.json updated: {meta['spade']}")

    ds = CachedSpadePairDataset(args.cache_dir, "train", target_class=args.target_class)
    zs = np.load(os.path.join(args.latent_dir, "train_zsrc.npy"), mmap_mode="r")
    zt = np.load(os.path.join(args.latent_dir, "train_ztgt.npy"), mmap_mode="r")
    zp = np.load(os.path.join(args.latent_dir, "train_zspade.npy"), mmap_mode="r")
    idx = list(range(64))
    y = torch.full((64,), args.target_class, dtype=torch.long)
    img_b = {"source": torch.stack([_img(ds[i][0]) for i in idx]), "target": torch.stack([_img(ds[i][1]) for i in idx]), "target_class": y}
    lat_b = {"source": torch.from_numpy(np.array(zs[idx])).float(), "target": torch.from_numpy(np.array(zt[idx])).float(),
             "zspade": torch.from_numpy(np.array(zp[idx])).float(), "target_class": y}

    orig_enc, spade = model.encode_first_stage, model.first_stage_model.spade
    stash = {}

    def use_cache(on):
        if on:
            model.encode_first_stage = lambda x: x if x.shape[1] == zs.shape[1] else orig_enc(x)
            spade.forward = lambda z, yy: stash["z"]
        else:
            model.encode_first_stage = orig_enc
            if "forward" in spade.__dict__:
                del spade.forward

    with torch.no_grad():
        x0_a, c_a = model.get_input(img_b, model.first_stage_key)
        stash["z"] = lat_b["zspade"].to(device)
        use_cache(True)
        x0_b, c_b = model.get_input(lat_b, model.first_stage_key)
        use_cache(False)
        x0_c, c_c = model.get_input(img_b, model.first_stage_key)
    d_cache = max((x0_a - x0_b).abs().max().item(), (c_a["c_concat"] - c_b["c_concat"]).abs().max().item())
    d_live = max((x0_a - x0_c).abs().max().item(), (c_a["c_concat"] - c_c["c_concat"]).abs().max().item())
    print(f"[check] live vs cache: {d_cache:.2e}   live vs live (same batch): {d_live:.2e}   (latent max {x0_a.abs().max().item():.2f})")
    print("[check] OK" if d_cache < 2e-2 else "[check] FAILED -- do not use the SPADE cache")

    if args.bench:
        model.train()

        def bench(batch, n=30):
            for i in range(n + 5):
                if i == 5:
                    torch.cuda.synchronize() if device == "cuda" else None
                    t0 = time.time()
                if "zspade" in batch:
                    stash["z"] = batch["zspade"].to(device)
                loss, _ = model.shared_step(batch)
                loss.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize() if device == "cuda" else None
            return (time.time() - t0) / n * 1000

        t_img = bench(img_b)
        model.encode_first_stage = lambda x: x if x.shape[1] == zs.shape[1] else orig_enc(x)
        enc_only = {k: v for k, v in lat_b.items() if k != "zspade"}
        t_enc = bench(enc_only)
        spade.forward = lambda z, yy: stash["z"]
        t_both = bench(lat_b)
        use_cache(False)
        print(f"[bench] batch 64 shared_step+backward:  images {t_img:.1f} ms | encoder cached {t_enc:.1f} ms | "
              f"encoder+SPADE cached {t_both:.1f} ms  ({(1 - t_both / t_img) * 100:.0f}% faster than images)")


if __name__ == "__main__":
    main()
