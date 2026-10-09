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
    ap.add_argument("--out_dir", default=os.path.expanduser("~/Desktop/data/latent_cache"))
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--target_class", type=int, default=1)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model).to(device).eval()
    dt = np.float32 if args.dtype == "float32" else np.float16

    meta = {"ae_checkpoint": str(config.model.params.first_stage_config.params.get("ae_checkpoint_path")),
            "dtype": args.dtype, "source_cache": args.cache_dir, "splits": {}}
    for split in ("train", "val"):
        ds = CachedSpadePairDataset(args.cache_dir, split, target_class=args.target_class)
        n = len(ds)
        with torch.no_grad():
            probe = model.encode_first_stage(_img(ds[0][0]).unsqueeze(0).to(device))
        shape = tuple(probe.shape[1:])
        zs = np.lib.format.open_memmap(os.path.join(args.out_dir, f"{split}_zsrc.npy"), mode="w+", dtype=dt, shape=(n, *shape))
        zt = np.lib.format.open_memmap(os.path.join(args.out_dir, f"{split}_ztgt.npy"), mode="w+", dtype=dt, shape=(n, *shape))
        t0 = time.time()
        for b in range(0, n, args.batch_size):
            items = [ds[i] for i in range(b, min(b + args.batch_size, n))]
            xs = torch.stack([_img(it[0]) for it in items]).to(device)
            xt = torch.stack([_img(it[1]) for it in items]).to(device)
            with torch.no_grad():
                zs[b:b + len(items)] = model.encode_first_stage(xs).float().cpu().numpy().astype(dt)
                zt[b:b + len(items)] = model.encode_first_stage(xt).float().cpu().numpy().astype(dt)
            if (b // args.batch_size) % 50 == 0:
                print(f"  {split}: {b + len(items)}/{n}  ({time.time() - t0:.0f} s)", flush=True)
        zs.flush(); zt.flush(); del zs, zt
        src_files = {f: os.path.getsize(os.path.join(args.cache_dir, f)) for f in (f"{split}_src.npy", f"{split}_tgt.npy")}
        meta["splits"][split] = {"n": n, "latent_shape": list(shape), "source_file_bytes": src_files}
        print(f"{split}: {n} slices, latent {shape}, {time.time() - t0:.0f} s")
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved to {args.out_dir}")

    ds = CachedSpadePairDataset(args.cache_dir, "train", target_class=args.target_class)
    zs = np.load(os.path.join(args.out_dir, "train_zsrc.npy"), mmap_mode="r")
    zt = np.load(os.path.join(args.out_dir, "train_ztgt.npy"), mmap_mode="r")
    idx = list(range(64))
    items = [ds[i] for i in idx]
    y = torch.full((len(idx),), args.target_class, dtype=torch.long)
    img_batch = {"source": torch.stack([_img(it[0]) for it in items]), "target": torch.stack([_img(it[1]) for it in items]), "target_class": y}
    lat_batch = {"source": torch.from_numpy(np.asarray(zs[idx])).float(), "target": torch.from_numpy(np.asarray(zt[idx])).float(), "target_class": y}
    orig_enc = model.encode_first_stage
    lat_ch = lat_batch["source"].shape[1]
    with torch.no_grad():
        x0_a, c_a = model.get_input(img_batch, model.first_stage_key)
        model.encode_first_stage = lambda x: x if x.shape[1] == lat_ch else orig_enc(x)
        x0_b, c_b = model.get_input(lat_batch, model.first_stage_key)
        model.encode_first_stage = orig_enc
    d_x0 = (x0_a - x0_b).abs().max().item()
    d_c = (c_a["c_concat"] - c_b["c_concat"]).abs().max().item()
    scale = x0_a.abs().max().item()
    tol = 1e-4 if args.dtype == "float32" else 5e-3 * max(1.0, scale)
    print(f"[check] max|x0 diff| = {d_x0:.2e}, max|c_concat diff| = {d_c:.2e}  (latent max |value| {scale:.2f}, tol {tol:.1e})")
    if max(d_x0, d_c) > tol:
        raise SystemExit("[check] FAILED -- cached latents do not reproduce get_input; do not use this cache")
    print("[check] OK -- cached latents reproduce get_input")

    if args.bench:
        model.train()
        def bench(batch, n=20):
            for i in range(n + 3):
                if i == 3:
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                loss, _ = model.shared_step(batch)
                loss.backward()
                model.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()
            return (time.time() - t0) / n * 1000
        bidx = list(range(64))
        items = [ds[i] for i in bidx]
        y = torch.full((64,), args.target_class, dtype=torch.long)
        img_b = {"source": torch.stack([_img(it[0]) for it in items]), "target": torch.stack([_img(it[1]) for it in items]), "target_class": y}
        lat_b = {"source": torch.from_numpy(np.asarray(zs[bidx])).float(), "target": torch.from_numpy(np.asarray(zt[bidx])).float(), "target_class": y}
        t_img = bench(img_b)
        model.encode_first_stage = lambda x: x if x.shape[1] == lat_ch else orig_enc(x)
        t_lat = bench(lat_b)
        model.encode_first_stage = orig_enc
        print(f"[bench] batch 64, shared_step+backward: images {t_img:.1f} ms/step  vs  cached latents {t_lat:.1f} ms/step "
              f"({(1 - t_lat / t_img) * 100:.0f}% faster; data loading not included)")


if __name__ == "__main__":
    main()
