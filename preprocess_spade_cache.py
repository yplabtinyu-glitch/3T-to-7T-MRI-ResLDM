"""
One-time preprocessing: pre-extract, resize, and normalize the exact slices
SpadePairDataset would produce on-the-fly, saved as flat memory-mapped .npy
arrays. This removes the per-epoch disk I/O bottleneck: slice-level shuffling
across ~778 subjects was defeating the small per-worker LRU volume cache,
causing a near-cold nib.load() of a full 3D volume on almost every sample
(measured: ~2.2s/step at batch=16, GPU util 0% -- i.e. purely I/O-bound).

Produces, under --cache_dir:
    train_src.npy, train_tgt.npy   shape (N_train, img_size, img_size) float32
    val_src.npy,   val_tgt.npy     shape (N_val,   img_size, img_size) float32
    meta.json

Values are fully preprocessed already: per-volume min-max normalized to
[0,1], resized to (img_size,img_size), rescaled to [-1,1] -- identical to
what SpadePairDataset._to_tensor produced, so training numerics don't change.

Run once from ~/Desktop/ALDM/LDM :
    python preprocess_spade_cache.py
"""
import argparse
import json
import os
import random
import time

import nibabel as nib
import numpy as np
from numpy.lib.format import open_memmap
import torch
import torch.nn.functional as F

from pretrain_spade import build_pairs  # reuse the exact same pairing logic


def load_and_normalize_volume(path):
    vol = nib.load(path).get_fdata(dtype=np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin > 1e-6:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol)
    return vol


def resize_slice(sl, img_size):
    t = torch.from_numpy(np.ascontiguousarray(sl)).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    if t.shape[-2:] != (img_size, img_size):
        t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()  # (img_size, img_size), still in [0,1]


def build_split_cache(pairs, img_size, out_src_path, out_tgt_path):
    ranges = []
    total = 0
    for src_path, _tgt_path in pairs:
        n_slices = nib.load(src_path).shape[2]
        lo, hi = int(n_slices * 0.2), int(n_slices * 0.8)
        ranges.append((lo, hi))
        total += (hi - lo)

    src_mm = open_memmap(out_src_path, mode="w+", dtype=np.float32, shape=(total, img_size, img_size))
    tgt_mm = open_memmap(out_tgt_path, mode="w+", dtype=np.float32, shape=(total, img_size, img_size))

    idx = 0
    t0 = time.time()
    for pi, (src_path, tgt_path) in enumerate(pairs):
        lo, hi = ranges[pi]
        src_vol = load_and_normalize_volume(src_path)
        tgt_vol = load_and_normalize_volume(tgt_path)
        for s in range(lo, hi):
            src_mm[idx] = resize_slice(src_vol[:, :, s], img_size) * 2.0 - 1.0
            tgt_mm[idx] = resize_slice(tgt_vol[:, :, s], img_size) * 2.0 - 1.0
            idx += 1
        if (pi + 1) % 50 == 0 or (pi + 1) == len(pairs):
            elapsed = time.time() - t0
            rate = (pi + 1) / elapsed
            eta = (len(pairs) - pi - 1) / rate if rate > 0 else float("inf")
            print(f"[{os.path.basename(out_src_path)}] {pi+1}/{len(pairs)} subjects "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    src_mm.flush()
    tgt_mm.flush()
    assert idx == total
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default=os.path.expanduser("~/Desktop/data/pairs_ventricle.csv"))
    ap.add_argument("--data_dir", default=os.path.expanduser("~/Desktop/data"))
    ap.add_argument("--cache_dir", default=os.path.expanduser("~/Desktop/data/spade_cache"))
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    pairs = build_pairs(args.csv_path, args.data_dir)
    print(f"usable subject pairs: {len(pairs)}")

    pairs_shuffled = pairs[:]
    random.Random(args.seed).shuffle(pairs_shuffled)
    n_val = max(1, int(len(pairs_shuffled) * args.val_frac))
    val_pairs = pairs_shuffled[:n_val]
    train_pairs = pairs_shuffled[n_val:]
    print(f"train subjects: {len(train_pairs)}  val subjects: {len(val_pairs)}")

    n_train = build_split_cache(
        train_pairs, args.img_size,
        os.path.join(args.cache_dir, "train_src.npy"),
        os.path.join(args.cache_dir, "train_tgt.npy"),
    )
    n_val_slices = build_split_cache(
        val_pairs, args.img_size,
        os.path.join(args.cache_dir, "val_src.npy"),
        os.path.join(args.cache_dir, "val_tgt.npy"),
    )

    meta = {
        "n_train": n_train, "n_val": n_val_slices,
        "img_size": args.img_size, "val_frac": args.val_frac, "seed": args.seed,
        "n_train_subjects": len(train_pairs), "n_val_subjects": len(val_pairs),
    }
    with open(os.path.join(args.cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done:", meta)


if __name__ == "__main__":
    main()
