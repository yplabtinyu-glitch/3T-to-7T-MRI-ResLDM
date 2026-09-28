"""
Stage-2-equivalent pretraining for SPADE, mirroring the original ALDM repo's
VQGAN stage2 recipe (verified by reading VQ-GAN/taming/models/vqgan.py +
taming/modules/losses/vqperceptual.py):

    z_src    = encode(source_3T_image)          # frozen AE, no grad
    z_srctgt = spade(z_src, target_class)        # only trainable part
    z_tgt    = encode(target_7T_image)          # frozen AE, no grad
    loss     = L1(z_srctgt, z_tgt)               # original config zeroes out
                                                  # the GAN/codebook terms, so
                                                  # this is really all it is.

Pairing (verified against the real data on 2026-09-23):
    3T_warped_to_7T/<7t_id>__from__<3t_id>.nii.gz  <->  raw_7T/<7t_id>.nii.gz
    filtered to rows with ventricle_dice >= 0.75 in pairs_ventricle.csv.

Slice selection matches MRISliceDataset in train_autoencoder.py: middle 60%
of slices along axis 2 (lo=0.2*N, hi=0.8*N), since both volumes are already
registered into the same 7T space so slice index s means the same anatomical
location in both.

Run from ~/Desktop/ALDM/LDM :
    python pretrain_spade.py
"""
import argparse
import csv
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ldm.models.frozen_ae_spade import FrozenAEWithSPADE


def build_pairs(csv_path, data_dir, dice_threshold=0.75):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    pairs = []
    for row in rows:
        if float(row["ventricle_dice"]) < dice_threshold:
            continue
        warped_name = f"{row['7t_id']}__from__{row['3t_id']}.nii.gz"
        warped_path = os.path.join(data_dir, "3T_warped_to_7T", warped_name)
        raw7t_path = os.path.join(data_dir, "raw_7T", f"{row['7t_id']}.nii.gz")
        if os.path.exists(warped_path) and os.path.exists(raw7t_path):
            pairs.append((warped_path, raw7t_path))
    return pairs


class SpadePairDataset(Dataset):
    def __init__(self, pairs, img_size=128, target_class=1, cache_size=8):
        self.pairs = pairs
        self.img_size = img_size
        self.target_class = target_class

        self.index = []  # (pair_idx, slice_idx)
        for pi, (src_path, _tgt_path) in enumerate(pairs):
            img = nib.load(src_path)
            n_slices = img.shape[2]
            lo = int(n_slices * 0.2)
            hi = int(n_slices * 0.8)
            for s in range(lo, hi):
                self.index.append((pi, s))

        self._cache = OrderedDict()
        self._cache_size = cache_size

    def __len__(self):
        return len(self.index)

    def _load_volume(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        vol = nib.load(path).get_fdata(dtype=np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin > 1e-6:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol = np.zeros_like(vol)
        self._cache[path] = vol
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vol

    def _to_tensor(self, sl):
        sl = torch.from_numpy(np.ascontiguousarray(sl)).float().unsqueeze(0)  # (1,H,W)
        if sl.shape[-2:] != (self.img_size, self.img_size):
            sl = F.interpolate(sl.unsqueeze(0), size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False).squeeze(0)
        sl = sl * 2.0 - 1.0  # [0,1] -> [-1,1], matches MRISliceDataset
        return sl

    def __getitem__(self, idx):
        pi, s = self.index[idx]
        src_path, tgt_path = self.pairs[pi]
        src_vol = self._load_volume(src_path)
        tgt_vol = self._load_volume(tgt_path)
        src_sl = self._to_tensor(src_vol[:, :, s])
        tgt_sl = self._to_tensor(tgt_vol[:, :, s])
        y = torch.tensor(self.target_class, dtype=torch.long)
        return src_sl, tgt_sl, y


class CachedSpadePairDataset(Dataset):
    """
    Reads slices pre-extracted by preprocess_spade_cache.py (memory-mapped
    .npy files, already normalized/resized/rescaled to [-1,1]).

    This replaces SpadePairDataset's on-the-fly nib.load() for training:
    measured at ~2.2s/step (batch=16), GPU util 0%, because slice-level
    shuffling across ~778 subjects defeated the small per-worker LRU volume
    cache -- almost every sample required a fresh full-volume disk read.
    Memory-mapped indexing removes that bottleneck entirely.
    """
    def __init__(self, cache_dir, split, target_class=1):
        self.src = np.load(os.path.join(cache_dir, f"{split}_src.npy"), mmap_mode="r")
        self.tgt = np.load(os.path.join(cache_dir, f"{split}_tgt.npy"), mmap_mode="r")
        assert self.src.shape == self.tgt.shape, (self.src.shape, self.tgt.shape)
        self.target_class = target_class

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, idx):
        src = torch.from_numpy(np.ascontiguousarray(self.src[idx])).float().unsqueeze(0)
        tgt = torch.from_numpy(np.ascontiguousarray(self.tgt[idx])).float().unsqueeze(0)
        y = torch.tensor(self.target_class, dtype=torch.long)
        return src, tgt, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default=os.path.expanduser("~/Desktop/data/pairs_ventricle.csv"))
    ap.add_argument("--data_dir", default=os.path.expanduser("~/Desktop/data"))
    ap.add_argument("--cache_dir", default=os.path.expanduser("~/Desktop/data/spade_cache"))
    ap.add_argument("--ae_checkpoint", default=os.path.expanduser("~/Desktop/LSC/checkpoints_ae/ae_epoch690.pth"))
    ap.add_argument("--out_dir", default=os.path.expanduser("~/Desktop/LSC/checkpoints_spade"))
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--target_class", type=int, default=1)  # 1 = 7T, matches mri-3t7t-ldm.yaml
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)  # matches original VQGAN stage2 base_learning_rate
    ap.add_argument("--epochs", type=int, default=10, help="total epoch count to train TO (not additional epochs)")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None,
                     help="path to a spade_epochN.pth / spade_latest.pth to continue from. "
                          "NOTE: these checkpoints are a bare model.spade.state_dict() (no "
                          "epoch/optimizer saved alongside), so the optimizer's Adam moment "
                          "estimates restart fresh -- only the learned weights carry over.")
    ap.add_argument("--start_epoch", type=int, default=0,
                     help="epoch number to resume AT (0-indexed, matches log.csv's epoch column). "
                          "Not recoverable from --resume's checkpoint file, must be passed explicitly.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    meta_path = os.path.join(args.cache_dir, "meta.json")
    if os.path.exists(meta_path):
        print(f"found precomputed slice cache at {args.cache_dir} -- using it (fast path)")
        train_ds = CachedSpadePairDataset(args.cache_dir, "train", target_class=args.target_class)
        val_ds = CachedSpadePairDataset(args.cache_dir, "val", target_class=args.target_class)
    else:
        print(f"no cache found at {args.cache_dir} -- run preprocess_spade_cache.py first "
              f"for much faster training. falling back to on-the-fly loading (slow).")
        pairs = build_pairs(args.csv_path, args.data_dir)
        print(f"usable subject pairs (dice>=0.75, files verified): {len(pairs)}")

        # split at the PAIR level (not slice level) so slices from the same
        # subject never appear in both train and val
        pairs_shuffled = pairs[:]
        random.Random(args.seed).shuffle(pairs_shuffled)
        n_val = max(1, int(len(pairs_shuffled) * args.val_frac))
        val_pairs = pairs_shuffled[:n_val]
        train_pairs = pairs_shuffled[n_val:]
        print(f"train subjects: {len(train_pairs)}  val subjects: {len(val_pairs)}")

        train_ds = SpadePairDataset(train_pairs, img_size=args.img_size, target_class=args.target_class)
        val_ds = SpadePairDataset(val_pairs, img_size=args.img_size, target_class=args.target_class)
    print(f"train slices: {len(train_ds)}  val slices: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrozenAEWithSPADE(args.ae_checkpoint, num_classes=args.num_classes).to(device)
    model.ae.eval()

    if args.resume:
        resume_path = os.path.expanduser(args.resume)
        model.spade.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"resumed SPADE weights from {resume_path}, continuing at epoch {args.start_epoch} "
              f"(optimizer state is NOT restored -- Adam moments restart fresh)")

    opt = torch.optim.AdamW(model.spade.parameters(), lr=args.lr, betas=(0.5, 0.9))

    log_path = os.path.join(args.out_dir, "log.csv")
    log_is_new = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow(["epoch", "step", "global_step", "train_loss", "sec_per_step", "val_loss"])

    def spade_l1_loss(src, tgt, y):
        with torch.no_grad():
            z_src, _, _ = model.encode(src)
            z_tgt, _, _ = model.encode(tgt)
        z_srctgt = model.spade(z_src, y)
        return F.l1_loss(z_srctgt, z_tgt)

    global_step = args.start_epoch * len(train_loader)
    train_start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.spade.train()
        running = 0.0
        running_n = 0
        window_start = time.time()
        for i, (src, tgt, y) in enumerate(train_loader):
            src, tgt, y = src.to(device), tgt.to(device), y.to(device)
            opt.zero_grad()
            loss = spade_l1_loss(src, tgt, y)
            loss.backward()
            opt.step()
            running += loss.item()
            running_n += 1
            global_step += 1

            is_log_step = (i + 1) % args.log_every == 0
            is_last_step = (i + 1) == len(train_loader)
            if is_log_step or is_last_step:
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - window_start
                sec_per_step = elapsed / running_n
                avg_loss = running / running_n
                print(f"[epoch {epoch}] step {i+1}/{len(train_loader)}  "
                      f"train_loss={avg_loss:.5f}  {sec_per_step*1000:.1f} ms/step")
                log_writer.writerow([epoch, i + 1, global_step, avg_loss, sec_per_step, ""])
                log_file.flush()
                running, running_n = 0.0, 0
                window_start = time.time()

        model.spade.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt, y in val_loader:
                src, tgt, y = src.to(device), tgt.to(device), y.to(device)
                val_loss += spade_l1_loss(src, tgt, y).item()
        val_loss /= max(1, len(val_loader))
        print(f"=== epoch {epoch} done | val_loss={val_loss:.5f} | "
              f"elapsed so far: {(time.time()-train_start)/60:.1f} min ===")
        log_writer.writerow([epoch, "epoch_end", global_step, "", "", val_loss])
        log_file.flush()

        ckpt_path = os.path.join(args.out_dir, f"spade_epoch{epoch}.pth")
        torch.save(model.spade.state_dict(), ckpt_path)
        print(f"saved {ckpt_path}")

    torch.save(model.spade.state_dict(), os.path.join(args.out_dir, "spade_latest.pth"))
    log_file.close()
    print("done.")


if __name__ == "__main__":
    main()
