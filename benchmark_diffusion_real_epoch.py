"""
Real (data-backed) timing benchmark for the diffusion (LatentDiffusion) stage
-- replaces benchmark_epoch_time.py's synthetic-tensor approach now that real
3T/7T pair data exists (reuses pretrain_spade.py's own CachedSpadePairDataset/
SpadePairDataset, same spade_cache this project already built for SPADE
training -- not reimplemented, imported directly).

Also resolves `scale_factor` for real, instead of leaving mri-3t7t-ldm.yaml's
default of 1.0.

IMPORTANT -- why this does NOT use the ALDM/LDM repo's own built-in
`scale_by_std` auto-rescale (ddpm.py's `on_train_batch_start`, confirmed by
reading ddpm.py directly): that hook calls
`super().get_input(batch, self.first_stage_key)`, which is plain DDPM's
`get_input(batch, k) => batch[k]`, with k = self.first_stage_key = "image"
(per mri-3t7t-ldm.yaml). Our batches only ever have "source"/"target"/
"target_class" keys -- this LatentDiffusion subclass's OWN get_input()
override (ddpm.py line ~541) reads THOSE, never "image" -- so
scale_by_std=True would crash on the very first training batch with
KeyError('image'). So scale_factor is computed HERE, manually, from a real
batch's z_tgt encoding specifically (NOT pooled with z_src/z_srctgt like the
earlier compute_latent_std.py draft did): z_tgt is the one quantity that
actually goes through the forward noising process in shared_step() (see
get_input(): `out = [z_tgt, cond]`) -- z_src/z_srctgt are channel-concatenated
conditioning, never noised -- so z_tgt's distribution is what the noise
schedule's unit-variance assumption is actually about, matching what the
repo's own (inapplicable-here) auto mechanism would have computed had our
first_stage_key been a single "image" like the original BraTS setup.

Run from ~/Desktop/ALDM/LDM :
    python benchmark_diffusion_real_epoch.py --cache_dir ~/Desktop/data/spade_cache
"""
import argparse
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from ldm.util import instantiate_from_config
from pretrain_spade import CachedSpadePairDataset, SpadePairDataset, build_pairs


class DiffusionPairDataset(Dataset):
    """Wraps a SPADE-style pair dataset (already [-1,1]-normalized 128x128
    PIXEL-space slices -- NOT latents, encode_first_stage() happens inside
    LatentDiffusion.get_input()) into the {"source","target","target_class"}
    dict format ddpm.py's LatentDiffusion.get_input() reads (confirmed by
    reading ddpm.py directly: x_src=batch["source"], x_tgt=batch["target"],
    y=batch["target_class"])."""
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
    ap.add_argument("--cache_dir", default=None,
                     help="spade_cache dir (train_src.npy/train_tgt.npy) from "
                          "preprocess_spade_cache.py -- fast path, same cache pretrain_spade.py "
                          "used. If omitted, falls back to on-the-fly pairs_ventricle.csv loading "
                          "via --csv_path/--data_dir (slow, same fallback pretrain_spade.py has).")
    ap.add_argument("--csv_path", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--num_warmup_batches", type=int, default=5)
    ap.add_argument("--num_timed_batches", type=int, default=30)
    ap.add_argument("--target_class", type=int, default=1)  # 1 = 7T, matches mri-3t7t-ldm.yaml/pretrain_spade.py
    ap.add_argument("--target_total_epochs", type=int, default=None,
                     help="if given, also print estimated total GPU-hours for this many epochs")
    cli = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- real dataset, reusing pretrain_spade.py's own classes ----
    if cli.cache_dir:
        inner = CachedSpadePairDataset(cli.cache_dir, "train", target_class=cli.target_class)
    else:
        assert cli.csv_path and cli.data_dir, "need --cache_dir, or both --csv_path and --data_dir"
        pairs = build_pairs(cli.csv_path, cli.data_dir)
        inner = SpadePairDataset(pairs, target_class=cli.target_class)
    ds = DiffusionPairDataset(inner)
    loader = DataLoader(ds, batch_size=cli.batch_size, shuffle=True,
                         num_workers=cli.num_workers, drop_last=True)
    n_steps_per_epoch = len(ds) // cli.batch_size
    print(f"real training slices: {len(ds)}  ->  {n_steps_per_epoch} steps/epoch @ batch_size={cli.batch_size}")

    # ---- build the model at scale_factor=1.0 first (need one real encoded
    # batch before we know the real scale_factor) ----
    config = OmegaConf.load(cli.config)
    config.model.params.scale_factor = 1.0
    config.model.params.scale_by_std = False
    model = instantiate_from_config(config.model)
    model.learning_rate = config.model.base_learning_rate
    model = model.to(device)
    model.train()

    # ---- real scale_factor from a real batch's z_tgt (see module docstring) ----
    it = iter(loader)
    probe_batch = next(it)
    with torch.no_grad():
        x_tgt = probe_batch["target"].to(device)
        z_tgt = model.encode_first_stage(x_tgt)  # scale_factor still 1.0 here -> raw AE latent
    real_std = z_tgt.flatten().std().item()
    scale_factor = 1.0 / real_std
    print(f"\nreal AE latent (z_tgt) std, one batch (n={x_tgt.shape[0]}): {real_std:.4f}")
    print(f"-> scale_factor = 1/std = {scale_factor:.4f}")
    print("(single-batch estimate, enough to unblock timing/training -- average over more probe "
          "batches for a tighter number before committing this to mri-3t7t-ldm.yaml)")
    model.scale_factor = torch.tensor(scale_factor, device=device)

    opt = model.configure_optimizers()
    if isinstance(opt, (list, tuple)):
        opt = opt[0][0] if isinstance(opt[0], (list, tuple)) else opt[0]

    def one_step():
        batch = next(it)
        opt.zero_grad()
        loss, _ = model.shared_step(batch)
        loss.backward()
        opt.step()

    print(f"\nwarming up ({cli.num_warmup_batches} real batches)...")
    for _ in range(cli.num_warmup_batches):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    print(f"timing {cli.num_timed_batches} real batches (real data + real I/O + real scale_factor)...")
    t0 = time.time()
    for _ in range(cli.num_timed_batches):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    sec_per_step = elapsed / cli.num_timed_batches
    epoch_min = n_steps_per_epoch * sec_per_step / 60

    print(f"\n{sec_per_step*1000:.1f} ms/step  (batch_size={cli.batch_size}, real data)")
    print(f"-> estimated real epoch time: {epoch_min:.1f} min "
          f"({n_steps_per_epoch} steps/epoch, {len(ds)} slices/epoch)")

    if cli.target_total_epochs:
        total_hours = epoch_min * cli.target_total_epochs / 60
        print(f"-> estimated total for {cli.target_total_epochs} epochs: {total_hours:.1f} GPU-hours")

    print("\nCAVEAT: same I/O-ramp caveat as the AE/SPADE benchmarks -- this times "
          "num_timed_batches real batches after a short warmup, not a full epoch. If --cache_dir's "
          "mmap npy files are used (fast path, recommended), I/O should already be small and "
          "steady-state; the on-the-fly fallback will show a SPADE-epoch-0-style slow start "
          "instead. For the fully trustworthy number, let one complete real epoch run.")


if __name__ == "__main__":
    main()
