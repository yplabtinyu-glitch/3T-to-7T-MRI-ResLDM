"""
train_diffusion.py
Real training entry point for the LatentDiffusion (UNet) stage, using the
same real 3T/7T pair data (spade_cache) and Dataset wiring already validated
in benchmark_diffusion_real_epoch.py (DiffusionPairDataset wrapping
pretrain_spade.py's CachedSpadePairDataset). Follows pretrain_spade.py's own
conventions (log.csv schema, checkpoint naming, nohup-friendly stdout) since
that's the pattern already proven on this project -- deliberately NOT using
ALDM's main.py/PyTorch-Lightning Trainer, which would need its own data
config section this project never wired up.

REQUIRES mri-3t7t-ldm.yaml's first_stage_config.params to have BOTH
ae_checkpoint_path AND spade_checkpoint_path set -- without the latter,
SPADEGenerator is randomly initialized and gets silently frozen that way
for the entire run (confirmed by reading FrozenAEWithSPADE.__init__ and
ddpm.py's instantiate_first_stage()). This script asserts on that at
startup instead of failing silently.

Run from ~/Desktop/ALDM/LDM (backgrounded, same pattern as pretrain_spade.py):
    nohup python train_diffusion.py --cache_dir ~/Desktop/data/spade_cache \
        --batch_size 64 --epochs 100 > train_diffusion.log 2>&1 &
"""
import argparse
import csv
import os
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from ldm.util import instantiate_from_config
from pretrain_spade import CachedSpadePairDataset, SpadePairDataset, build_pairs


class DiffusionPairDataset(Dataset):
    """Same wrapper as benchmark_diffusion_real_epoch.py -- {"source","target",
    "target_class"} dict format ddpm.py's LatentDiffusion.get_input() reads."""
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
                     help="spade_cache dir (train_src.npy/train_tgt.npy/val_src.npy/val_tgt.npy)")
    ap.add_argument("--csv_path", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default="checkpoints_diffusion")
    ap.add_argument("--batch_size", type=int, default=64,
                     help="64 was the fastest of the 4/8/16/32/64 sweep on 2026-09-24 "
                          "(182.6 ms/step, no OOM on the RTX A5500's 24GB) -- see "
                          "diffusion_batch_sweep.log")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="defaults to config's base_learning_rate")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--target_class", type=int, default=1)  # 1 = 7T
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=1, help="epochs between validation passes")
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- real data (same source as benchmark_diffusion_real_epoch.py) ----
    if args.cache_dir:
        train_inner = CachedSpadePairDataset(args.cache_dir, "train", target_class=args.target_class)
        val_inner = CachedSpadePairDataset(args.cache_dir, "val", target_class=args.target_class)
    else:
        assert args.csv_path and args.data_dir, "need --cache_dir, or both --csv_path and --data_dir"
        pairs = build_pairs(args.csv_path, args.data_dir)
        import random
        pairs_shuffled = pairs[:]
        random.Random(args.seed).shuffle(pairs_shuffled)
        n_val = max(1, int(len(pairs_shuffled) * 0.1))
        val_inner = SpadePairDataset(pairs_shuffled[:n_val], target_class=args.target_class)
        train_inner = SpadePairDataset(pairs_shuffled[n_val:], target_class=args.target_class)

    train_ds = DiffusionPairDataset(train_inner)
    val_ds = DiffusionPairDataset(val_inner)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)
    print(f"train slices: {len(train_ds)}  val slices: {len(val_ds)}  "
          f"-> {len(train_loader)} steps/epoch @ batch_size={args.batch_size}")

    # ---- model ----
    config = OmegaConf.load(args.config)
    fsc_params = config.model.params.first_stage_config.params
    assert "spade_checkpoint_path" in fsc_params and fsc_params.spade_checkpoint_path, (
        "mri-3t7t-ldm.yaml's first_stage_config.params has no spade_checkpoint_path -- "
        "SPADE would be randomly initialized and silently frozen that way for the whole "
        "run. Fix the yaml (see the comment added there on 2026-09-24) before training.")
    model = instantiate_from_config(config.model)
    model.learning_rate = args.lr if args.lr is not None else config.model.base_learning_rate
    model = model.to(device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {args.resume}, starting at epoch {start_epoch}")

    opt = model.configure_optimizers()
    if isinstance(opt, (list, tuple)):
        opt = opt[0][0] if isinstance(opt[0], (list, tuple)) else opt[0]
    if args.resume and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    log_path = os.path.join(args.out_dir, "log.csv")
    log_is_new = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow(["epoch", "step", "global_step", "train_loss", "sec_per_step", "val_loss"])

    global_step = start_epoch * len(train_loader)
    train_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, running_n = 0.0, 0
        window_start = time.time()
        for i, batch in enumerate(train_loader):
            opt.zero_grad()
            loss, _ = model.shared_step(batch)
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

        val_loss = None
        if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
            model.eval()
            v_total, v_n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    loss, _ = model.shared_step(batch)
                    v_total += loss.item()
                    v_n += 1
            val_loss = v_total / max(1, v_n)
            print(f"=== epoch {epoch} done | val_loss={val_loss:.5f} | "
                  f"elapsed so far: {(time.time()-train_start)/60:.1f} min ===")
            log_writer.writerow([epoch, "epoch_end", global_step, "", "", val_loss])
            log_file.flush()

        if (epoch + 1) % args.ckpt_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f"diffusion_epoch{epoch}.pth")
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch, "val_loss": val_loss, "args": vars(args)}, ckpt_path)
            print(f"saved {ckpt_path}")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": epoch, "val_loss": val_loss, "args": vars(args)},
                   os.path.join(args.out_dir, "diffusion_latest.pth"))

    log_file.close()
    print("done.")


if __name__ == "__main__":
    main()
