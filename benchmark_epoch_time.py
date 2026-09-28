"""
Synthetic-data timing benchmark for the ALDM diffusion (LDM) stage.
Run from ~/Desktop/ALDM/LDM :  python benchmark_epoch_time.py
"""
import time
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

CONFIG_PATH = "configs/latent-diffusion/mri-3t7t-ldm.yaml"
BATCH_SIZE = 64
IMG_SIZE = 128
NUM_WARMUP = 3
NUM_TIMED = 10


def make_batch(bs, num_classes, img_size, device):
    return {
        "source": torch.randn(bs, 1, img_size, img_size, device=device),
        "target": torch.randn(bs, 1, img_size, img_size, device=device),
        "target_class": torch.randint(0, num_classes, (bs,), device=device),
    }


def main():
    config = OmegaConf.load(CONFIG_PATH)
    model = instantiate_from_config(config.model)
    model.learning_rate = config.model.base_learning_rate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    num_classes = config.model.params.unet_config.params.num_classes

    opt = model.configure_optimizers()
    if isinstance(opt, (list, tuple)):
        opt = opt[0][0] if isinstance(opt[0], (list, tuple)) else opt[0]

    print(f"device: {device}")
    print(f"batch_size: {BATCH_SIZE}")

    for _ in range(NUM_WARMUP):
        opt.zero_grad()
        loss, _ = model.shared_step(make_batch(BATCH_SIZE, num_classes, IMG_SIZE, device))
        loss.backward()
        opt.step()

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(NUM_TIMED):
        opt.zero_grad()
        loss, _ = model.shared_step(make_batch(BATCH_SIZE, num_classes, IMG_SIZE, device))
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    per_step = (t1 - t0) / NUM_TIMED
    print(f"time per training step: {per_step * 1000:.1f} ms  (batch_size={BATCH_SIZE})")
    print(f"-> for N steps/epoch, epoch time ~= N * {per_step:.4f} s")
    print(f"   e.g. dataset of 5000 pairs, batch_size {BATCH_SIZE} -> "
          f"{5000 // BATCH_SIZE} steps/epoch -> ~{(5000 // BATCH_SIZE) * per_step / 60:.1f} min/epoch")


if __name__ == "__main__":
    main()
