"""
train_autoencoder.py
Shared AutoencoderKL for 3T/7T MRI, trained with LSC's own self-supervised
corruption recipe (global blur + patch holes) instead of plain clean-image
reconstruction -- per advisor's directive: keep LSC's "detail restoration"
logic, just move it inside a proper VAE bottleneck so the resulting z is
usable for latent diffusion.

Pipeline per batch:
    imgs (clean, [-1,1])
      -> mae_masking()  =>  x (corrupted: global blur + patch holes), hole_mask
      -> model.encode(x) => z_mu, z_sigma   (+ skip feature maps, unused for loss)
      -> model.sampling() => z
      -> model.decode(z) => recon
      -> compare recon vs imgs (the CLEAN target), split into hole-region vs
         whole-image SSIM/L2, exactly like train_lsc.py's compute_metrics,
         plus a KL term on z_mu/z_sigma, plus domain split (3T vs 7T), plus
         a brain-only (background-excluded) SSIM diagnostic. The perceptual
         (LPIPS) loss term is BRAIN-ONLY as of this version -- background is
         near-constant and was diluting it hard (whole-image ~0.046 vs
         brain-only ~0.1335 on the same checkpoint); perceptual_whole is kept
         as a whole-image DIAGNOSTIC column for comparison, not used for
         backprop. SSIM/L2's "whole" term is still whole-image for now (the
         existing hole/whole split makes masking it a separate, messier
         change) -- deferred, not done in this version.

Data layout expected:
    <data_dir>/3T/train/*.nii(.gz)
    <data_dir>/3T/valid/*.nii(.gz)
    <data_dir>/7T/train/*.nii(.gz)
    <data_dir>/7T/valid/*.nii(.gz)
"""

import argparse
import csv
import os
import random
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_fn
from tqdm import tqdm

from monai.networks.nets import AutoencoderKL

try:
    import lpips
except ImportError:
    lpips = None  # only required if --perceptual_weight > 0; see main()'s check below

try:
    # ASSUMPTION flagged explicitly: these import paths are for the standalone
    # "generative" package (pip install monai-generative), which is what MONAI's
    # own AutoencoderKL+PatchDiscriminator tutorials use. If your MONAI version
    # has merged this into core, these imports will fail and --adv_weight > 0
    # will raise a clear error telling you to fix the import path below instead
    # of silently doing the wrong thing.
    from generative.networks.nets import PatchDiscriminator
    from generative.losses.adversarial_loss import PatchAdversarialLoss
except ImportError:
    PatchDiscriminator = None
    PatchAdversarialLoss = None  # only required if --adv_weight > 0; see main()'s check below


# ----------------------------------------------------------------------------
# LSC corruption model -- copied/vectorized straight from train_lsc.py
# ----------------------------------------------------------------------------
def gaussian_kernel2d(kernel_size, sigma, device):
    ax = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def mae_masking(image, num_patches=24, mask_ratio=0.2, global_blur_prob=0.7,
                 blur_sigma_range=(1.0, 2.5)):
    """
    image: (B,1,H,W) in [-1,1]
    returns: corrupted (B,1,H,W), hole_mask (B,1,H,W) 1=hole region
    """
    B, C, H, W = image.shape
    device = image.device

    # --- global blur, applied probabilistically per-sample ---
    blurred = image.clone()
    do_blur = (torch.rand(B, device=device) < global_blur_prob)
    if do_blur.any():
        sigma = torch.empty(1, device=device).uniform_(*blur_sigma_range).item()
        k = max(3, int(2 * round(3 * sigma) + 1))
        kernel = gaussian_kernel2d(k, sigma, device)
        blurred_all = F.conv2d(image, kernel, padding=k // 2)
        blurred = torch.where(do_blur.view(B, 1, 1, 1), blurred_all, image)

    # --- patch holes, vectorized (no python loop over patches) ---
    # Pick n_rows/n_cols directly as the grid closest to num_patches, THEN derive
    # patch size from that -- computing patch size first and flooring rows/cols
    # from it (the old way) silently shrinks the grid (e.g. num_patches=24 on a
    # 128x128 image was actually only giving a 4x4=16-cell grid), which both
    # under-shoots the requested mask_ratio coverage and makes each patch much
    # bigger than intended (near-1/4-of-the-image blocks instead of small,
    # scattered patches).
    n_side = max(1, int(round(num_patches ** 0.5)))
    n_rows = n_cols = n_side
    patch_h = max(2, H // n_rows)
    patch_w = max(2, W // n_cols)
    n_grid = n_rows * n_cols
    # NOTE: mask_ratio=0 must mean zero holes -- do NOT floor this to 1.
    # (Previously `max(1, ...)` forced one patch to always be masked out even
    # at mask_ratio=0, so "no masking" runs were silently still corrupting
    # ~1/n_grid of the image and ssim_hole was measuring that leftover patch
    # instead of being undefined/skipped.)
    n_hole = int(round(n_grid * mask_ratio))

    if n_hole == 0:
        # no holes requested: `blurred` already equals image.clone() when
        # global blur didn't fire, or the blurred version when it did --
        # either way it's the correct "corrupted" output with zero holes.
        hole_mask = torch.zeros(B, 1, H, W, device=device)
        return blurred, hole_mask

    rand_scores = torch.rand(B, n_grid, device=device)
    hole_idx = rand_scores.argsort(dim=1)[:, :n_hole]  # (B, n_hole)

    grid_mask = torch.zeros(B, n_grid, device=device)
    grid_mask.scatter_(1, hole_idx, 1.0)
    grid_mask = grid_mask.view(B, 1, n_rows, n_cols)
    hole_mask = F.interpolate(grid_mask, size=(n_rows * patch_h, n_cols * patch_w), mode="nearest")
    if hole_mask.shape[-2:] != (H, W):
        pad_h = H - hole_mask.shape[-2]
        pad_w = W - hole_mask.shape[-1]
        hole_mask = F.pad(hole_mask, (0, pad_w, 0, pad_h))

    fill_value = image.amin(dim=(2, 3), keepdim=True)
    corrupted = torch.where(hole_mask.bool(), fill_value, blurred)
    return corrupted, hole_mask


def auto_ssim_kernel(img_size, num_patches):
    patch_size = int(round((img_size * img_size / num_patches) ** 0.5))
    kernel_size = max(3, patch_size // 2)
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 3.0
    return kernel_size, sigma


def masked_avg(value_map, mask):
    denom = mask.sum()
    if denom.item() == 0:
        # nothing in the mask (e.g. mask_ratio=0 -> zero hole pixels) -- this
        # metric is undefined here, NOT zero. Returning 0/eps used to silently
        # read as "SSIM collapsed to 0" instead of "not applicable".
        return torch.tensor(float("nan"), device=value_map.device)
    return (value_map * mask).sum() / denom


def weighted_masked_avg(value_map, mask, sample_weight):
    """Same as masked_avg, but each sample in the batch is additionally scaled
    by sample_weight (B,1,1,1) before averaging -- e.g. sample_weight=domain_w
    where domain_w is higher for 7T samples than 3T ones. This makes 7T
    samples count for more in the gradient than their raw slice count in the
    batch would otherwise give them, WITHOUT changing what mask/region the
    loss is computed over.

    NOTE: mean() semantics, not sum() -- if sample_weight is all 1s this is
    identical to masked_avg (backward compatible)."""
    w = mask * sample_weight
    denom = w.sum()
    if denom.item() == 0:
        return torch.tensor(float("nan"), device=value_map.device)
    return (value_map * w).sum() / denom


def build_brain_mask(imgs_01, threshold=0.02):
    """
    imgs_01: (B,1,H,W) in [0,1], the CLEAN target (same imgs_01 convention as
    inside compute_metrics). Background in these MRI slices sits at/near 0
    after the dataset's per-volume min-max normalization, so a simple
    intensity threshold separates brain tissue from background -- no
    segmentation model needed, same trick as score_registration_quality.py's
    mask=(a>0)|(b>0).

    Thresholding the CLEAN target only (never recon) so the mask doesn't
    shift around as reconstruction quality changes epoch to epoch -- you
    want to compare the same region over time.
    """
    return (imgs_01 > threshold).float()


# ----------------------------------------------------------------------------
# Dataset: pools 3T + 7T slices, tags each sample with a domain flag (0=3T,1=7T)
# ----------------------------------------------------------------------------
class MRISliceDataset(Dataset):
    def __init__(self, data_dir, domain, split, img_size=128, cache_size=8):
        self.img_size = img_size
        self.domain = domain  # 0 = 3T, 1 = 7T
        folder = Path(data_dir) / ("3T" if domain == 0 else "7T") / split
        self.files = sorted(list(folder.glob("*.nii.gz")) + list(folder.glob("*.nii")))
        if len(self.files) == 0:
            raise RuntimeError(f"No nifti files found in {folder}")

        self.index = []  # (file_idx, slice_idx)
        for fi, f in enumerate(self.files):
            img = nib.load(str(f))
            n_slices = img.shape[2]
            lo = int(n_slices * 0.2)
            hi = int(n_slices * 0.8)
            for s in range(lo, hi):
                self.index.append((fi, s))

        self._cache = OrderedDict()
        self._cache_size = cache_size

    def __len__(self):
        return len(self.index)

    def _load_volume(self, fi):
        if fi in self._cache:
            self._cache.move_to_end(fi)
            return self._cache[fi]
        vol = nib.load(str(self.files[fi])).get_fdata(dtype=np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin > 1e-6:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol = np.zeros_like(vol)
        self._cache[fi] = vol
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vol

    def volume_id_for_index(self, idx):
        fi, _ = self.index[idx]
        return fi

    def __getitem__(self, idx):
        fi, s = self.index[idx]
        vol = self._load_volume(fi)
        sl = vol[:, :, s]
        sl = torch.from_numpy(sl).float().unsqueeze(0)  # (1,H,W)
        if sl.shape[-2:] != (self.img_size, self.img_size):
            sl = F.interpolate(sl.unsqueeze(0), size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False).squeeze(0)
        sl = sl * 2.0 - 1.0  # [0,1] -> [-1,1]
        domain = torch.tensor(self.domain, dtype=torch.long)
        return sl, domain


class ConcatDomainDataset(Dataset):
    def __init__(self, ds_3t, ds_7t):
        self.ds_3t = ds_3t
        self.ds_7t = ds_7t
        self.len_3t = len(ds_3t)

    def __len__(self):
        return self.len_3t + len(self.ds_7t)

    def volume_id_for_index(self, idx):
        if idx < self.len_3t:
            return ("3t", self.ds_3t.volume_id_for_index(idx))
        else:
            return ("7t", self.ds_7t.volume_id_for_index(idx - self.len_3t))

    def __getitem__(self, idx):
        if idx < self.len_3t:
            return self.ds_3t[idx]
        else:
            return self.ds_7t[idx - self.len_3t]


class VolumeGroupedShuffleSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        groups = {}
        for idx in range(len(self.dataset)):
            key = self.dataset.volume_id_for_index(idx)
            groups.setdefault(key, []).append(idx)
        keys = list(groups.keys())
        random.shuffle(keys)
        order = []
        for k in keys:
            idxs = groups[k]
            random.shuffle(idxs)
            order.extend(idxs)
        return iter(order)

    def __len__(self):
        return len(self.dataset)


# ----------------------------------------------------------------------------
# Encoder walk that exposes skip candidates + the raw mu/sigma
# ----------------------------------------------------------------------------
def encode_with_skips(model, x):
    h = x
    skips = []
    for block in model.encoder.blocks:
        if type(block).__name__ == "AEKLDownsample":
            skips.append(h)
        h = block(h)
    z_mu = model.quant_conv_mu(h)
    z_log_var = torch.clamp(model.quant_conv_log_sigma(h), -30.0, 20.0)
    z_sigma = torch.exp(z_log_var / 2)
    return z_mu, z_sigma, skips


def get_z(model, x):
    z_mu, z_sigma, skips = encode_with_skips(model, x)
    z = model.sampling(z_mu, z_sigma)
    return z, z_mu, z_sigma, skips


def kl_divergence(z_mu, z_sigma):
    z_log_var = 2 * torch.log(z_sigma.clamp_min(1e-8))
    kl = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - z_log_var - 1, dim=[1, 2, 3])
    return kl.mean()


# ----------------------------------------------------------------------------
# Metrics -- LSC-style hole/whole split, PLUS domain (3T/7T) split, PLUS KL,
# PLUS a brain-only (background-excluded) SSIM diagnostic
# ----------------------------------------------------------------------------
def compute_metrics(model, imgs, domains, args, ssim_kernel, lpips_fn=None):
    kernel_size, sigma = ssim_kernel

    x, hole_mask = mae_masking(
        imgs, num_patches=args.num_patches, mask_ratio=args.mask_ratio,
        global_blur_prob=args.global_blur_prob, blur_sigma_range=args.blur_sigma_range,
    )
    whole_mask = torch.ones_like(hole_mask)

    z_mu, z_sigma, _ = encode_with_skips(model, x)
    z = model.sampling(z_mu, z_sigma)
    recon = model.decode(z)

    # -- everything below is metric/loss math, not the heavy conv/attention work --
    # force fp32 here regardless of the surrounding AMP autocast context.
    # SSIM's local-variance ratio is exactly 1.0 on constant regions (e.g. the large
    # black background outside the brain) in theory, but under fp16 that ratio
    # rounds to slightly ABOVE 1.0 -- and MRI slices are mostly background, so the
    # error dominates the mean and grows every epoch as reconstruction improves
    # (verified empirically: fp16 SSIM on a synthetic mostly-constant image came
    # back ~0.998 with a per-pixel max of 1.004, vs a correct ~0.913 in fp32).
    with torch.autocast(device_type="cuda", enabled=False):
        recon_f = recon.float()
        imgs_f = imgs.float()
        x_f = x.float()

        recon_01 = (recon_f.clamp(-1, 1) + 1) / 2
        imgs_01 = (imgs_f.clamp(-1, 1) + 1) / 2
        x_01 = (x_f.clamp(-1, 1) + 1) / 2

        ssim_map_pred = ssim_fn(recon_01, imgs_01, data_range=1.0, kernel_size=kernel_size,
                                 sigma=sigma, return_full_image=True)[1]
        ssim_map_base = ssim_fn(x_01, imgs_01, data_range=1.0, kernel_size=kernel_size,
                                 sigma=sigma, return_full_image=True)[1]

        # diagnostic metrics -- UNWEIGHTED, so ssim_hole/ssim_whole in the logs/CSV
        # stay a plain, comparable-across-runs average regardless of domain_weight_7t.
        ssim_hole = masked_avg(ssim_map_pred, hole_mask)
        ssim_whole = masked_avg(ssim_map_pred, whole_mask)
        baseline_hole = masked_avg(ssim_map_base, hole_mask)
        baseline_whole = masked_avg(ssim_map_base, whole_mask)

        # brain-only SSIM: excludes background, so it isn't diluted by the
        # large near-perfect-by-default black region outside the brain -- a
        # stricter, more representative read of actual tissue reconstruction
        # quality. Diagnostic only, not part of recon_loss below.
        brain_mask = build_brain_mask(imgs_01, threshold=args.brain_mask_threshold)
        ssim_brain = masked_avg(ssim_map_pred, brain_mask)
        baseline_brain = masked_avg(ssim_map_base, brain_mask)

        l2_map = (recon_f - imgs_f) ** 2

        # per-sample domain weight used ONLY for the backprop loss below: 1.0
        # for 3T, args.domain_weight_7t for 7T. Default 1.0 -> identical to the
        # old unweighted behavior. This does NOT touch the diagnostic ssim_hole/
        # ssim_whole/ssim_brain/ssim_3t/ssim_7t above -- those stay a plain read
        # of how well each domain/region is actually reconstructing, so you can
        # still tell whether upweighting 7T is closing the gap or not.
        dom_w = torch.where(domains.view(-1, 1, 1, 1) == 1,
                             torch.tensor(args.domain_weight_7t, device=imgs_f.device, dtype=imgs_f.dtype),
                             torch.tensor(1.0, device=imgs_f.device, dtype=imgs_f.dtype))

        l2_hole_w = weighted_masked_avg(l2_map, hole_mask, dom_w)
        l2_whole_w = weighted_masked_avg(l2_map, whole_mask, dom_w)
        ssim_hole_w = weighted_masked_avg(ssim_map_pred, hole_mask, dom_w)
        ssim_whole_w = weighted_masked_avg(ssim_map_pred, whole_mask, dom_w)

        kl = kl_divergence(z_mu.float(), z_sigma.float())

        # perceptual (LPIPS) loss -- BRAIN-ONLY as of this version. Background
        # is near-constant and was diluting this hard: your own spatial-LPIPS
        # diagnostic showed whole-image ~0.046 vs brain-only ~0.1335 on the
        # SAME checkpoint, i.e. actual tissue perceptual error is ~3x worse
        # than the number the loss/log were tracking. Can't just zero out
        # background pixels before running the network though (LPIPS has real
        # receptive fields -> that corrupts local context at every boundary
        # near the brain edge, same rule spatial_lpips_diagnostic already
        # follows) -- correct order is: full-image forward pass (spatial=True,
        # per-pixel map) -> THEN masked_avg over brain_mask. Still fully
        # differentiable (masked_avg is just a weighted mean), so this trains
        # exactly like before, just against the real (harder) signal.
        # perceptual_whole is the OLD whole-image number, kept as a
        # side-by-side diagnostic only -- NOT part of recon_loss.
        # Grayscale (1,H,W) -> repeat to 3 channels, LPIPS expects RGB-like input
        # already in [-1,1], which is exactly the range imgs/recon are already in.
        #
        # NOTE ON perceptual_weight: this ~3x jump in the raw perceptual value
        # means the SAME --perceptual_weight now pulls roughly 3x harder than
        # before (weight is a multiplier on a number that just got ~3x bigger).
        # Resuming from ae_latest.pth with perceptual_weight still at 2.5 will
        # NOT behave like "one more epoch at 2.5" -- it's closer to jumping to
        # ~7-ish under the old (whole-image) scale. Consider dropping the
        # weight back down (roughly weight_new ~= weight_old * 0.046/0.1335 =
        # weight_old * 0.34, so 2.5 -> ~0.85) if you want a comparable
        # starting pull, and watch loss/kl closely for a few epochs either way.
        perceptual = torch.tensor(0.0, device=imgs_f.device)
        perceptual_whole = torch.tensor(0.0, device=imgs_f.device)
        if lpips_fn is not None and args.perceptual_weight > 0:
            recon_rgb = recon_f.clamp(-1, 1).repeat(1, 3, 1, 1)
            imgs_rgb = imgs_f.repeat(1, 3, 1, 1)
            perceptual_map = lpips_fn(recon_rgb, imgs_rgb)  # (B,1,H,W), spatial=True
            perceptual = masked_avg(perceptual_map, brain_mask)
            perceptual_whole = perceptual_map.mean()

        recon_loss = (l2_hole_w + args.alpha * (1 - ssim_hole_w)) + \
                     args.whole_weight * (l2_whole_w + args.alpha * (1 - ssim_whole_w)) + \
                     args.perceptual_weight * perceptual
        loss = recon_loss + args.kl_weight * kl

        mask_3t = domains == 0
        mask_7t = domains == 1
        ssim_3t = torch.tensor(float("nan"))
        ssim_7t = torch.tensor(float("nan"))
        if mask_3t.any():
            ssim_3t = ssim_fn(recon_01[mask_3t], imgs_01[mask_3t], data_range=1.0)
        if mask_7t.any():
            ssim_7t = ssim_fn(recon_01[mask_7t], imgs_01[mask_7t], data_range=1.0)

    return {
        "loss": loss,
        "kl": kl,
        "perceptual": perceptual.detach() if torch.is_tensor(perceptual) else torch.tensor(perceptual),
        "perceptual_whole": perceptual_whole.detach() if torch.is_tensor(perceptual_whole) else torch.tensor(perceptual_whole),
        "ssim_hole": ssim_hole,
        "baseline_hole": baseline_hole,
        "ssim_whole": ssim_whole,
        "baseline_whole": baseline_whole,
        "ssim_brain": ssim_brain,
        "baseline_brain": baseline_brain,
        "ssim_3t": ssim_3t,
        "ssim_7t": ssim_7t,
        # fp32, [-1,1]-clamped, STILL ATTACHED TO THE GRAPH (not detached) --
        # needed by the generator's adversarial loss term below, which must
        # backprop through the decoder. Detach it yourself if you just want
        # to look at it (e.g. the discriminator's own training step does
        # recon.detach()).
        "recon": recon_f.clamp(-1, 1),
    }


# ----------------------------------------------------------------------------
# Spatial LPIPS diagnostic -- answers "感知損失度低,但不知道確切是哪裡低"
# without any patch cropping/stitching (the thing that reintroduced blur for
# the labmate's approach). lpips.LPIPS(spatial=True) returns a full-resolution
# per-pixel perceptual-difference map from ONE whole-image forward pass (the
# network's receptive fields act like implicit overlapping patches -- nothing
# is physically cropped or blended). We average that map over many validation
# slices per domain so you get a stable picture of WHERE (which anatomical
# region) the perceptual error concentrates, not just a single noisy slice.
# Also reports a brain-only (background-excluded) mean alongside the
# whole-image mean, using the same intensity-threshold mask as ssim_brain --
# background pixels are trivially near-perfect, so the whole-image mean
# understates how much perceptual error is actually sitting in tissue.
#
# Uses the CLEAN encode/decode path (z_mu, no mae_masking corruption) -- same
# convention as eval_clean_ssim.py / save_recon_comparison.py, since the
# "flattened detail" complaint was about clean reconstruction quality, not
# about hole-filling.
# ----------------------------------------------------------------------------
def spatial_lpips_diagnostic(model, valid_loader, device, lpips_spatial_fn, out_dir, epoch, n_samples=32,
                              brain_mask_threshold=0.02):
    model.eval()
    target_per_domain = n_samples
    count = {0: 0, 1: 0}
    sum_map = {0: None, 1: None}
    sum_img = {0: None, 1: None}  # running average of the original image, as an anatomical reference underlay
    sum_masked_lpips = {0: 0.0, 1: 0.0}  # running sum of per-slice brain-only mean LPIPS
    masked_count = {0: 0, 1: 0}          # slices where the brain mask was non-empty

    # NOTE: valid_loader's ConcatDomainDataset lays out ALL 3T indices before
    # ANY 7T ones (shuffle=False on the valid split), so this loop scans the
    # full 3T tail before 7T samples start arriving -- bounded by one pass
    # over the validation set, which is fine since this only runs every
    # --spatial_diag_every epochs, not every epoch.
    with torch.no_grad():
        for imgs, domains in valid_loader:
            if count[0] >= target_per_domain and count[1] >= target_per_domain:
                break
            imgs = imgs.to(device)
            d_np = domains.numpy()
            for i in range(len(d_np)):
                dv = int(d_np[i])
                if count[dv] >= target_per_domain:
                    continue
                img1 = imgs[i:i + 1]  # clean, [-1,1]
                z_mu, z_sigma, _ = encode_with_skips(model, img1)
                recon = model.decode(z_mu)
                recon_rgb = recon.clamp(-1, 1).repeat(1, 3, 1, 1)
                img_rgb = img1.repeat(1, 3, 1, 1)
                dmap = lpips_spatial_fn(recon_rgb, img_rgb).float().cpu()  # (1,1,H,W)

                img1_01 = ((img1.clamp(-1, 1) + 1) / 2).float().cpu()
                brain_mask = build_brain_mask(img1_01, threshold=brain_mask_threshold)
                mval = masked_avg(dmap, brain_mask)
                if torch.isfinite(mval):
                    sum_masked_lpips[dv] += mval.item()
                    masked_count[dv] += 1

                if sum_map[dv] is None:
                    sum_map[dv] = dmap.clone()
                    sum_img[dv] = img1.float().cpu()
                else:
                    sum_map[dv] += dmap
                    sum_img[dv] += img1.float().cpu()
                count[dv] += 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    domain_names = {0: "3T", 1: "7T"}
    means = {}
    masked_means = {}
    for col, dv in enumerate([0, 1]):
        if sum_map[dv] is None or count[dv] == 0:
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            continue
        avg_map = (sum_map[dv] / count[dv])[0, 0].numpy()
        avg_img = ((sum_img[dv] / count[dv])[0, 0].numpy() + 1) / 2  # [-1,1] -> [0,1] for display
        means[dv] = float(avg_map.mean())
        masked_means[dv] = (sum_masked_lpips[dv] / masked_count[dv]
                             if masked_count[dv] > 0 else float("nan"))

        axes[0, col].imshow(avg_img, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{domain_names[dv]} avg image (n={count[dv]})")
        axes[0, col].axis("off")

        im = axes[1, col].imshow(avg_map, cmap="inferno")
        axes[1, col].set_title(f"{domain_names[dv]} avg spatial LPIPS\n"
                                f"whole-image mean={means[dv]:.4f}, brain-only mean={masked_means[dv]:.4f}")
        axes[1, col].axis("off")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    fig.suptitle(f"Spatial LPIPS diagnostic -- epoch {epoch}\n"
                 f"clean recon vs clean target, z_mu decode -- brighter = worse perceptual match here")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"spatial_lpips_epoch{epoch}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mean_str = ", ".join(
        f"{domain_names[dv]} whole={means[dv]:.4f}/brain={masked_means[dv]:.4f}" for dv in means)
    print(f"  [spatial LPIPS diagnostic] saved {out_path} ({mean_str})")


# ----------------------------------------------------------------------------
# Clean (uncorrupted) validation metrics -- PSNR/SSIM/MAE/LPIPS with NO
# mae_masking at all, i.e. "how good is the AE by itself", same convention as
# eval_full_metrics.py / the "validation(不挖空不模糊)" table: z_mu deterministic
# encode/decode of the untouched clean imgs, no sampling, no hole/blur.
# Answers a DIFFERENT question than log.csv's val_* columns (those all score
# detail-restoration-from-corruption, since compute_metrics always runs
# mae_masking first) -- this is the number you actually want to watch when
# deciding whether perceptual_weight is helping or hurting real reconstruction.
# ----------------------------------------------------------------------------
CLEAN_REGIONS = ["all", "3t", "7t"]
CLEAN_METRIC_NAMES = ["ssim", "psnr", "mae", "perceptual", "n"]


def compute_clean_domain_metrics(model, imgs, domains, ssim_kernel, lpips_fn=None):
    """One batch -> dict region -> {ssim, psnr, mae, perceptual, n} (region in
    CLEAN_REGIONS). No_grad, no AMP -- this only runs at validation time and
    is metric math, not something that needs to be fast under autocast."""
    kernel_size, sigma = ssim_kernel
    with torch.no_grad():
        z_mu, z_sigma, _ = encode_with_skips(model, imgs)
        recon = model.decode(z_mu)

        recon_f = recon.float().clamp(-1, 1)
        imgs_f = imgs.float()
        recon_01 = (recon_f + 1) / 2
        imgs_01 = (imgs_f.clamp(-1, 1) + 1) / 2

        def region_metrics(sel):
            n = int(sel.sum().item())
            if n == 0:
                return None
            r01, i01 = recon_01[sel], imgs_01[sel]
            rf, iF = recon_f[sel], imgs_f[sel]
            ssim_val = ssim_fn(r01, i01, data_range=1.0, kernel_size=kernel_size, sigma=sigma).item()
            psnr_val = psnr_fn(r01, i01, data_range=1.0).item()
            mae_val = (rf - iF).abs().mean().item()
            perceptual_val = float("nan")
            if lpips_fn is not None:
                r_rgb = rf.repeat(1, 3, 1, 1)
                i_rgb = iF.repeat(1, 3, 1, 1)
                perceptual_val = lpips_fn(r_rgb, i_rgb).mean().item()
            return {"ssim": ssim_val, "psnr": psnr_val, "mae": mae_val,
                    "perceptual": perceptual_val, "n": n}

        out = {
            "all": region_metrics(torch.ones_like(domains, dtype=torch.bool)),
            "3t": region_metrics(domains == 0),
            "7t": region_metrics(domains == 1),
        }
    return out


def run_clean_eval(model, valid_loader, device, ssim_kernel, lpips_fn):
    """Full pass over valid_loader (every slice, not a sample) in clean z_mu
    decode mode. Returns region -> {ssim, psnr, mae, perceptual, n}, each a
    weighted mean across all batches (weighted by how many samples of that
    region actually appeared -- VolumeGroupedShuffleSampler makes most
    batches single-domain, so a plain mean-of-batch-means would over-weight
    small-n batches relative to their real sample count)."""
    model.eval()
    sums = {r: {"ssim": 0.0, "psnr": 0.0, "mae": 0.0, "perceptual": 0.0, "n": 0}
            for r in CLEAN_REGIONS}
    with torch.no_grad():
        for imgs, domains in valid_loader:
            imgs = imgs.to(device, non_blocking=True)
            domains = domains.to(device, non_blocking=True)
            batch_out = compute_clean_domain_metrics(model, imgs, domains, ssim_kernel, lpips_fn)
            for r, m in batch_out.items():
                if m is None:
                    continue
                n = m["n"]
                sums[r]["ssim"] += m["ssim"] * n
                sums[r]["psnr"] += m["psnr"] * n
                sums[r]["mae"] += m["mae"] * n
                if lpips_fn is not None and m["perceptual"] == m["perceptual"]:  # not NaN
                    sums[r]["perceptual"] += m["perceptual"] * n
                sums[r]["n"] += n

    result = {}
    for r in CLEAN_REGIONS:
        n = sums[r]["n"]
        if n == 0:
            result[r] = {"ssim": float("nan"), "psnr": float("nan"), "mae": float("nan"),
                         "perceptual": float("nan"), "n": 0}
        else:
            result[r] = {
                "ssim": sums[r]["ssim"] / n,
                "psnr": sums[r]["psnr"] / n,
                "mae": sums[r]["mae"] / n,
                "perceptual": (sums[r]["perceptual"] / n) if lpips_fn is not None else float("nan"),
                "n": n,
            }
    return result


# ----------------------------------------------------------------------------
# Train / validate
# ----------------------------------------------------------------------------
def make_loader(data_dir, split, img_size, batch_size, num_workers):
    ds_3t = MRISliceDataset(data_dir, domain=0, split=split, img_size=img_size)
    ds_7t = MRISliceDataset(data_dir, domain=1, split=split, img_size=img_size)
    ds = ConcatDomainDataset(ds_3t, ds_7t)
    sampler = VolumeGroupedShuffleSampler(ds, batch_size) if split == "train" else None
    loader = DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        shuffle=False, num_workers=num_workers, drop_last=(split == "train"),
        pin_memory=True,
    )
    return loader


KEYS = ["loss", "kl", "perceptual", "perceptual_whole", "ssim_hole", "baseline_hole", "ssim_whole",
        "baseline_whole", "ssim_brain", "baseline_brain", "ssim_3t", "ssim_7t", "adv_g", "adv_d"]


def open_csv_with_schema_check(csv_path, expected_header, label):
    """Open csv_path for appending, writing expected_header only if the file
    is new. If it already exists with a DIFFERENT header (schema drifted
    since the file was created -- e.g. a column got added), move it aside as
    <path>.schema_mismatch_backup[N] (never delete) and start fresh with the
    current header, so old and new rows never end up misaligned under one
    header. Same protection log.csv already had, factored out so
    clean_val_metrics.csv gets it too. Returns (file_handle, csv_writer)."""
    write_header = True
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header == expected_header:
            write_header = False
        else:
            backup_path = csv_path + ".schema_mismatch_backup"
            n = 1
            while os.path.exists(backup_path):
                n += 1
                backup_path = csv_path + f".schema_mismatch_backup{n}"
            os.rename(csv_path, backup_path)
            print(f"[warn] {csv_path}'s header didn't match the current {label} schema "
                  f"(likely an older run, before some column existed) -- moved it to "
                  f"{backup_path} and starting a fresh file with the correct header.")
    f = open(csv_path, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(expected_header)
    return f, writer


def run_epoch(model, loader, optimizer, device, args, ssim_kernel, train, scaler=None, fixed_seed=None,
              lpips_fn=None, discriminator=None, disc_optimizer=None, adv_loss_fn=None, use_adv=False):
    model.train(train)
    if discriminator is not None:
        discriminator.train(train and use_adv)
    totals = {k: 0.0 for k in KEYS}
    counts = {k: 0 for k in KEYS}
    n_batches = 0
    use_amp = args.amp and device.type == "cuda"

    pbar = tqdm(loader, desc="train" if train else "valid")
    for imgs, domains in pbar:
        imgs = imgs.to(device, non_blocking=True)
        domains = domains.to(device, non_blocking=True)

        if fixed_seed is not None:
            g_state = torch.get_rng_state()
            torch.manual_seed(fixed_seed + n_batches)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                m = compute_metrics(model, imgs, domains, args, ssim_kernel, lpips_fn=lpips_fn)

            # adversarial (generator) term -- computed whenever the discriminator
            # is active, even during validation (forward-only there, just a
            # health-check number, no grad since torch.set_grad_enabled(train)
            # is already False for val). Run in plain fp32, outside the AMP
            # autocast used for the AE forward, matching how compute_metrics
            # already force-fp32's its own loss math (see the comment there
            # about SSIM under fp16).
            adv_g = torch.tensor(0.0, device=imgs.device)
            if use_adv and discriminator is not None:
                with torch.autocast(device_type="cuda", enabled=False):
                    fake_logits_g = discriminator(m["recon"])[-1]
                    adv_g = adv_loss_fn(fake_logits_g, target_is_real=True, for_discriminator=False)

        if fixed_seed is not None:
            torch.set_rng_state(g_state)

        if not torch.isfinite(m["loss"]):
            continue

        g_loss = m["loss"] + args.adv_weight * adv_g if (use_adv and discriminator is not None) else m["loss"]

        adv_d = torch.tensor(0.0, device=imgs.device)
        if train:
            # ---- generator (AE) step -- uses g_loss, which includes adv_g on top
            # of the usual recon+kl+perceptual loss when the discriminator is on ----
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(g_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    # bad batch under fp16 -- skip the step but keep the scaler happy
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # ---- discriminator step -- separate optimizer, real vs recon.detach()
            # (detached so this step's gradient never touches the AE's weights) ----
            if use_adv and discriminator is not None:
                with torch.autocast(device_type="cuda", enabled=False):
                    real_logits = discriminator(imgs.float())[-1]
                    fake_logits_d = discriminator(m["recon"].detach())[-1]
                    loss_d_real = adv_loss_fn(real_logits, target_is_real=True, for_discriminator=True)
                    loss_d_fake = adv_loss_fn(fake_logits_d, target_is_real=False, for_discriminator=True)
                    adv_d = (loss_d_real + loss_d_fake) * 0.5
                disc_optimizer.zero_grad()
                adv_d.backward()
                disc_optimizer.step()

        # loss/kl/perceptual/ssim_hole/ssim_whole/adv_g/adv_d always come from a
        # live model pass -> finite (adv_g/adv_d are just 0.0 tensors when the
        # discriminator is off, still finite)
        for k in ["loss", "kl", "perceptual", "perceptual_whole", "ssim_hole", "ssim_whole"]:
            totals[k] += m[k].item()
            counts[k] += 1
        totals["adv_g"] += adv_g.item(); counts["adv_g"] += 1
        totals["adv_d"] += adv_d.item(); counts["adv_d"] += 1
        # baseline_* / ssim_brain / baseline_brain / ssim_3t / ssim_7t can be NaN
        # (SSIM edge case on constant hole-fill regions, an empty brain mask, or
        # a domain missing from this batch) -- guard each independently so one
        # NaN doesn't poison the whole epoch's running sum
        for k in ["baseline_hole", "baseline_whole", "ssim_brain", "baseline_brain", "ssim_3t", "ssim_7t"]:
            if torch.isfinite(m[k]):
                totals[k] += m[k].item()
                counts[k] += 1

        n_batches += 1
        pbar.set_postfix(loss=m["loss"].item(), ssim_hole=m["ssim_hole"].item(),
                          ssim_whole=m["ssim_whole"].item(),
                          **({"adv_g": adv_g.item(), "adv_d": adv_d.item()} if use_adv else {}))

    result = {k: (totals[k] / counts[k] if counts[k] > 0 else float("nan")) for k in KEYS}
    result["n_batches"] = n_batches
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="checkpoints_ae")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl_weight", type=float, default=1e-6)
    p.add_argument("--latent_channels", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    # LSC corruption params -- same defaults as train_lsc.py
    p.add_argument("--num_patches", type=int, default=24)
    p.add_argument("--mask_ratio", type=float, default=0.2)
    p.add_argument("--global_blur_prob", type=float, default=0.7)
    p.add_argument("--blur_sigma_range", type=float, nargs=2, default=(1.0, 2.5))
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--whole_weight", type=float, default=0.5)
    p.add_argument("--domain_weight_7t", type=float, default=1.0,
                    help="multiplies 7T samples' contribution to the recon loss (l2/ssim, "
                         "hole+whole) relative to 3T's fixed weight of 1.0. Default 1.0 = no "
                         "change from before. This does NOT touch the model input (it still "
                         "has no domain signal) or the diagnostic ssim_3t/ssim_7t metrics -- "
                         "watch those two in the logs to see if raising this actually closes "
                         "the 3T-vs-7T reconstruction gap, or if the gap is coming from "
                         "something else (e.g. 7T being intrinsically harder to reconstruct "
                         "under the same corruption recipe, not just under-weighted).")
    p.add_argument("--perceptual_weight", type=float, default=0.0,
                    help="weight on an LPIPS perceptual loss term (brain-masked -- background "
                         "excluded via --brain_mask_threshold -- VGG-based), added "
                         "on TOP of the existing L2+SSIM recon_loss -- does not replace it. Default "
                         "0.0 = off (identical behavior to before this flag existed, and lpips does "
                         "not need to be installed). Start small (e.g. 0.1-0.5) if you turn it on; "
                         "this is step 1 of the blur fix (see save_recon_comparison.py to check "
                         "whether fine texture actually comes back before considering adversarial "
                         "loss on top of this).")
    p.add_argument("--spatial_diag_every", type=int, default=0,
                    help="if > 0, every N epochs save a per-domain (3T/7T) AVERAGE spatial-LPIPS "
                         "heatmap (lpips.LPIPS(spatial=True)) over --spatial_diag_samples clean "
                         "validation slices per domain, to out_dir/spatial_lpips_epoch<N>.png. "
                         "Shows WHERE the perceptual error concentrates without any patch cropping "
                         "or stitching, and reports both a whole-image mean and a brain-only "
                         "(background-excluded) mean. Default 0 = off. Independent of "
                         "--perceptual_weight -- you can turn this on for diagnosis even while "
                         "perceptual_weight=0.")
    p.add_argument("--spatial_diag_samples", type=int, default=32,
                    help="clean validation slices per domain to average into the spatial LPIPS "
                         "diagnostic heatmap (see --spatial_diag_every).")
    p.add_argument("--brain_mask_threshold", type=float, default=0.02,
                    help="intensity threshold (on the [0,1]-normalized clean target) used to "
                         "separate brain tissue from background for the ssim_brain/baseline_brain "
                         "diagnostic metrics and the spatial-LPIPS brain-only mean. Background in "
                         "these slices sits at/near 0 after per-volume min-max normalization, so "
                         "this doesn't need a segmentation model.")
    p.add_argument("--clean_eval_every", type=int, default=1,
                    help="if > 0, every N epochs run a FULL pass over the validation set with NO "
                         "mae_masking at all (z_mu deterministic encode/decode of the untouched "
                         "clean images) and log whole-image + per-domain (3T/7T) PSNR/SSIM/MAE/"
                         "LPIPS to out_dir/clean_val_metrics.csv -- same convention as "
                         "eval_full_metrics.py / the 'validation(不挖空不模糊)' table, and a "
                         "DIFFERENT question than log.csv's val_* columns (those score detail-"
                         "restoration-from-corruption, since compute_metrics always runs "
                         "mae_masking first). Default 1 = every epoch, matching log.csv's own "
                         "cadence. This is an EXTRA full forward pass over the whole validation "
                         "set on top of the normal val epoch (unlike --spatial_diag_every, which "
                         "only samples --spatial_diag_samples slices/domain) -- if it measurably "
                         "slows training, raise this (e.g. 5) rather than turning it off, since "
                         "this is the number you actually want when deciding whether a "
                         "perceptual_weight change helped or hurt. 0 = off.")
    p.add_argument("--adv_weight", type=float, default=0.0,
                    help="weight on a PatchGAN adversarial loss term, added on top of the existing "
                         "recon+kl+perceptual loss -- does not replace it. Default 0.0 = off (no "
                         "discriminator is even constructed, identical behavior to before this flag "
                         "existed). This is step 2 of the blur fix, AFTER perceptual_weight has "
                         "plateaued (LPIPS stops improving at a fixed weight even with more epochs) "
                         "-- pure reconstruction+perceptual loss tends toward a smoothed 'best "
                         "average' output; adversarial loss is what can push fine texture past that "
                         "point, at the cost of some risk of hallucinated-but-plausible detail, which "
                         "matters more here than in natural-image work since this is medical imaging. "
                         "Start small (e.g. 0.1-0.5) and watch ssim_hole/ssim_whole for a real "
                         "downward trend -- that's the sign adv_weight is too high for the recon loss "
                         "to hold structure against.")
    p.add_argument("--disc_lr", type=float, default=1e-4,
                    help="learning rate for the discriminator's own Adam optimizer (separate from "
                         "--lr, which is only the AE's). Only used if --adv_weight > 0.")
    p.add_argument("--disc_start_epoch", type=int, default=0,
                    help="epoch (in the ABSOLUTE numbering used by --epochs/checkpoints, not "
                         "'epochs since resume') at which the discriminator starts training and "
                         "contributing to the AE loss. Default 0 = starts immediately on resume. "
                         "Common practice when training a VAE from scratch is a warmup of several "
                         "epochs of pure reconstruction before introducing the discriminator (an "
                         "undertrained decoder + an already-trainable discriminator is a classic "
                         "recipe for unstable/collapsing GAN training) -- less critical here since "
                         "you're resuming a model that's already well into training, but raise this "
                         "if you see the first few epochs after turning --adv_weight on look unstable "
                         "(loss spikes, ssim_hole/ssim_whole dropping hard).")
    p.add_argument("--val_seed", type=int, default=1234)
    p.add_argument("--amp", action="store_true", default=True,
                    help="Mixed precision (fp16) training. On by default -- cuts VRAM "
                         "usage roughly in half, which matters a lot on an 8GB card.")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler(device="cuda", enabled=(args.amp and device.type == "cuda"))
    if args.amp and device.type == "cuda":
        print("AMP (fp16) enabled.")

    ssim_kernel = auto_ssim_kernel(args.img_size, args.num_patches)

    model = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 128),
        latent_channels=args.latent_channels,
        num_res_blocks=2,
        attention_levels=(False, False, False, True),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # One shared LPIPS net, now ALWAYS built with spatial=True: the perceptual
    # loss term (--perceptual_weight > 0) needs the per-pixel map so it can be
    # masked to brain_mask before averaging (see compute_metrics) -- zeroing
    # background pixels before running the network would corrupt local context
    # at every receptive-field boundary near the brain edge, so the correct
    # order is full-image forward -> per-pixel map -> masked average, same
    # rule spatial_lpips_diagnostic already followed. spatial=True's per-pixel
    # map gives an identical number to the old spatial=False scalar when you
    # just .mean() it with no masking (an unweighted mean over a rectangular
    # H*W grid doesn't care whether the reduction happens inside the network
    # or in your own code) -- so compute_clean_domain_metrics/run_clean_eval's
    # existing `.mean()` calls need no change and keep reporting the same
    # whole-image numbers as before (clean_val_metrics.csv is still
    # whole-image only, not brain-masked -- separate question from the loss).
    # Built once here if either the loss or the clean-eval CSV needs it, so
    # turning clean_eval on with perceptual_weight=0 doesn't allocate a
    # second VGG network on top of the (already-off) loss one.
    lpips_fn = None
    if args.perceptual_weight > 0 or args.clean_eval_every > 0:
        if lpips is None:
            raise SystemExit("--perceptual_weight > 0 or --clean_eval_every > 0 requires the "
                              "lpips package: pip install lpips")
        lpips_fn = lpips.LPIPS(net="vgg", spatial=True).to(device)
        lpips_fn.eval()
        for p_ in lpips_fn.parameters():
            p_.requires_grad_(False)
        if args.perceptual_weight > 0:
            print(f"Perceptual (LPIPS/VGG) loss enabled, weight={args.perceptual_weight} "
                  f"(brain-masked, brain_mask_threshold={args.brain_mask_threshold})")
        if args.clean_eval_every > 0:
            print(f"Clean-eval CSV enabled: every {args.clean_eval_every} epoch(s), full "
                  f"validation pass, no corruption -> {args.out_dir}/clean_val_metrics.csv")

    # spatial_lpips_diagnostic just needs a spatial=True net too -- if lpips_fn
    # already exists (built above) it's already in that mode, so reuse it
    # instead of allocating a second VGG network. Only build a dedicated one
    # when --spatial_diag_every is on by itself (perceptual_weight==0 AND
    # clean_eval_every==0).
    lpips_spatial_fn = None
    if args.spatial_diag_every > 0:
        if lpips_fn is not None:
            lpips_spatial_fn = lpips_fn
            print(f"Spatial LPIPS diagnostic enabled: every {args.spatial_diag_every} epochs, "
                  f"{args.spatial_diag_samples} slices/domain -> {args.out_dir}/spatial_lpips_epoch<N>.png "
                  f"(brain_mask_threshold={args.brain_mask_threshold}, reusing the shared LPIPS network)")
        else:
            if lpips is None:
                raise SystemExit("--spatial_diag_every > 0 requires the lpips package: "
                                  "pip install lpips")
            lpips_spatial_fn = lpips.LPIPS(net="vgg", spatial=True).to(device)
            lpips_spatial_fn.eval()
            for p_ in lpips_spatial_fn.parameters():
                p_.requires_grad_(False)
            print(f"Spatial LPIPS diagnostic enabled: every {args.spatial_diag_every} epochs, "
                  f"{args.spatial_diag_samples} slices/domain -> {args.out_dir}/spatial_lpips_epoch<N>.png "
                  f"(brain_mask_threshold={args.brain_mask_threshold})")

    discriminator = None
    disc_optimizer = None
    adv_loss_fn = None
    if args.adv_weight > 0:
        if PatchDiscriminator is None or PatchAdversarialLoss is None:
            raise SystemExit(
                "--adv_weight > 0 requires MONAI's generative extension: "
                "pip install monai-generative\n"
                "(if your MONAI version has this merged into core instead of the separate "
                "'generative' package, edit the import block near the top of this file to match "
                "wherever PatchDiscriminator/PatchAdversarialLoss actually live for your install)"
            )
        discriminator = PatchDiscriminator(
            spatial_dims=2, in_channels=1, num_layers_d=3, num_channels=64,
        ).to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr)
        adv_loss_fn = PatchAdversarialLoss(criterion="least_squares")
        print(f"Adversarial (PatchGAN) loss enabled: adv_weight={args.adv_weight}, "
              f"disc_lr={args.disc_lr}, active from epoch {args.disc_start_epoch} onward")

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")
        if discriminator is not None:
            if "discriminator" in ckpt:
                discriminator.load_state_dict(ckpt["discriminator"])
                disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
                print("  resumed discriminator + its optimizer state too")
            else:
                print("  [info] resume checkpoint has no discriminator state (it predates "
                      "--adv_weight) -- starting a freshly-initialized discriminator")

    # Best-validation checkpoint tracking -- a safety net against the model
    # continuing to overfit (train metrics improving, val metrics flat or
    # worsening) the longer you keep training. ae_best.pth always holds the
    # checkpoint with the LOWEST val_loss seen so far (the composite loss:
    # l2+ssim+perceptual+kl), so you can keep training past the point things
    # start overfitting without losing the best version -- no need to
    # manually watch every epoch and decide when to stop.
    best_val_path = os.path.join(args.out_dir, "ae_best.pth")
    best_val_loss = float("inf")
    if os.path.exists(best_val_path):
        try:
            best_ckpt_probe = torch.load(best_val_path, map_location=device)
            best_val_loss = best_ckpt_probe.get("val_loss", float("inf"))
            print(f"Existing ae_best.pth found (epoch {best_ckpt_probe.get('epoch', '?')}, "
                  f"val_loss={best_val_loss:.4f}) -- will only overwrite it if a new epoch beats this.")
        except Exception as e:
            print(f"[warn] couldn't read val_loss from existing {best_val_path} ({e}); "
                  f"treating best_val_loss as unknown (inf) -- next epoch will overwrite it.")

    train_loader = make_loader(args.data_dir, "train", args.img_size, args.batch_size, args.num_workers)
    valid_loader = make_loader(args.data_dir, "valid", args.img_size, args.batch_size, args.num_workers)

    n_3t_slices = len(train_loader.dataset.ds_3t)
    n_7t_slices = len(train_loader.dataset.ds_7t)
    n_3t_vols = len(train_loader.dataset.ds_3t.files)
    n_7t_vols = len(train_loader.dataset.ds_7t.files)
    print(f"domain_weight_7t={args.domain_weight_7t}  |  "
          f"train slices: 3T={n_3t_slices} ({n_3t_vols} volumes), 7T={n_7t_slices} ({n_7t_vols} volumes)")
    if n_3t_slices > 0 and n_7t_slices > 0:
        ratio = n_3t_slices / n_7t_slices
        if ratio > 1.3 or ratio < 1 / 1.3:
            print(f"  -> {ratio:.2f}x slice-count imbalance between domains. Batches are drawn "
                  f"per-volume (VolumeGroupedShuffleSampler), so most batches end up single-domain "
                  f"-- with this much imbalance, the model sees far more 3T-only (or 7T-only) "
                  f"batches per epoch than the other, independent of anything domain_weight_7t does.")

    csv_path = os.path.join(args.out_dir, "log.csv")
    expected_header = ["epoch"]
    for prefix in ["train", "val"]:
        expected_header += [f"{prefix}_{k}" for k in KEYS]
    expected_header += ["ssim_kernel", "n_train_batches", "n_val_batches"]
    csv_file, csv_writer = open_csv_with_schema_check(csv_path, expected_header, "KEYS")

    # clean_val_metrics.csv -- separate file, separate cadence (--clean_eval_every),
    # separate question ("how good is the AE with no corruption at all") from
    # log.csv's val_* columns. Same schema-mismatch backup protection as log.csv.
    clean_csv_file = None
    clean_csv_writer = None
    if args.clean_eval_every > 0:
        clean_csv_path = os.path.join(args.out_dir, "clean_val_metrics.csv")
        clean_expected_header = ["epoch"]
        for region in CLEAN_REGIONS:
            clean_expected_header += [f"{region}_{m}" for m in CLEAN_METRIC_NAMES]
        clean_csv_file, clean_csv_writer = open_csv_with_schema_check(
            clean_csv_path, clean_expected_header, "CLEAN_REGIONS/CLEAN_METRIC_NAMES")

    for epoch in range(start_epoch, args.epochs + 1):
        use_adv = discriminator is not None and epoch >= args.disc_start_epoch

        train_res = run_epoch(model, train_loader, optimizer, device, args, ssim_kernel,
                               train=True, scaler=scaler, fixed_seed=None, lpips_fn=lpips_fn,
                               discriminator=discriminator, disc_optimizer=disc_optimizer,
                               adv_loss_fn=adv_loss_fn, use_adv=use_adv)
        with torch.no_grad():
            val_res = run_epoch(model, valid_loader, optimizer, device, args, ssim_kernel,
                                 train=False, scaler=scaler, fixed_seed=args.val_seed, lpips_fn=lpips_fn,
                                 discriminator=discriminator, disc_optimizer=disc_optimizer,
                                 adv_loss_fn=adv_loss_fn, use_adv=use_adv)

        adv_str = (f" | adv_g={train_res['adv_g']:.4f} adv_d={train_res['adv_d']:.4f}"
                   if use_adv else "")
        print(f"[Epoch {epoch}] "
              f"train loss={train_res['loss']:.4f} ssim_hole={train_res['ssim_hole']:.4f} "
              f"ssim_whole={train_res['ssim_whole']:.4f} ssim_brain={train_res['ssim_brain']:.4f} "
              f"ssim_3t={train_res['ssim_3t']:.4f} ssim_7t={train_res['ssim_7t']:.4f} | "
              f"val loss={val_res['loss']:.4f} ssim_hole={val_res['ssim_hole']:.4f} "
              f"ssim_whole={val_res['ssim_whole']:.4f} ssim_brain={val_res['ssim_brain']:.4f}{adv_str}")

        row = [epoch]
        for res in [train_res, val_res]:
            row += [res[k] for k in KEYS]
        row += [f"{ssim_kernel[0]}_{ssim_kernel[1]:.3f}", train_res["n_batches"], val_res["n_batches"]]
        csv_writer.writerow(row)
        csv_file.flush()

        if lpips_spatial_fn is not None and epoch % args.spatial_diag_every == 0:
            spatial_lpips_diagnostic(model, valid_loader, device, lpips_spatial_fn,
                                      args.out_dir, epoch, n_samples=args.spatial_diag_samples,
                                      brain_mask_threshold=args.brain_mask_threshold)

        if clean_csv_writer is not None and epoch % args.clean_eval_every == 0:
            clean_res = run_clean_eval(model, valid_loader, device, ssim_kernel, lpips_fn)
            clean_row = [epoch]
            for region in CLEAN_REGIONS:
                clean_row += [clean_res[region][m] for m in CLEAN_METRIC_NAMES]
            clean_csv_writer.writerow(clean_row)
            clean_csv_file.flush()
            a = clean_res["all"]
            print(f"  [clean eval] epoch {epoch}: SSIM={a['ssim']:.4f} PSNR={a['psnr']:.2f} "
                  f"MAE={a['mae']:.4f} LPIPS={a['perceptual']:.4f} (n={a['n']})")

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "domain_weight_7t": args.domain_weight_7t, "val_loss": val_res["loss"],
                # full CLI args this checkpoint was produced under -- so
                # perceptual_weight, kl_weight, mask_ratio, adv_weight, etc. are
                # always recoverable straight from the .pth file itself, instead
                # of having to dig through shell history or old terminal output.
                "args": vars(args)}
        if discriminator is not None:
            ckpt["discriminator"] = discriminator.state_dict()
            ckpt["disc_optimizer"] = disc_optimizer.state_dict()
        torch.save(ckpt, os.path.join(args.out_dir, "ae_latest.pth"))
        if epoch % 10 == 0:
            torch.save(ckpt, os.path.join(args.out_dir, f"ae_epoch{epoch}.pth"))

        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            torch.save(ckpt, best_val_path)
            print(f"  [new best] val_loss={best_val_loss:.4f} -> saved {best_val_path}")

    csv_file.close()
    if clean_csv_file is not None:
        clean_csv_file.close()


if __name__ == "__main__":
    main()
