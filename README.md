cat > README.md << 'ZZEOF_README'
# Unpaired SPADE-ResShift Latent Diffusion Model for 3T-to-7T MRI Synthesis

Cross-field-strength brain MRI synthesis: translating 3T MRI to 7T-level detail,
resolution and tissue contrast, trained from **unpaired** 3T/7T data.

Built on top of [ALDM](https://github.com/jongdory/ALDM) (Adaptive Latent
Diffusion Model, WACV 2024), replacing its DDPM sampler with a
[ResShift](https://arxiv.org/abs/2307.12348)-style residual-shifting diffusion
process anchored at the SPADE-transformed latent.

## Pipeline

Three frozen/staged training steps, all operating in a 4x16x16 latent space:

1. **Autoencoder (masked VAE)** -- `train_autoencoder.py`
   A 2D convolutional VAE that jointly encodes 3T and 7T images, compressing
   128x128 slices to a 4x16x16 latent. Trained self-supervised with a
   corruption objective (global random Gaussian blur + regional masking) so the
   decoder learns to reconstruct missing/blurred detail -- this is the
   foundation the later diffusion stage relies on to preserve fine structure.

2. **SPADE domain translation** -- `pretrain_spade.py` (+ `preprocess_spade_cache.py`)
   With the autoencoder frozen, a SPADE generator (`ldm/models/normalization_2d.py`)
   is trained with an L1 loss to map a 3T latent to the corresponding 7T
   latent distribution: `z_srctgt = spade(z_src, target_class)`.

3. **ResShift latent diffusion** -- `train_diffusion.py`
   With the autoencoder and SPADE frozen, a UNet-based residual-shifting
   diffusion model (`ldm/models/diffusion/resshift.py`) is trained to correct
   the residual between the SPADE output and the true 7T latent. The SPADE
   output `z_srctgt` is used both as the diffusion's shift anchor (forward
   process target mean) and, concatenated with the raw 3T latent `z_src`, as
   UNet conditioning. Per-channel noise coefficients (kappa) are set from the
   measured per-channel residual statistics (`measure_residual_stats.py`)
   instead of a single global kappa.

Data preprocessing (Hungarian ventricle-DICE pairing + SynthMorph nonlinear
registration + DICE>=0.75 filtering, used to build pseudo-pairs from unpaired
3T/7T subjects) lives in a separate pipeline/repo and is not part of this
codebase -- this repo assumes access to registered, paired 3T/7T volumes.

## Repository structure
.
├── configs/latent-diffusion/
│ ├── mri-3t7t-resshift.yaml # ResShift training config (in use)
│ └── brats-ldm-vq-4.yaml # inherited from base ALDM, unused here
├── ldm/
│ ├── models/
│ │ ├── frozen_ae_spade.py # frozen-AE + SPADE wrapper (encode/decode/spade)
│ │ ├── normalization_2d.py # 2D SPADEGenerator / SPADEResnetBlock
│ │ ├── autoencoder.py
│ │ └── diffusion/
│ │ ├── ddpm.py # base LatentDiffusion.get_input() (SPADE latent concat)
│ │ ├── ddim.py / plms.py # samplers inherited from base ALDM
│ │ └── resshift.py # this project: residual-shifting diffusion
│ ├── modules/ # UNet, attention, distributions, etc. (base ALDM)
│ └── util.py
├── preprocess_spade_cache.py # builds cached 3T/7T latent pairs for SPADE/diffusion training
├── train_autoencoder.py # Stage 1
├── pretrain_spade.py # Stage 2 (supports --resume/--start_epoch)
├── train_diffusion.py # Stage 3
├── evaluate_ae_ceiling.py # diagnostic: AE reconstruction ceiling (skip SPADE+diffusion)
├── evaluate_spade_ceiling.py # diagnostic: SPADE-only quality (skip diffusion)
├── evaluate_diffusion_resshift.py # full-pipeline evaluation
├── measure_residual_stats.py # per-channel residual stats -> per-channel kappa
├── measure_residual_stats_direct.py
├── requirements.txt
└── README.md


`src/clip/` and `src/taming-transformers/` are vendored copies of third-party
repos already installed from source via `requirements.txt`; they are not part
of this project's own code and are excluded from version control (see
`.gitignore`).

## Setup

```bash
conda create -n ldm python=3.10
conda activate ldm
pip install -r requirements.txt
```

External (non-pip) tools used by the data-preprocessing pipeline (not required
to run training/evaluation if you already have paired latents cached):
FSL (`flirt`), SynthMorph, SynthSeg.

## Training

```bash
# Stage 1: autoencoder
python train_autoencoder.py --config configs/latent-diffusion/mri-3t7t-resshift.yaml

# Build the SPADE/diffusion training cache once the autoencoder is frozen
python preprocess_spade_cache.py --config configs/latent-diffusion/mri-3t7t-resshift.yaml \
    --cache_dir <path/to/spade_cache>

# Stage 2: SPADE (supports resuming from a checkpoint)
python pretrain_spade.py --cache_dir <path/to/spade_cache>
# to continue from a checkpoint:
python pretrain_spade.py --resume <path/to/spade_latest.pth> --start_epoch <N> --epochs 100

# Stage 3: ResShift diffusion
python train_diffusion.py --config configs/latent-diffusion/mri-3t7t-resshift.yaml
```

## Evaluation

```bash
# AE reconstruction ceiling (upper bound of the whole pipeline)
python evaluate_ae_ceiling.py --test_dir <path> --subjects <ids...>

# SPADE-only quality (isolates SPADE's contribution, skipping diffusion)
python evaluate_spade_ceiling.py --test_dir <path> --subjects <ids...>

# Full pipeline
python evaluate_diffusion_resshift.py --checkpoint <path> --test_dir <path> --subjects <ids...>
```

## Current performance

Evaluated on 4 held-out test subjects (n=924 slices), brain-masked LPIPS:

| Stage | SSIM | PSNR (dB) | MAE | LPIPS |
|---|---|---|---|---|
| Autoencoder ceiling (7T -> encode -> decode) | 0.9664 | 33.30 | 0.0287 | 0.135 |
| SPADE-only (3T -> encode -> SPADE -> decode, no diffusion) | 0.8096 | 23.80 | 0.0696 | 0.369 |
| Full pipeline (ResShift v2, per-channel kappa, 15-step) | 0.7643 | 21.88 | 0.0844 | 0.4016 |

SPADE-only currently outperforms the full pipeline on every metric -- ongoing
work is focused on closing this gap (see the project's application document /
lab notes for the full list of planned experiments: continued SPADE training,
SPADE cycle-consistency, inference-step ablation, larger UNet, diffusion
condition ablation, perceptual loss on the decoded image).

## Acknowledgments

- [ALDM](https://github.com/jongdory/ALDM) -- Kim et al., *Adaptive Latent
  Diffusion Model for 3D Medical Image to Image Translation*, WACV 2024
  (base SPADE + latent-diffusion architecture)
- [ResShift](https://arxiv.org/abs/2307.12348) -- Yue et al., *Efficient
  Diffusion Model for Image Super-resolution by Residual Shifting*, NeurIPS 2023
- [SPADE](https://arxiv.org/abs/1903.07291) -- Park et al., *Semantic Image
  Synthesis with Spatially-Adaptive Normalization*, CVPR 2019
- [MONAI](https://monai.io/) `AutoencoderKL`, architecture concept from
  [Stable Diffusion](https://arxiv.org/abs/2112.10752) (Rombach et al., CVPR 2022)
- [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers),
  [OpenAI CLIP](https://github.com/openai/CLIP) -- dependencies of the base
  ALDM/latent-diffusion codebase

## License

MIT -- see `LICENSE` (inherited from the base CompVis/latent-diffusion codebase,
Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich).
