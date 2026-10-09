# Unpaired SPADE-ResShift Latent Diffusion Model for 3T-to-7T MRI Synthesis

Cross-field-strength brain MRI synthesis: translating 3T MRI to 7T-level detail,
resolution and tissue contrast, trained from **unpaired** 3T/7T data.

Built on top of [ALDM](https://github.com/jongdory/ALDM) (Adaptive Latent
Diffusion Model, WACV 2024), replacing its DDPM sampler with a
[ResShift](https://arxiv.org/abs/2307.12348)-style residual-shifting diffusion
process anchored at the SPADE-translated latent.

## Pipeline

Three staged training steps, all operating in a 4x16x16 latent space
(128x128 slices):

1. **Autoencoder** -- `train_autoencoder.py`
   A 2D convolutional VAE (MONAI `AutoencoderKL`) that jointly encodes 3T and
   7T images. Trained self-supervised with a corruption objective (global
   random Gaussian blur + regional masking). Frozen afterwards; `encode`
   returns the posterior mean `z_mu` (deterministic).

2. **SPADE domain translation** -- `pretrain_spade.py`
   With the autoencoder frozen, a SPADE generator (`ldm/models/normalization_2d.py`)
   maps a 3T latent towards the 7T latent distribution:
   `z_spade = SPADE(z_3T, class=7T)`.
   Loss terms (all optional, set by flags):
   - `--lambda_latent` latent L1 to the 7T latent (original objective)
   - `--lambda_mind` **MIND structure loss** between the decoded SPADE output
     and the decoded 3T input (same subject, so anatomy must be preserved)
   - `--lambda_l1 / --lambda_ssim / --lambda_lpips` decoded-image losses vs. 7T
   - `--lambda_back / --lambda_cyc` reverse (SPADE(z_7T, class=3T) -> z_3T) / cycle constraints (latent)
   Current SPADE: latent L1 + MIND (`checkpoints_spade_mind/`).

3. **ResShift latent diffusion** -- `train_diffusion.py`
   With autoencoder and SPADE frozen, a UNet (13M params,
   `ldm/models/diffusion/resshift.py`) learns to move from the SPADE latent
   to the real 7T latent:

```
   forward:  z_t = x0 + eta_t (y0 - x0) + kappa * sqrt(eta_t) * eps
             x0 = z_7T,  y0 = z_spade
   loss   =  MSE(x0_hat, x0)
          +  lambda_mind * MIND( D(x0_hat), D(z_3T) )          # structure: same-subject 3T
          +  lambda_lf   * L1( LP(D(x0_hat)), LP(D(z_spade)) )  # low-frequency contrast: SPADE
```
   `D` = frozen decoder (gradients pass through), `LP` = 8x8 average pooling,
   both image losses are computed inside the brain mask only.
   Config options in `model.params`: `resshift_T`, `resshift_kappa`
   (per-channel), `y0_source`, `condition_select` (`3t` / `spade` / `all` --
   which latent is concatenated to the UNet input; `ablation_no_condition: true`
   removes the condition), `lambda_mind`, `lambda_lf`, `lf_factor`.

Data preprocessing (ventricle-DICE pairing + SynthMorph nonlinear registration
+ DICE filtering, used to build pseudo-pairs from unpaired 3T/7T subjects)
lives in a separate pipeline and is not part of this codebase -- this repo
assumes registered 3T/7T slices.

### Why MIND for the structure loss

[MIND](https://doi.org/10.1016/j.media.2012.05.006) (Heinrich et al.,
*Medical Image Analysis* 2012) describes each pixel by its self-similarity to
its neighbours, so it compares *structure* across modalities/contrasts rather
than raw intensity. Here a 2D, 4-neighbour version is used. Training is
cross-subject (3T and 7T from different subjects), so a pure latent MSE lets
the diffusion hallucinate sulci; anchoring MIND to the same-subject 3T stops
that. MIND is not fully contrast-invariant (non-linear tissue-contrast changes
still move it), which is why intensity/contrast is anchored separately to the
SPADE output through the low-frequency term.

## Evaluation protocol

Training and test data come from different datasets, so test intensities are
**histogram-matched** to the training distribution (Nyul-Udupa piecewise-linear
landmark matching, percentiles 1, 10, ..., 90, 99, landmarks learned from the
training set -> `hist_ref_train.npz`, built by `hist_match.py build`).

- `--hist_match both` (main protocol): test 3T and test 7T are both mapped to
  the training 3T / 7T references
- `--hist_match 3t`: only the network input is matched
- `--hist_match none`: raw intensities

Metrics on 4 held-out subjects (924 slices): SSIM, PSNR, MAE, MAE inside the
brain mask, LPIPS.

## Training acceleration (latent cache)

The autoencoder and SPADE are frozen during diffusion training, so their
outputs are precomputed once:

```bash
python cache_latents.py ...            # {train,val}_{zsrc,ztgt}.npy  (encoder outputs)
python verify_latent_cache.py ...      # cache vs. live encoding (equal within TF32 jitter)
python cache_spade_latents.py --config <cfg> --bench   # {train,val}_zspade.npy (SPADE outputs)
python train_diffusion.py --config <cfg> --latent_cache ~/Desktop/data/latent_cache
```

`meta.json` in the cache records which SPADE checkpoint produced
`zspade.npy`; training refuses to start if the config points to a different
SPADE (`NO_SPADE_CACHE=1` falls back to live SPADE).

## Ablations

```bash
python kappa_from_cache.py                                  # per-channel kappa from cached residuals
python make_ablation.py --phase 1 --kappa <k0,k1,k2,k3>     # condition: none / 3t / spade / all
python make_ablation.py --phase 2 --condition 3t --kappa <k0,k1,k2,k3>   # T x kappa grid
EPOCHS=60 ./run_ablation.sh p2_3t                           # train (early stopping) + test + collect
```

Configs go to `configs/latent-diffusion/abl/`, named
`p<phase>_<condition>[_T<steps>_k<kappa multiplier>]`. Each run is tested with
its best-validation checkpoint (`--hist_match both`) and summarised in
`ablation_results.csv`.

## Repository structure

```
.
├── configs/latent-diffusion/
│   ├── mri-3t7t-resshift.yaml      # base ResShift config
│   └── abl/                        # generated ablation configs
├── ldm/
│   ├── models/
│   │   ├── frozen_ae_spade.py      # frozen AE + SPADE wrapper
│   │   ├── normalization_2d.py     # 2D SPADE generator
│   │   └── diffusion/
│   │       ├── ddpm.py             # LatentDiffusion.get_input (latent concat)
│   │       └── resshift.py         # residual-shifting diffusion + MIND / LF losses
│   └── modules/                    # UNet etc. (base ALDM)
├── train_autoencoder.py            # stage 1
├── pretrain_spade.py               # stage 2
├── train_diffusion.py              # stage 3 (--latent_cache)
├── cache_latents.py / verify_latent_cache.py / cache_spade_latents.py
├── kappa_from_cache.py / make_ablation.py / run_ablation.sh
├── evaluate_diffusion_resshift_v2.py   # full-pipeline test (--hist_match, --spade_ckpt)
├── eval_all_checkpoints.py         # metrics vs. epoch
├── hist_match.py                   # build / plot histogram-matching reference
├── compare_spade.py / spade_detail.py / watch_eval_spade.py   # SPADE evaluation
├── mind_distance.py                # pairwise MIND structure distance (3T / SPADE / diffusion / 7T)
├── visualize_slice_compare.py      # side-by-side figures
├── evaluate_spade_ceiling_valset.py   # SPADE-only quality over checkpoints (val split)
└── evaluate_ae_ceiling.py / evaluate_spade_ceiling.py          # upper-bound diagnostics
```

## Results

Diffusion ablations (4 test subjects, `--hist_match both`, best-validation
checkpoint), generated from `ablation_results.csv`:

| run | condition | T | kappa_mult | best_epoch | ssim | psnr | mae | mae_brain | lpips |
|---|---|---|---|---|---|---|---|---|---|
| p1_3t | 3t | 15 | 1.0000 | 40 | 0.7697 | 20.5348 | 0.0410 | 0.1580 | 0.4048 |
| p2_3t_T4_k1 | 3t | 4 | 1.0000 | 35 | 0.7830 | 22.2278 | 0.0384 | 0.1470 | 0.3883 |
| p2_3t_T50_k1 | 3t | 50 | 1.0000 | 35 | 0.7736 | 23.4646 | 0.0410 | 0.1627 | 0.4074 |
| p3_3tmind_T4_k1 | 3tmind | 4 | 1.0000 | 22 | 0.8291 | 24.6727 | 0.0381 | 0.1404 | 0.3174 |

Earlier version (Sep 2026: no histogram matching, old latent-L1 SPADE, 15
steps), kept for reference:

| Stage | SSIM | PSNR (dB) | MAE | LPIPS |
|---|---|---|---|---|
| Autoencoder ceiling (7T -> encode -> decode) | 0.9664 | 33.30 | 0.0287 | 0.135 |
| SPADE-only (old SPADE) | 0.8096 | 23.80 | 0.0696 | 0.369 |
| Full pipeline (ResShift v2, per-channel kappa) | 0.7643 | 21.88 | 0.0844 | 0.4016 |

## Roadmap

- Structure-preserving diffusion training (MIND to 3T + low-frequency anchor to SPADE)
- Condition ablation (none / 3T / SPADE / all) under the current SPADE
- Faster inference: fewer steps, one-step distillation of ResShift (SinSR-style)
- Three-subject validation split
- 2.5D / 3D extension (inter-slice consistency)

## Setup

```bash
conda create -n ldm python=3.10
conda activate ldm
pip install -r requirements.txt
```

## Acknowledgments

- [ALDM](https://github.com/jongdory/ALDM) -- Kim et al., *Adaptive Latent
  Diffusion Model for 3D Medical Image to Image Translation*, WACV 2024
- [ResShift](https://arxiv.org/abs/2307.12348) -- Yue et al., NeurIPS 2023
- [SPADE](https://arxiv.org/abs/1903.07291) -- Park et al., CVPR 2019
- [MIND](https://doi.org/10.1016/j.media.2012.05.006) -- Heinrich et al., *Medical Image Analysis* 2012
- Nyul, Udupa & Zhang, *New variants of a method of MRI scale standardization*, IEEE TMI 2000
- [MONAI](https://monai.io/) `AutoencoderKL`; [Stable Diffusion](https://arxiv.org/abs/2112.10752) (Rombach et al., CVPR 2022)
- [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers),
  [OpenAI CLIP](https://github.com/openai/CLIP) -- dependencies of the base codebase

## License

MIT -- see `LICENSE` (inherited from CompVis/latent-diffusion,
Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich).
