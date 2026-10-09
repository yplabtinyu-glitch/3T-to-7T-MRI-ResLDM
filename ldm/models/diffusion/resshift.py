"""
LatentResShift: ResShift-style diffusion in this project's latent space,
with the SPADE-transformed 3T latent (z_srctgt) as the shift anchor y0
instead of a raw source latent or pure noise.

Math grounded in (verified against both, formulas are self-consistent at
the boundary conditions t=1 and t=T):
  - Yue et al., "ResShift: Efficient Diffusion Model for Image
    Super-resolution by Residual Shifting", NeurIPS 2023, arXiv:2307.12348
  - Yue et al., "Efficient Diffusion Model for Image Restoration by
    Residual Shifting" (journal extension), arXiv:2403.07319 -- this is
    the version that explicitly confirms the SAME formulas apply in latent
    space: "This does not require any modifications on our model other
    than substituting x_0 and y_0 with their latent codes."
  - Reference implementation: https://github.com/zsyOAOA/ResShift
    (journal branch configs/realsr_swinunet_realesrgan256_journal.yaml
    uses T=4, kappa=2.0, p=0.3; the original NeurIPS paper's main
    experiments use T=15, p=0.3, kappa=2.0 -- we default to T=15 as the
    more thoroughly-validated setting)

Forward process:   q(z_t | z_0, y0) = N(z_t; z_0 + eta_t*(y0-z_0), kappa^2*eta_t*I)
Posterior (sampling): q(z_{t-1} | z_t, z_0, y0) =
    N(z_{t-1}; (eta_{t-1}/eta_t)*z_t + (alpha_t/eta_t)*z_0,
               kappa^2*(eta_{t-1}/eta_t)*alpha_t*I)
Network target: predicts z_0 directly (x0-parameterization), loss = MSE,
no extra timestep weighting (paper reports weighting hurts).

IMPORTANT -- things you (N) must verify before trusting this, not just run:
1. The exact base class name below is a placeholder (`LatentDiffusion`) --
   confirm it against your own ddpm.py (grep "^class.*DDPM\|^class.*Diffusion").
2. `get_input_resshift` slices y0 out of cond['c_concat'][:, 4:8] assuming
   c_concat = concat([z_src, z_srctgt], dim=1) with 4 latent channels each,
   which matches your established in_channels=12=4(noise)+8(cond) fact --
   but re-check this against your actual get_input() before trusting it.
3. kappa=2.0 was tuned by the ResShift authors for their own VQGAN latent
   statistics -- your AE's latents go through your existing scale_factor
   normalization, which should put you in a similar regime, but this is
   an assumption, not a verified fact. Sanity-check by printing
   (z_srctgt - z_tgt).std() on a real batch and comparing to kappa before
   the overnight run.

PER-CHANNEL KAPPA (added 2026-09-26): measured over the FULL training set
(120,705 pairs, measure_residual_stats.py) the residual (z_srctgt - z_tgt)
std per latent channel is NOT uniform: [0.7646, 0.3341, 0.4287, 0.7411]
(pooled/global std=0.5978, which matches the earlier smoke-test 0.5967 --
that number was representative, not a fluke). A single global kappa=0.5
under-covers channels 0/3's true residual spread and over-injects noise on
channels 1/2. `resshift_kappa` now accepts EITHER a scalar (old behavior,
broadcasts to every channel identically) OR a per-channel list/tuple of
length C (e.g. [0.76, 0.33, 0.43, 0.74]).

Design choice: only the NOISE term (kappa^2*eta_t) is made per-channel. The
eta_t/alpha_t schedule itself (the fraction of the source->target bridge
crossed by step t) stays a single shared length-T schedule -- it is a
property of the sampler's time discretization, not of the data; the actual
per-channel magnitude is already carried by (y0-x0) in the mean term. Only
eta_1's numerical-stability floor formula needs *some* kappa value plugged
in, so we use mean(per-channel kappa) as that reference (verified this
barely matters: with kappa in the ~0.3-0.8 range, (0.04/kappa)^2 is always
>> 0.001, so eta_1 hits the 0.001 floor regardless of which channel's kappa
you'd plug in).

kappa is registered as a non-persistent buffer (persistent=False): it is
recomputed fresh from the yaml's resshift_kappa on every __init__ and is
deliberately NOT saved into/expected from checkpoints, so `--resume`
against a checkpoint saved by the old scalar-kappa version still loads
cleanly (state_dict has no "kappa" key to mismatch on either version).
"""
import numpy as np
import torch
import torch.nn.functional as F

from ldm.models.diffusion.ddpm import LatentDiffusion  # <-- VERIFY this class name


class LatentResShift(LatentDiffusion):
    def __init__(self, resshift_T=15, resshift_p=0.3, resshift_kappa=2.0,
                 ablation_no_condition=False,
                 y0_source='spade',
                 lambda_mind=0.0,
                 mind_brain_thr=0.02,
                 lambda_lf=0.0,
                 lf_factor=8,
                 condition_select='all',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_timesteps = resshift_T
        # ablation (老師指示, 2026-09): UNet完全不吃c_concat(z_src+z_srctgt)跟
        # c_crossattn(class label)這兩個額外condition,只看noisy latent本身。
        # y0(z_srctgt)完全不受影響 -- 它是ResShift自己forward/reverse process的
        # 數學錨點,不是"餵給UNet的condition",SPADE的3T->7T轉換照樣算、照樣用。
        # 實作上為什麼不能只靠yaml改conditioning_key:
        #   - conditioning_key設null時,LatentDiffusion.__init__有段backward-
        #     compat邏輯「if conditioning_key is None: conditioning_key =
        #     'concat' if concat_mode else 'crossattn'」,會把null偷偷改成
        #     'concat',不是真的None,送進DiffusionWrapper後conditioning_key
        #     根本不是None。(2026-10-01, 實際跑check_ablation_config_shapes.py
        #     炸出來才發現這個,之前光讀code沒注意到這段)
        #   - 'concat'分支寫的是`torch.cat([x] + c_concat, dim=1)`,預期
        #     c_concat是list,但我們的get_input()給的是tensor,一樣會炸
        #     (TypeError: can only concatenate list (not "Tensor") to list)
        #   - 'crossattn'分支也預期c_crossattn是list、且是用在真的cross-attention
        #     (context_dim需對上),跟我們的class label(純一個int tensor)完全
        #     對不上,這條路也走不通
        # 所以保留conditioning_key: hybrid不變(這是目前唯一跟get_input()的
        # tensor格式相容的分支),改成讓c_concat變成0-channel tensor(等於沒接
        # 任何東西進去,torch.cat後channel數不變)、c_crossattn直接設None
        # (UNetModel.forward有 `assert (y is not None) == (self.num_classes
        # is not None)`,所以yaml那邊num_classes也要拿掉，這點跟之前講的一樣)。
        self.ablation_no_condition = ablation_no_condition
        assert y0_source in ('spade', '3t'), y0_source
        self.y0_source = y0_source
        self.lambda_mind = float(lambda_mind)
        self.mind_brain_thr = float(mind_brain_thr)
        self.lambda_lf = float(lambda_lf)
        self.lf_factor = int(lf_factor)
        if self.lambda_lf > 0:
            print(f'[LF] low-frequency anchor to SPADE on: lambda_lf={self.lambda_lf}, pool={self.lf_factor}')
        if self.lambda_mind > 0:
            print(f'[MIND] structure loss on: lambda_mind={self.lambda_mind}')
        assert condition_select in ('all', '3t', 'spade'), condition_select
        self.condition_select = condition_select
        # resshift_kappa may arrive as a plain float (old behavior) or as an
        # OmegaConf ListConfig (from a yaml list like [0.76, 0.33, 0.43,
        # 0.74]) -- ListConfig isn't a type torch.as_tensor recognizes
        # directly, so normalize any iterable to a plain list first.
        kappa_val = list(resshift_kappa) if hasattr(resshift_kappa, "__iter__") else resshift_kappa
        kappa_t = torch.as_tensor(kappa_val, dtype=torch.float32)
        # non-persistent: NOT part of state_dict, always rebuilt from the
        # yaml's resshift_kappa (see PER-CHANNEL KAPPA note above) -- keeps
        # --resume compatible with checkpoints from the old scalar-kappa code.
        self.register_buffer('kappa', kappa_t, persistent=False)
        kappa_ref = float(kappa_t.float().mean().item())
        self._register_resshift_schedule(resshift_T, resshift_p, kappa_ref)

    def _kappa_view(self, ndim):
        """Reshape self.kappa for broadcasting against a (B, C, H, W, ...)
        tensor with `ndim` dims. A 0-dim (scalar) kappa broadcasts as-is
        (old behavior); a 1-D per-channel kappa of length C is reshaped to
        (1, C, 1, ..., 1)."""
        if self.kappa.ndim == 0:
            return self.kappa
        view_shape = (1, self.kappa.shape[0]) + (1,) * (ndim - 2)
        return self.kappa.view(*view_shape)

    def _register_resshift_schedule(self, T, p, kappa):
        eta_1 = min((0.04 / kappa) ** 2, 0.001)
        eta_T = 0.999
        b0 = np.exp(np.log(eta_T / eta_1) / (2 * (T - 1)))
        sqrt_eta_1 = np.sqrt(eta_1)

        etas = np.zeros(T, dtype=np.float64)
        for t in range(1, T + 1):
            beta_t = ((t - 1) / (T - 1)) ** p * (T - 1)
            sqrt_eta_t = sqrt_eta_1 * (b0 ** beta_t)
            etas[t - 1] = sqrt_eta_t ** 2
        etas[-1] = eta_T  # enforce exact boundary condition
        etas_prev = np.concatenate([[0.0], etas[:-1]])
        alphas = etas - etas_prev

        assert np.all(alphas > 0), "schedule produced a non-increasing eta_t -- check T/p/kappa"

        to_torch = lambda a: torch.tensor(a, dtype=torch.float32)
        self.register_buffer('resshift_etas', to_torch(etas))
        self.register_buffer('resshift_etas_prev', to_torch(etas_prev))
        self.register_buffer('resshift_alphas', to_torch(alphas))

    @staticmethod
    def _extract(arr, t, x_shape):
        out = arr.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    # ---- forward process ----
    def q_sample_resshift(self, x_start, y0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        eta_t = self._extract(self.resshift_etas, t, x_start.shape)
        mean = x_start + eta_t * (y0 - x_start)
        kappa = self._kappa_view(x_start.ndim)
        std = kappa * torch.sqrt(eta_t)
        return mean + std * noise

    # ---- get_input wrapper: reuses the parent's AE+SPADE pipeline, just
    # also hands back y0 (the SPADE-transformed latent) as a separate tensor
    # instead of only inside the concatenated cond dict ----
    def get_input_resshift(self, batch, k, *args, **kwargs):
        x_start, cond = super().get_input(batch, k, *args, **kwargs)
        y0 = (cond['c_concat'][:, 0:4, ...] if self.y0_source == '3t' else cond['c_concat'][:, 4:8, ...]).clone()  # 0:4 raw 3T latent, 4:8 SPADE latent
        self._z3t_for_mind = cond['c_concat'][:, 0:4, ...].detach()  # 3T latent (scaled), for the MIND loss
        self._zsp_for_lf = cond['c_concat'][:, 4:8, ...].detach()  # SPADE latent (scaled), low-frequency anchor
        if self.ablation_no_condition:
            b = x_start.shape[0]
            zero_concat = x_start.new_zeros((b, 0, *x_start.shape[2:]))
            cond = {'c_concat': zero_concat, 'c_crossattn': None}
        elif self.condition_select == '3t':
            cond = {'c_concat': cond['c_concat'][:, 0:4, ...], 'c_crossattn': cond['c_crossattn']}  # raw 3T latent only
        elif self.condition_select == 'spade':
            cond = {'c_concat': cond['c_concat'][:, 4:8, ...], 'c_crossattn': cond['c_crossattn']}  # SPADE latent only
        return x_start, y0, cond

    # ---- training loss ----
    def p_losses_resshift(self, x_start, y0, cond, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        z_t = self.q_sample_resshift(x_start, y0, t, noise)
        x0_pred = self.apply_model(z_t, t, cond)
        loss = F.mse_loss(x0_pred, x_start, reduction='none')
        loss = loss.mean(dim=list(range(1, loss.ndim)))
        loss_dict = {'loss_simple': loss.mean().detach()}
        total = loss.mean()
        if self.lambda_mind > 0 or self.lambda_lf > 0:
            fsm = self.first_stage_model
            dec = getattr(type(fsm).decode, "__wrapped__", None)
            if dec is None:
                raise RuntimeError("first_stage_model.decode has no __wrapped__ -- cannot backprop through the decoder")
            img_pred = (dec(fsm, x0_pred / self.scale_factor).float() + 1) / 2
            with torch.no_grad():
                img_3t = (fsm.decode(self._z3t_for_mind / self.scale_factor).float().clamp(-1, 1) + 1) / 2
            if torch.is_grad_enabled() and not img_pred.requires_grad:
                raise RuntimeError("decoded x0_pred has no grad -- MIND loss would not train the UNet")

            def _mind(x, eps=1e-3):
                H, W = x.shape[-2:]
                p = F.pad(x, (1, 1, 1, 1), mode="replicate")
                D = torch.cat([F.avg_pool2d((x - p[:, :, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]) ** 2, 3, stride=1, padding=1)
                               for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))], 1)
                V = D.mean(1, keepdim=True).clamp_min(eps)
                M = torch.exp(-D / V)
                return M / M.amax(1, keepdim=True).clamp_min(eps)

            bm = (img_3t > self.mind_brain_thr).float()
            l_mind = ((_mind(img_pred) - _mind(img_3t)).abs().mean(1, keepdim=True) * bm).sum() / bm.sum().clamp_min(1.0)
            total = total + self.lambda_mind * l_mind
            loss_dict['loss_mind'] = l_mind.detach()
            if self.lambda_lf > 0:
                with torch.no_grad():
                    img_sp = (fsm.decode(self._zsp_for_lf / self.scale_factor).float().clamp(-1, 1) + 1) / 2
                f_ = self.lf_factor
                bml = (F.avg_pool2d(bm, f_) > 0.5).float()
                l_lf = ((F.avg_pool2d(img_pred, f_) - F.avg_pool2d(img_sp, f_)).abs() * bml).sum() / bml.sum().clamp_min(1.0)
                total = total + self.lambda_lf * l_lf
                loss_dict['loss_lf'] = l_lf.detach()
        return total, loss_dict

    def shared_step(self, batch):
        # overrides LatentDiffusion.shared_step BY NAME (not a new method) so
        # train_diffusion.py's existing `loss, _ = model.shared_step(batch)`
        # call works completely unmodified.
        x_start, y0, cond = self.get_input_resshift(batch, self.first_stage_key)
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],),
                           device=x_start.device).long()
        loss, loss_dict = self.p_losses_resshift(x_start, y0, cond, t)
        return loss, loss_dict

    # ---- sampling (reverse process) ----
    @torch.no_grad()
    def p_sample_loop_resshift(self, y0, cond, shape, noise=None, verbose=False):
        device = y0.device
        b = shape[0]
        eta_T = self.resshift_etas[-1]
        kappa = self._kappa_view(len(shape))
        if noise is None:
            noise = torch.randn(shape, device=device)
        z_t = y0 + kappa * torch.sqrt(eta_T) * noise

        r = range(self.num_timesteps - 1, -1, -1)
        if verbose:
            from tqdm import tqdm
            r = tqdm(r, desc='ResShift sampling', total=self.num_timesteps)

        for i in r:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x0_pred = self.apply_model(z_t, t, cond)
            if i > 0:
                eta_t = self.resshift_etas[i]
                eta_prev = self.resshift_etas_prev[i]
                alpha_t = self.resshift_alphas[i]
                mean = (eta_prev / eta_t) * z_t + (alpha_t / eta_t) * x0_pred
                var = (kappa ** 2) * (eta_prev / eta_t) * alpha_t
                z_t = mean + torch.sqrt(var) * torch.randn_like(z_t)
            else:
                z_t = x0_pred
        return z_t
