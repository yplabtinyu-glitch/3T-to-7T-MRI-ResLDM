import os
import torch
import torch.nn as nn
from monai.networks.nets import AutoencoderKL

from ldm.models.normalization_2d import SPADEGenerator


class FrozenAEWithSPADE(nn.Module):
    """
    Drop-in replacement for VQModel as the `first_stage_model` inside
    ddpm.py's LatentDiffusion. Wraps our own pretrained MONAI AutoencoderKL
    (frozen) plus a SPADEGenerator (domain-conditioning module).

    ddpm.py calls this object exactly like it calls VQModel:
      - z, _, _ = self.first_stage_model.encode(x)        (get_input / encode_first_stage)
      - self.first_stage_model.decode(z)                  (decode_first_stage)
      - self.first_stage_model.spade(z_src, y)             (get_input, domain conditioning)

    Note: ddpm.py's instantiate_first_stage() sets requires_grad=False on
    EVERY parameter of first_stage_model after construction (encoder,
    decoder, AND spade) -- the diffusion-stage training in ddpm.py only
    ever trains the UNet (self.model), never first_stage_model. So a
    freshly / randomly initialized SPADEGenerator is fine here for a
    forward/backward timing benchmark; it is not being trained in this
    stage regardless of how it was initialized.
    """

    def __init__(self, ae_checkpoint_path, num_classes=2, ae_channels=(32, 64, 128, 128),
                 spade_checkpoint_path=None):
        super().__init__()
        ckpt_path = os.path.expanduser(ae_checkpoint_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = ckpt["args"]
        self.latent_channels = args["latent_channels"]

        self.ae = AutoencoderKL(
            spatial_dims=2, in_channels=1, out_channels=1,
            channels=ae_channels,
            latent_channels=self.latent_channels,
            num_res_blocks=2,
            attention_levels=(False, False, False, True),
        )
        self.ae.load_state_dict(ckpt["model"])
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

        self.spade = SPADEGenerator(num_classes=num_classes, z_dim=self.latent_channels)

        if spade_checkpoint_path is not None:
            spade_path = os.path.expanduser(spade_checkpoint_path)
            self.spade.load_state_dict(torch.load(spade_path, map_location="cpu"))
            print(f"[FrozenAEWithSPADE] loaded pretrained SPADE weights from {spade_path}")

    @torch.no_grad()
    def encode(self, x):
        z_mu, _ = self.ae.encode(x)
        return z_mu, None, None

    @torch.no_grad()
    def decode(self, z):
        return self.ae.decode(z)
