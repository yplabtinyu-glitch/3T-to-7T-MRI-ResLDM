import os
import torch
from monai.networks.nets import AutoencoderKL

ckpt_path = os.path.expanduser("~/Desktop/LSC/checkpoints_ae/ae_epoch690.pth")
ckpt = torch.load(ckpt_path, map_location="cpu")
args = ckpt["args"]

model = AutoencoderKL(
    spatial_dims=2, in_channels=1, out_channels=1,
    channels=(32, 64, 128, 128),
    latent_channels=args["latent_channels"],
    num_res_blocks=2,
    attention_levels=(False, False, False, True),
)
model.load_state_dict(ckpt["model"])
model.eval()

dummy = torch.randn(1, 1, args["img_size"], args["img_size"])
with torch.no_grad():
    z_mu, z_log_var = model.encode(dummy)

print(f"input {tuple(dummy.shape)} -> latent {tuple(z_mu.shape)}")
