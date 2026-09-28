# test.py
import torch
from omegaconf import OmegaConf
from ldm.models.diffusion.resshift import LatentResShift

config = OmegaConf.load("/home/ubuntu/Desktop/ALDM/LDM/configs/latent-diffusion/mri-3t7t-ldm.yaml")
model = LatentResShift(resshift_T=15, resshift_p=0.3, resshift_kappa=0.5,
                        **config.model.params).to('cuda')
model.eval()

batch = {
    "source": torch.randn(2, 1, 128, 128).clamp(-1, 1).to('cuda'),
    "target": torch.randn(2, 1, 128, 128).clamp(-1, 1).to('cuda'),
    "target_class": torch.randint(0, 2, (2,)).to('cuda'),
}

x_start, y0, cond = model.get_input_resshift(batch, model.first_stage_key)
print("x_start shape:", x_start.shape, "std:", x_start.std().item())
print("y0 shape:", y0.shape, "std:", y0.std().item())
print("residual (y0-x_start) std:", (y0 - x_start).std().item(), "  <- 拿這個跟kappa比")

loss, loss_dict = model.shared_step_resshift(batch)
print("loss:", loss.item())  # 要是有限數字,不是nan/inf
