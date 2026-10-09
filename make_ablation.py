import argparse
import glob
import os

import torch
from omegaconf import OmegaConf

SOURCE_RUNS = {
    "none": "checkpoints_diffusion_ablation_nocondition",
    "3t": "checkpoints_diffusion_ablation_3t",
    "spade": "checkpoints_diffusion_ablation_spade",
    "all": "checkpoints_diffusion_resshift_perchkappa",
}
EXPECTED_IN_CH = {"none": 4, "3t": 8, "spade": 8, "all": 12}


def source_config(cond):
    cks = sorted(glob.glob(os.path.join(SOURCE_RUNS[cond], "diffusion_epoch*.pth")))
    if not cks:
        raise SystemExit(f"no checkpoint in {SOURCE_RUNS[cond]} to read the {cond} config from")
    return torch.load(cks[0], map_location="cpu", weights_only=False)["args"]["config"]


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flat(v, key))
        else:
            out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spade", default=os.path.expanduser("~/Desktop/LSC/checkpoints_spade_mind/spade_epoch19.pth"))
    ap.add_argument("--kappa", required=True)
    ap.add_argument("--phase", type=int, choices=[1, 2], required=True)
    ap.add_argument("--condition", choices=list(SOURCE_RUNS), default=None)
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 15, 50])
    ap.add_argument("--kmults", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
    ap.add_argument("--out_dir", default="configs/latent-diffusion/abl")
    args = ap.parse_args()
    if not os.path.exists(args.spade):
        raise SystemExit(f"SPADE checkpoint not found: {args.spade}")
    kappa = [float(v) for v in args.kappa.split(",")]

    if args.phase == 1:
        runs = [(f"p1_{c}", c, 15, 1.0) for c in SOURCE_RUNS]
    else:
        if args.condition is None:
            raise SystemExit("--phase 2 needs --condition")
        runs = [(f"p2_{args.condition}_T{T}_k{m:g}", args.condition, T, m)
                for T in args.steps for m in args.kmults if not (T == 15 and m == 1.0)]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"SPADE -> {args.spade}\nkappa x1 = {kappa}\n")
    for name, c, T, m in runs:
        src = source_config(c)
        cfg = OmegaConf.load(src)
        q = cfg.model.params
        in_ch = int(q.unet_config.params.in_channels)
        if in_ch != EXPECTED_IN_CH[c]:
            raise SystemExit(f"{src}: unet in_channels={in_ch}, expected {EXPECTED_IN_CH[c]} for condition '{c}'")
        if len(kappa) != int(q.channels):
            raise SystemExit(f"kappa has {len(kappa)} values, latent has {q.channels} channels")
        before = flat(OmegaConf.to_container(cfg, resolve=True))
        q.resshift_T = T
        q.resshift_kappa = [round(k * m, 4) for k in kappa]
        q.y0_source = "spade"
        q.first_stage_config.params.spade_checkpoint_path = args.spade
        after = flat(OmegaConf.to_container(cfg, resolve=True))
        path = os.path.join(args.out_dir, f"{name}.yaml")
        OmegaConf.save(cfg, path)
        changed = {k: (before.get(k), after[k]) for k in after if before.get(k) != after[k]}
        print(f"{name}  (cond={c}, T={T}, kappa x{m:g}, in_ch={in_ch})  <- {src}")
        for k, (a, b) in changed.items():
            print(f"    {k}: {a} -> {b}")
    print(f"\nsaved {len(runs)} configs to {args.out_dir}/")


if __name__ == "__main__":
    main()
