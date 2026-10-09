import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from evaluate_diffusion_resshift_v2 import iter_test_slices, load_autoencoder_utils


def mind(x, eps=1e-3):
    H, W = x.shape[-2:]
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    D = torch.cat([F.avg_pool2d((x - p[:, :, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]) ** 2, 3, stride=1, padding=1)
                   for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))], 1)
    V = D.mean(1, keepdim=True).clamp_min(eps)
    M = torch.exp(-D / V)
    return M / M.amax(1, keepdim=True).clamp_min(eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--test_dir", default=os.path.expanduser("~/Desktop/test"))
    ap.add_argument("--subjects", nargs="+", default=["212730", "212731", "216393", "216394"])
    ap.add_argument("--hist_match", choices=["none", "3t", "both"], default="both")
    ap.add_argument("--hist_ref", default="hist_ref_train.npz")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--autoencoder_script_dir", default=".")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_autoencoder_utils(args.autoencoder_script_dir)
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model).to(device).eval()
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])
    print("SPADE:", config.model.params.first_stage_config.params.spade_checkpoint_path)
    HM = None
    if args.hist_match != "none":
        r = np.load(args.hist_ref)
        HM = {"lm_3t": r["lm_3t"], "lm_7t": r["lm_7t"], "thr": float(r["thr"]), "mode": args.hist_match}

    pairs = [("3T", "7T"), ("SPADE", "7T"), ("SPADE", "3T")]
    if args.ckpt:
        pairs += [("diff", "7T"), ("diff", "3T"), ("diff", "SPADE")]
    pairs += [("7T_rec", "7T")]
    tot = {p: 0.0 for p in pairs}
    vox = 0
    dec = lambda z: (model.decode_first_stage(z).clamp(-1, 1).float() + 1) / 2

    def run(src, tgt):
        nonlocal vox
        xs, xt = torch.cat(src).to(device), torch.cat(tgt).to(device)
        batch = {"source": xs, "target": xt, "target_class": torch.ones(len(xs), dtype=torch.long)}
        with torch.no_grad():
            x0, cond = model.get_input(batch, model.first_stage_key)
            c = cond["c_concat"]
            im = {"3T": dec(c[:, 0:4]), "SPADE": dec(c[:, 4:8]), "7T_rec": dec(x0),
                  "7T": (xt.float().clamp(-1, 1) + 1) / 2}
            if args.ckpt:
                torch.manual_seed(args.seed)
                _, y0, cr = model.get_input_resshift(batch, model.first_stage_key)
                im["diff"] = dec(model.p_sample_loop_resshift(y0, cr, y0.shape, verbose=False))
            M = {k: mind(v) for k, v in im.items()}
            bm = (im["7T"] > 0.02).float()
            for a, b in pairs:
                tot[(a, b)] += ((M[a] - M[b]).abs().mean(1, keepdim=True) * bm).sum().item()
            vox += bm.sum().item()

    for s in args.subjects:
        sb, tb = [], []
        for a, b in iter_test_slices(args.test_dir, s, 128, hm=HM):
            sb.append(a); tb.append(b)
            if len(sb) == args.batch_size:
                run(sb, tb); sb, tb = [], []
        if sb:
            run(sb, tb)
        print(f"  done {s}")

    base = tot[("3T", "7T")] / vox
    print(f"\nMIND distance (brain mean, hist_match={args.hist_match}; lower = more similar structure)")
    for a, b in pairs:
        d = tot[(a, b)] / vox
        print(f"  {a:>7s} vs {b:<6s}: {d:.4f}   ({d / base:5.2f} x the 3T-vs-7T baseline)")


if __name__ == "__main__":
    main()
