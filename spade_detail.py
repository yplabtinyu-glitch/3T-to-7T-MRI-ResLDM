import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from evaluate_diffusion_resshift_v2 import load_and_normalize_volume, slice_to_tensor
from hist_match import match_volume
from compare_spade import load_spade_weights

COLORS = ["#d1495b", "#2e86ab", "#3bb273", "#8e5ea2", "#e8a33d"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/latent-diffusion/mri-3t7t-resshift-ablation-nocondition.yaml")
    ap.add_argument("--spade", action="append", required=True, help='"label=path", repeatable (first = reference for the diff)')
    ap.add_argument("--subject", required=True)
    ap.add_argument("--slices", type=int, nargs="+", required=True)
    ap.add_argument("--test_dir", default=os.path.expanduser("~/Desktop/test"))
    ap.add_argument("--hist_match", choices=["none", "3t", "both"], default="both")
    ap.add_argument("--hist_ref", default="hist_ref_train.npz")
    ap.add_argument("--rot", type=int, default=0)
    ap.add_argument("--crop", action="append", default=None, help='"x,y" centre in displayed image, up to 2')
    ap.add_argument("--crop_size", type=int, default=32)
    ap.add_argument("--profile_row", type=int, default=None)
    ap.add_argument("--target_class", type=int, default=1)
    ap.add_argument("--autoencoder_script_dir", default=".")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.load(args.config)
    ae_ckpt = os.path.expanduser(config.model.params.first_stage_config.params.ae_checkpoint_path)
    img_size = torch.load(ae_ckpt, map_location="cpu", weights_only=False).get("args", {}).get("img_size", 128)
    model = instantiate_from_config(config.model).to(device).eval()
    spades = [s.split("=", 1) for s in args.spade]

    src = load_and_normalize_volume(os.path.join(args.test_dir, "reg_3T", f"{args.subject}.nii.gz"))
    tgt = load_and_normalize_volume(os.path.join(args.test_dir, "7T", f"{args.subject}.nii.gz"))
    if args.hist_match != "none":
        r = np.load(args.hist_ref)
        src, _ = match_volume(src, r["lm_3t"], float(r["thr"]))
        if args.hist_match == "both":
            tgt, _ = match_volume(tgt, r["lm_7t"], float(r["thr"]))

    xs = torch.cat([slice_to_tensor(src[:, :, i], img_size) for i in args.slices])
    xt = torch.cat([slice_to_tensor(tgt[:, :, i], img_size) for i in args.slices])
    batch = {"source": xs.to(device), "target": xt.to(device),
             "target_class": torch.full((len(args.slices),), args.target_class, dtype=torch.long)}
    dec = lambda z: ((model.decode_first_stage(z).clamp(-1, 1).float() + 1) / 2).cpu().numpy()[:, 0]
    imgs = {}
    with torch.no_grad():
        for lab, path in spades:
            load_spade_weights(model, path)
            x0, cond = model.get_input(batch, model.first_stage_key)
            imgs[lab] = dec(cond["c_concat"][:, 4:8])
        imgs["3T (AE recon)"] = dec(cond["c_concat"][:, 0:4])
        imgs["7T (AE recon)"] = dec(x0)
    imgs["3T input"] = ((xs.clamp(-1, 1) + 1) / 2).numpy()[:, 0]
    imgs["7T GT"] = ((xt.clamp(-1, 1) + 1) / 2).numpy()[:, 0]

    order = ["3T input", "3T (AE recon)"] + [l for l, _ in spades] + ["7T (AE recon)", "7T GT"]
    labs = [l for l, _ in spades]

    for r_i, s in enumerate(args.slices):
        D = {k: np.rot90(v[r_i], k=args.rot) for k, v in imgs.items()}
        H, W = D["7T GT"].shape
        cs = args.crop_size
        brain = D["7T GT"] > 0.02
        ys, xs_ = np.nonzero(brain)
        cy = int(ys.mean()) if len(ys) else H // 2
        if args.crop:
            centers = [tuple(int(v) for v in c.split(",")) for c in args.crop][:2]
        else:
            x_left = int(xs_[ys == cy].min()) if (ys == cy).any() else W // 4
            centers = [(x_left + cs // 2 - 2, cy), (W // 2, cy)]
        boxes = [(int(np.clip(x - cs // 2, 0, W - cs)), int(np.clip(y - cs // 2, 0, H - cs))) for x, y in centers]
        prow = args.profile_row if args.profile_row is not None else cy

        nc = len(order)
        fig = plt.figure(figsize=(2.2 * nc, 13.5), dpi=150)
        gs = fig.add_gridspec(5, nc, height_ratios=[1, 1, 1, 1, 0.9], hspace=0.25, wspace=0.05)

        for j, k in enumerate(order):
            a = fig.add_subplot(gs[0, j])
            a.imshow(D[k], cmap="gray", vmin=0, vmax=1)
            for bi, (bx, by) in enumerate(boxes):
                a.add_patch(Rectangle((bx - .5, by - .5), cs, cs, fill=False, ec=["#ffd400", "#00e5ff"][bi], lw=1))
            a.axhline(prow, color="#ff4fa0", lw=0.6, alpha=0.8)
            a.set_title(k, fontsize=8.5); a.set_xticks([]); a.set_yticks([])

        diffs = [(f"|{l} - 3T recon|", np.abs(D[l] - D["3T (AE recon)"])) for l in labs]
        if len(labs) >= 2:
            diffs.append((f"|{labs[-1]} - {labs[0]}|", np.abs(D[labs[-1]] - D[labs[0]])))
        im = None
        for j, (t, d) in enumerate(diffs[:nc]):
            a = fig.add_subplot(gs[1, j])
            im = a.imshow(d, cmap="magma", vmin=0, vmax=0.3)
            a.set_title(t, fontsize=7.5); a.set_xticks([]); a.set_yticks([])
            a.set_xlabel(f"mean in brain {d[brain].mean():.3f}", fontsize=7)
        if im is not None:
            cax = fig.add_subplot(gs[1, len(diffs[:nc])])
            cax.set_axis_off()
            fig.colorbar(im, ax=cax, fraction=0.5, label="|difference|")

        for bi, (bx, by) in enumerate(boxes):
            for j, k in enumerate(order):
                a = fig.add_subplot(gs[2 + bi, j])
                a.imshow(D[k][by:by + cs, bx:bx + cs], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_edgecolor(["#ffd400", "#00e5ff"][bi]); sp.set_linewidth(1.5)
                if j == 0:
                    a.set_ylabel(f"zoom {bi + 1}", fontsize=8)

        a = fig.add_subplot(gs[4, :])
        row_brain = np.nonzero(brain[prow])[0]
        x0_, x1_ = (row_brain.min(), row_brain.max() + 1) if len(row_brain) else (0, W)
        xx = np.arange(x0_, x1_)
        a.plot(xx, D["7T GT"][prow, x0_:x1_], color="k", lw=2.2, label="7T GT")
        a.plot(xx, D["3T (AE recon)"][prow, x0_:x1_], color="#999999", lw=1.2, ls="--", label="3T (AE recon)")
        for ci, l in enumerate(labs):
            a.plot(xx, D[l][prow, x0_:x1_], color=COLORS[ci % len(COLORS)], lw=1.4, label=l)
        a.set_xlabel(f"x (pixel) along row {prow}"); a.set_ylabel("intensity [0,1]")
        a.legend(fontsize=7, frameon=False, ncol=len(labs) + 2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(alpha=0.25)

        fig.suptitle(f"subject {args.subject}, slice {s}  (hist_match={args.hist_match})", fontsize=11, y=0.995)
        out = f"spade_detail_{args.subject}_s{s}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}  (crops at {boxes}, size {cs}; profile row {prow})")


if __name__ == "__main__":
    main()
