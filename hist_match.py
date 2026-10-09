import argparse
import os

import numpy as np

PCTS = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99], dtype=np.float64)


def landmarks(vals):
    return np.percentile(vals, PCTS)


def apply_mapping(vol01, src_lm, ref_lm, thr):
    out = vol01.copy()
    m = vol01 > thr
    v = vol01[m].astype(np.float64)
    mapped = np.interp(v, src_lm, ref_lm)
    lo, hi = v < src_lm[0], v > src_lm[-1]
    s_lo = (ref_lm[1] - ref_lm[0]) / max(src_lm[1] - src_lm[0], 1e-8)
    s_hi = (ref_lm[-1] - ref_lm[-2]) / max(src_lm[-1] - src_lm[-2], 1e-8)
    mapped[lo] = ref_lm[0] + (v[lo] - src_lm[0]) * s_lo
    mapped[hi] = ref_lm[-1] + (v[hi] - src_lm[-1]) * s_hi
    out[m] = np.clip(mapped, 0.0, 1.0).astype(vol01.dtype)
    return out


def match_volume(vol01, ref_lm, thr, z_lo=0.2, z_hi=0.8):
    n = vol01.shape[2]
    sub = vol01[:, :, int(n * z_lo):int(n * z_hi)]
    src_lm = landmarks(sub[sub > thr])
    return apply_mapping(vol01, src_lm, ref_lm, thr), src_lm


def build(args):
    rng = np.random.default_rng(args.seed)
    out = {"pcts": PCTS, "thr": np.float64(args.thr)}
    for key, fn in (("3t", "train_src.npy"), ("7t", "train_tgt.npy")):
        a = np.load(os.path.join(args.cache_dir, fn), mmap_mode="r")
        idx = np.sort(rng.choice(len(a), min(args.n_slices, len(a)), replace=False))
        x = (np.asarray(a[idx], dtype=np.float32) + 1) / 2
        v = x[x > args.thr]
        out[f"lm_{key}"] = landmarks(v)
        h, e = np.histogram(v, bins=200, range=(0, 1), density=True)
        out[f"hist_{key}"], out["bin_edges"] = h, e
        print(f"train {key.upper()}: {len(idx)} slices, {v.size} brain voxels")
        print("  landmarks " + " ".join(f"p{int(p)}={l:.3f}" for p, l in zip(PCTS, out[f'lm_{key}'])))
    np.savez(args.out, **out)
    print(f"saved {args.out}")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    ref = np.load(args.ref)
    thr = float(ref["thr"])
    e = ref["bin_edges"]; c = (e[:-1] + e[1:]) / 2
    subjects = args.subjects or sorted(f[:-7] for f in os.listdir(os.path.join(args.test_dir, "7T")) if f.endswith(".nii.gz"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=140, sharex=True)
    rows = []
    for col, (key, sub_dir) in enumerate((("3t", "reg_3T"), ("7t", "7T"))):
        for row, matched in enumerate((False, True)):
            ax = axes[row, col]
            ax.plot(c, ref[f"hist_{key}"], color="k", lw=2.5, label="train (spade_cache)")
            for s in subjects:
                vol = nib.load(os.path.join(args.test_dir, sub_dir, f"{s}.nii.gz")).get_fdata(dtype=np.float32)
                vol = (vol - vol.min()) / max(vol.max() - vol.min(), 1e-6)
                lm_before = None
                if matched:
                    vol, lm_before = match_volume(vol, ref[f"lm_{key}"], thr)
                n = vol.shape[2]
                sub = vol[:, :, int(n * 0.2):int(n * 0.8)]
                v = sub[sub > thr]
                h, _ = np.histogram(v, bins=e, density=True)
                ax.plot(c, h, lw=1.3, label=f"test {s}")
                if matched:
                    rows.append((key.upper(), s, lm_before, landmarks(v)))
            ax.set_title(f"{key.upper()} brain voxels - {'after' if matched else 'before'} histogram matching", fontsize=10)
            ax.set_yticks([]); ax.legend(fontsize=7, frameon=False)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("intensity ([0,1], per-volume min-max as in evaluation)")
    fig.tight_layout()
    fig.savefig(args.out_png, bbox_inches="tight")
    print(f"saved {args.out_png}\n")
    hdr = "        " + " ".join(f"{'p'+str(int(p)):>6s}" for p in PCTS)
    for key in ("3T", "7T"):
        print(f"[{key}] landmarks{'':4s}{hdr}")
        print(f"  {'train':<14s}" + " ".join(f"{l:6.3f}" for l in ref[f'lm_{key.lower()}']))
        for k, s, before, after in rows:
            if k == key:
                print(f"  {s+' before':<14s}" + " ".join(f"{l:6.3f}" for l in before))
                print(f"  {s+' after':<14s}" + " ".join(f"{l:6.3f}" for l in after))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--cache_dir", default=os.path.expanduser("~/Desktop/data/spade_cache"))
    b.add_argument("--n_slices", type=int, default=3000)
    b.add_argument("--thr", type=float, default=0.02)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="hist_ref_train.npz")
    p = sp.add_parser("plot")
    p.add_argument("--ref", default="hist_ref_train.npz")
    p.add_argument("--test_dir", default=os.path.expanduser("~/Desktop/test"))
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--out_png", default="hist_train_vs_test.png")
    args = ap.parse_args()
    build(args) if args.cmd == "build" else plot(args)
