#!/bin/bash
set -u
cd ~/Desktop/ALDM/LDM
CFG_DIR=configs/latent-diffusion/abl
OUT_ROOT=checkpoints_abl
AE_DIR=$(dirname "$(find ~/Desktop -name train_autoencoder.py | head -1)")
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}
mkdir -p $OUT_ROOT eval_abl

NAMES=()
for a in "$@"; do
  if [ -f "$CFG_DIR/$a.yaml" ]; then NAMES+=("$a"); else
    for f in $CFG_DIR/${a}_*.yaml; do [ -f "$f" ] && NAMES+=("$(basename "$f" .yaml)"); done
  fi
done
[ ${#NAMES[@]} -eq 0 ] && { echo "no configs matched: $*"; exit 1; }
echo "queue (${#NAMES[@]}): ${NAMES[*]}"

for NAME in "${NAMES[@]}"; do
  OUT=$OUT_ROOT/$NAME
  if [ -f "$OUT/DONE" ]; then echo "== $NAME already done, skip"; continue; fi
  echo "== [$(date '+%m-%d %H:%M')] train $NAME"
  mkdir -p "$OUT"
  python -u train_diffusion.py --config "$CFG_DIR/$NAME.yaml" --out_dir "$OUT" \
    --cache_dir ~/Desktop/data/spade_cache --latent_cache ~/Desktop/data/latent_cache \
    --epochs $EPOCHS --patience $PATIENCE --val_every 1 --ckpt_every 10 > "$OUT/train.log" 2>&1
  if [ ! -f "$OUT/diffusion_best.pth" ]; then echo "   !! $NAME: no diffusion_best.pth -- see $OUT/train.log"; continue; fi
  echo "== [$(date '+%m-%d %H:%M')] test $NAME (best ckpt, hist_match both)"
  python evaluate_diffusion_resshift_v2.py --config "$CFG_DIR/$NAME.yaml" --checkpoint "$OUT/diffusion_best.pth" \
    --test_dir ~/Desktop/test --subjects 212730 212731 216393 216394 \
    --hist_match both --autoencoder_script_dir "$AE_DIR" --no_ema \
    --out_csv "eval_abl/$NAME.csv" > "$OUT/test.log" 2>&1 && touch "$OUT/DONE"
  grep "^SSIM=" "$OUT/test.log"

  python - <<'PY'
import glob, os, re, torch, pandas as pd
rows = []
for f in sorted(glob.glob("eval_abl/*.csv")):
    name = os.path.basename(f)[:-4]
    d = pd.read_csv(f); o = d[d["subject"].astype(str) == "overall"].iloc[0]
    ck = torch.load(f"checkpoints_abl/{name}/diffusion_best.pth", map_location="cpu", weights_only=False)
    log = pd.read_csv(f"checkpoints_abl/{name}/log.csv")
    m = re.match(r"p(\d)_(\w+?)(?:_T(\d+)_k([\d.]+))?$", name)
    rows.append({"run": name, "phase": int(m.group(1)), "condition": m.group(2),
                 "T": int(m.group(3) or 15), "kappa_mult": float(m.group(4) or 1.0),
                 "best_epoch": ck.get("epoch"), "epochs_run": int(log["epoch"].max()) + 1,
                 "best_val_loss": ck.get("val_loss"),
                 **{k: float(o[k]) for k in ("ssim", "psnr", "mae", "mae_brain", "lpips")}})
df = pd.DataFrame(rows)
df.to_csv("ablation_results.csv", index=False)
print(df.round(4).to_string(index=False))
PY
done
echo "== all done [$(date '+%m-%d %H:%M')]"
