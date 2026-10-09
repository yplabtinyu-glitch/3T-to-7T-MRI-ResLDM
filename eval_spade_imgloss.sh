#!/bin/bash
# EVERY=10 (default): only epochs divisible by EVERY, plus the newest checkpoint
# HM=none|3t|both  : histogram matching mode (default none)
EVERY=${EVERY:-10}
HM=${HM:-none}
AE_DIR=$(dirname "$(find ~/Desktop -name train_autoencoder.py | head -1)")
CK=checkpoints_diffusion_ablation_3t/diffusion_epoch9.pth
CFG=$(python -c "import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['args']['config'])")
OUTD=eval_$(basename ${SPADE_DIR:-imgloss})_hm${HM}
mkdir -p "$OUTD"
ALL=$(ls ${SPADE_DIR:-$HOME/Desktop/LSC/checkpoints_spade_imgloss}/spade_epoch*.pth 2>/dev/null | sort -V)
LAST=$(echo "$ALL" | tail -1)
for S in $ALL; do
  EP=$(basename "$S" .pth | sed 's/spade_epoch//')
  if [ $((EP % EVERY)) -ne 0 ] && [ "$S" != "$LAST" ]; then continue; fi
  OUT=$OUTD/ep${EP}.csv
  [ -f "$OUT" ] && continue
  echo "== SPADE epoch $EP (hist_match=$HM)"
  python evaluate_diffusion_resshift_v2.py --config "$CFG" --checkpoint "$CK" \
    --test_dir ~/Desktop/test --subjects 212730 212731 216393 216394 \
    --spade_only --spade_ckpt "$S" --hist_match "$HM" --autoencoder_script_dir "$AE_DIR" --no_ema \
    --out_csv "$OUT" 2>&1 | grep "spade_ckpt\|SSIM=" | tail -2
done
OUTD="$OUTD" python - <<'PY'
import glob, os, re, pandas as pd
d = os.environ["OUTD"]
rows = [{"epoch": 10, "ssim": 0.7980, "psnr": 24.51, "mae": 0.0365, "mae_brain": 0.1358, "lpips": 0.3754, "note": "baseline old SPADE (hm none)"}]
for f in glob.glob(f"{d}/ep*.csv"):
    t = pd.read_csv(f); o = t[t["subject"].astype(str) == "overall"].iloc[0]
    rows.append({"epoch": int(re.search(r"ep(\d+)", f).group(1)), **{k: float(o[k]) for k in ("ssim", "psnr", "mae", "mae_brain", "lpips")}, "note": "imgloss"})
df = pd.DataFrame(rows).sort_values(["epoch", "note"])
df.to_csv(f"{d}.csv", index=False)
print(df.round(4).to_string(index=False))
PY
