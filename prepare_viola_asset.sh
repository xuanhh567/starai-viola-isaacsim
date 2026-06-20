#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/ubuntu/starai_isaac_viola"
ISAACLAB_DIR="/home/ubuntu/mcx/IsaacLab"
SRC_URDF="/home/ubuntu/starai_ws/src/fashionstar-starai-arm-ros2/src/viola_gazebo/urdf/viola_gazebo.urdf"
PKG_DIR="/home/ubuntu/starai_ws/src/fashionstar-starai-arm-ros2/src/viola_gazebo"
ISAAC_URDF="${PROJECT_DIR}/assets/viola_isaac.urdf"
OUT_USD="${PROJECT_DIR}/assets/viola.usd"

mkdir -p "${PROJECT_DIR}/assets"

python3 - "${SRC_URDF}" "${ISAAC_URDF}" "${PKG_DIR}" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
pkg = Path(sys.argv[3])

text = src.read_text()
text = text.replace("file://$(find viola_gazebo)", f"file://{pkg}")
text = text.replace("$(find viola_gazebo)", str(pkg))
dst.write_text(text)
print(f"[INFO] Wrote Isaac-compatible URDF: {dst}")
PY

cd "${ISAACLAB_DIR}"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

./isaaclab.sh -p scripts/tools/convert_urdf.py \
  "${ISAAC_URDF}" \
  "${OUT_USD}" \
  --fix-base \
  --joint-target-type position

ls -lh "${ISAAC_URDF}" "${OUT_USD}"
