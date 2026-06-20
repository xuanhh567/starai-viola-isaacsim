#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-}"

if [[ -z "$ISAACLAB_PATH" ]]; then
  if [[ -d /home/ubuntu/wjx/IsaacLab ]]; then
    ISAACLAB_PATH=/home/ubuntu/wjx/IsaacLab
  elif [[ -d /home/ubuntu/mcx/IsaacLab ]]; then
    ISAACLAB_PATH=/home/ubuntu/mcx/IsaacLab
  else
    echo "IsaacLab not found. Set ISAACLAB_PATH=/path/to/IsaacLab." >&2
    exit 1
  fi
fi

cd "$ISAACLAB_PATH"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export STARAI_VIOLA_ASSET="${STARAI_VIOLA_ASSET:-$SCRIPT_DIR/assets/viola.usd}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"

exec ./isaaclab.sh -p "$SCRIPT_DIR/run_viola_lab_task.py" \
  --task Isaac-Lift-Cube-Viola-IK-Pose-Abs-Play-v0 \
  --mode lift-sm \
  --visual-attach \
  --num-envs 1 \
  --headless \
  --livestream 2 \
  --kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0" \
  "$@"
