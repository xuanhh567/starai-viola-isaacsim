#!/usr/bin/env python3
"""Stream a static StarAI Viola USD scene through Isaac Sim WebRTC."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="StarAI Viola WebRTC viewer.")
parser.add_argument(
    "--asset",
    default="/home/ubuntu/starai_isaac_viola/assets/viola.usd",
    help="Path to the converted Viola USD asset.",
)
parser.add_argument("--max-frames", type=int, default=0, help="Stop after N rendered frames. 0 means run forever.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    dome_cfg = sim_utils.DomeLightCfg(intensity=4500.0, color=(0.8, 0.82, 0.85))
    dome_cfg.func("/World/DomeLight", dome_cfg)

    key_cfg = sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.96, 0.9))
    key_cfg.func("/World/KeyLight", key_cfg, rotation=(0.6, 0.3, -0.2))

    robot_prim = sim_utils.create_prim(
        "/World/Viola",
        "Xform",
        usd_path=args_cli.asset,
        translation=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )
    if not robot_prim.IsValid():
        raise RuntimeError(f"Failed to create Viola prim from {args_cli.asset}")

    sim.set_camera_view([1.8, -1.8, 1.35], [0.0, 0.0, 0.45])

    print(f"[INFO] Streaming static Viola scene from: {args_cli.asset}", flush=True)
    print("[INFO] WebRTC should show the robot now. Press Ctrl-C or kill the process to stop.", flush=True)

    frame = 0
    while simulation_app.is_running():
        # Keep the viewport alive and move the view slowly so clients get changing frames.
        if frame % 120 == 0:
            angle = frame / 120 * 0.18
            eye = [1.8 * math.cos(angle), -1.8 * math.sin(angle) - 0.2, 1.35]
            sim.set_camera_view(eye, [0.0, 0.0, 0.45])
            print(f"[INFO] Render frame {frame}", flush=True)
        simulation_app.update()
        frame += 1
        if args_cli.max_frames > 0 and frame >= args_cli.max_frames:
            print(f"[INFO] Reached max frames: {args_cli.max_frames}", flush=True)
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args_cli.max_frames > 0)
