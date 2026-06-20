#!/usr/bin/env python3
"""Run a minimal StarAI Viola Isaac Sim scene with optional WebRTC livestream.

Typical use from /home/ubuntu/mcx/IsaacLab:

    ./isaaclab.sh -p /home/ubuntu/starai_isaac_viola/run_viola_stream.py \
        --asset /home/ubuntu/starai_isaac_viola/assets/viola.usd \
        --headless --livestream 2
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="StarAI Viola Isaac Sim livestream runner.")
parser.add_argument(
    "--asset",
    default="/home/ubuntu/starai_isaac_viola/assets/viola.usd",
    help="Path to the converted Viola USD asset.",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="Stop after this many sim steps. Use 0 for an interactive/streaming run.",
)
parser.add_argument(
    "--motion-scale",
    type=float,
    default=0.35,
    help="Joint sinusoid amplitude in radians for the smoke-test motion.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
requested_livestream = getattr(args_cli, "livestream", 0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.assets.articulation import ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def design_scene(asset_path: str) -> Articulation:
    """Create ground, lights, and a single Viola articulation."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    dome_cfg = sim_utils.DomeLightCfg(intensity=4500.0, color=(0.8, 0.82, 0.85))
    dome_cfg.func("/World/DomeLight", dome_cfg)

    key_cfg = sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.96, 0.9))
    key_cfg.func("/World/KeyLight", key_cfg, rotation=(0.6, 0.3, -0.2))

    sim_utils.create_prim("/World/ViolaOrigin", "Xform", translation=(0.0, 0.0, 0.0))

    robot_cfg = ArticulationCfg(
        prim_path="/World/ViolaOrigin/Viola",
        spawn=sim_utils.UsdFileCfg(
            usd_path=asset_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=120.0,
                velocity_limit_sim=8.0,
                stiffness=2500.0,
                damping=120.0,
            )
        },
    )
    return Articulation(cfg=robot_cfg)


def reset_robot(robot: Articulation) -> None:
    """Reset articulation root and joints to imported defaults."""
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()


def run_simulator(sim: SimulationContext, robot: Articulation) -> None:
    """Run a simple, bounded joint target pattern for visual validation."""
    sim_dt = sim.get_physics_dt()
    device = sim.device
    count = 0

    default_joint_pos = robot.data.default_joint_pos.clone()
    joint_count = default_joint_pos.shape[1]
    phase = torch.linspace(0.0, math.pi, joint_count, device=device).unsqueeze(0)
    amplitude = torch.full_like(default_joint_pos, args_cli.motion_scale)
    joint_limits = robot.data.soft_joint_pos_limits.clone()
    lower_limits = joint_limits[:, :, 0] + 1.0e-3
    upper_limits = joint_limits[:, :, 1] - 1.0e-3
    render_each_step = requested_livestream in (1, 2)

    print(f"[INFO] Viola USD: {args_cli.asset}", flush=True)
    print(f"[INFO] Joint count: {joint_count}", flush=True)
    print(f"[INFO] Joint names: {robot.data.joint_names}", flush=True)
    print(f"[INFO] Render each step: {render_each_step}", flush=True)
    print("[INFO] Setup complete. Starting simulation loop.", flush=True)

    while simulation_app.is_running():
        t = count * sim_dt
        target = default_joint_pos + amplitude * torch.sin(0.8 * t + phase)
        target = torch.clamp(target, lower_limits, upper_limits)

        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=render_each_step)
        robot.update(sim_dt)

        count += 1
        if count % 60 == 0:
            print(f"[INFO] Step {count}", flush=True)
        if args_cli.max_steps > 0 and count >= args_cli.max_steps:
            print(f"[INFO] Reached max steps: {args_cli.max_steps}", flush=True)
            break


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.8, -1.8, 1.4], [0.0, 0.0, 0.45])
    livestream_enabled = requested_livestream in (1, 2)

    print("[INFO] Creating Viola scene.", flush=True)
    robot = design_scene(args_cli.asset)
    print(f"[INFO] Resetting simulation. soft={livestream_enabled}", flush=True)
    sim.reset(soft=livestream_enabled)
    print("[INFO] Resetting robot state.", flush=True)
    reset_robot(robot)
    print("[INFO] Starting timeline.", flush=True)
    sim.play()
    run_simulator(sim, robot)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args_cli.max_steps > 0)
