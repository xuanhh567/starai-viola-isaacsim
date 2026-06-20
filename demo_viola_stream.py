#!/usr/bin/env python3
"""Stable joint-space motion demo for StarAI Viola."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="StarAI Viola joint motion WebRTC demo.")
parser.add_argument("--asset", default="/home/ubuntu/wjx/starai_isaac_viola/assets/viola.usd")
parser.add_argument("--max-steps", type=int, default=0)
parser.add_argument("--motion-scale", type=float, default=0.28)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from viola_scene import (  # noqa: E402
    add_lights_and_ground,
    circular_camera,
    make_viola,
    reset_robot,
    resolve_viola_handles,
    set_gripper_positions,
    setup_basic_sim,
)


def main() -> None:
    sim = setup_basic_sim(args_cli.device)
    add_lights_and_ground()
    robot = make_viola(args_cli.asset)
    sim.reset(soft=False)
    handles = resolve_viola_handles(robot)
    joint_target = reset_robot(robot, handles, gripper_open=True)
    sim.play()

    arm_ids = handles["arm_joint_ids"]
    default_joint_pos = robot.data.default_joint_pos.clone()
    limits = robot.data.soft_joint_pos_limits
    phase = torch.linspace(0.0, math.pi, len(arm_ids), device=sim.device).unsqueeze(0)
    step = 0
    print("[INFO] Joint motion demo setup complete.", flush=True)

    while simulation_app.is_running():
        t = step * sim.get_physics_dt()
        arm_target = default_joint_pos[:, arm_ids] + args_cli.motion_scale * torch.sin(0.8 * t + phase)
        arm_target = torch.clamp(arm_target, limits[:, arm_ids, 0] + 1.0e-3, limits[:, arm_ids, 1] - 1.0e-3)
        joint_target[:, arm_ids] = arm_target
        set_gripper_positions(robot, handles, joint_target, open_gripper=True)
        robot.set_joint_position_target(joint_target)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim.get_physics_dt())
        circular_camera(sim, step)
        step += 1
        if step % 60 == 0:
            print(f"[INFO] Step {step}", flush=True)
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            print(f"[INFO] Reached max steps: {args_cli.max_steps}", flush=True)
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args_cli.max_steps > 0)
