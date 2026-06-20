#!/usr/bin/env python3
"""IK reach demo for StarAI Viola."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="StarAI Viola IK reach WebRTC demo.")
parser.add_argument("--asset", default="/home/ubuntu/wjx/starai_isaac_viola/assets/viola.usd")
parser.add_argument("--max-steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from viola_scene import (  # noqa: E402
    add_lights_and_ground,
    circular_camera,
    compute_ik_joint_target,
    gripper_center_w,
    make_ik,
    make_sphere_marker,
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
    target_marker = make_sphere_marker("/Visuals/reach_target", (0.1, 0.85, 0.2), 0.025)
    ee_marker = make_sphere_marker("/Visuals/gripper_center", (0.1, 0.35, 1.0), 0.018)
    sim.reset(soft=False)
    handles = resolve_viola_handles(robot)
    joint_target = reset_robot(robot, handles, gripper_open=True)
    ik = make_ik(sim.device)
    sim.play()

    goals = torch.tensor(
        [[0.28, 0.12, 0.24], [0.36, -0.10, 0.20], [0.24, 0.0, 0.34]],
        device=sim.device,
    )
    goal_idx = 0
    step = 0
    print("[INFO] Reach demo setup complete.", flush=True)

    while simulation_app.is_running():
        goal = goals[goal_idx : goal_idx + 1]
        joint_target, err = compute_ik_joint_target(ik, robot, handles, goal, joint_target)
        set_gripper_positions(robot, handles, joint_target, open_gripper=True)
        robot.set_joint_position_target(joint_target)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim.get_physics_dt())
        center = gripper_center_w(robot, handles)
        target_marker.visualize(goal)
        ee_marker.visualize(center)
        circular_camera(sim, step)

        if step % 30 == 0:
            print(f"[INFO] goal={goal_idx} step={step} err={err:.4f}", flush=True)
        if err < 0.035 or step % 180 == 179:
            goal_idx = (goal_idx + 1) % goals.shape[0]
            print(f"[INFO] Switching reach goal to {goal_idx}", flush=True)
        step += 1
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            print(f"[INFO] Reached max steps: {args_cli.max_steps}", flush=True)
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args_cli.max_steps > 0)
