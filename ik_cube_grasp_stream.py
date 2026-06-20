#!/usr/bin/env python3
"""IK cube grasp demo for StarAI Viola in Isaac Sim WebRTC."""

from __future__ import annotations

import argparse
from enum import Enum

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="StarAI Viola IK cube grasp WebRTC demo.")
parser.add_argument("--asset", default="/home/ubuntu/wjx/starai_isaac_viola/assets/viola.usd")
parser.add_argument("--max-steps", type=int, default=0, help="0 means run forever.")
parser.add_argument("--success-height", type=float, default=0.08)
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
    make_cube,
    make_ik,
    make_sphere_marker,
    make_viola,
    reset_robot,
    resolve_viola_handles,
    set_gripper_positions,
    setup_basic_sim,
    write_cube_pose,
)


class Phase(Enum):
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    CLOSE = "CLOSE"
    ATTACH = "ATTACH"
    LIFT = "LIFT"
    HOLD = "HOLD"
    RESET = "RESET"


def phase_goal(phase: Phase, anchor_pos: torch.Tensor, device: str) -> torch.Tensor:
    goal = anchor_pos.clone()
    if phase == Phase.APPROACH:
        goal[:, 2] += 0.20
    elif phase == Phase.DESCEND:
        goal[:, 2] += 0.075
    elif phase in (Phase.CLOSE, Phase.ATTACH):
        goal[:, 2] += 0.060
    elif phase in (Phase.LIFT, Phase.HOLD):
        goal[:, 2] += 0.24
    elif phase == Phase.RESET:
        goal = torch.tensor([[0.23, -0.25, 0.28]], device=device)
    return goal


def main() -> None:
    sim = setup_basic_sim(args_cli.device)
    add_lights_and_ground()
    robot = make_viola(args_cli.asset)
    cube = make_cube()
    target_marker = make_sphere_marker("/Visuals/grasp_target", (0.1, 0.85, 0.2), 0.025)
    ee_marker = make_sphere_marker("/Visuals/gripper_center", (0.1, 0.35, 1.0), 0.018)

    sim.reset(soft=False)
    handles = resolve_viola_handles(robot)
    joint_target = reset_robot(robot, handles, gripper_open=True)
    cube_initial_pos = torch.tensor([[0.32, 0.0, 0.035]], device=sim.device)
    write_cube_pose(cube, cube_initial_pos)
    ik = make_ik(sim.device)
    sim.play()

    phase = Phase.APPROACH
    phase_step = 0
    step = 0
    attached = False
    success_printed = False
    cube_lifted = False
    grasp_anchor_pos = cube_initial_pos.clone()
    smoothed_goal = phase_goal(phase, grasp_anchor_pos, sim.device)

    print("[INFO] IK cube grasp demo setup complete.", flush=True)
    print(f"[INFO] Robot joints: {robot.data.joint_names}", flush=True)
    print(f"[INFO] Robot bodies: {robot.data.body_names}", flush=True)

    while simulation_app.is_running():
        if not attached and phase in (Phase.APPROACH, Phase.DESCEND, Phase.CLOSE):
            grasp_anchor_pos = cube.data.root_pos_w.clone()
        if phase == Phase.RESET:
            grasp_anchor_pos = cube_initial_pos.clone()

        raw_goal = phase_goal(phase, grasp_anchor_pos, sim.device)
        alpha = 0.035 if phase in (Phase.LIFT, Phase.HOLD) else 0.055
        smoothed_goal = smoothed_goal + alpha * (raw_goal - smoothed_goal)
        goal = smoothed_goal.clone()
        joint_target, err = compute_ik_joint_target(ik, robot, handles, goal, joint_target)
        set_gripper_positions(robot, handles, joint_target, open_gripper=phase in (Phase.APPROACH, Phase.DESCEND, Phase.RESET))

        center = gripper_center_w(robot, handles)
        if phase in (Phase.ATTACH, Phase.LIFT, Phase.HOLD):
            attached = True
        if attached:
            cube_follow_pos = center + torch.tensor([[0.0, 0.0, -0.052]], device=sim.device)
            write_cube_pose(cube, cube_follow_pos)

        robot.set_joint_position_target(joint_target)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim.get_physics_dt())
        cube.update(sim.get_physics_dt())
        if attached:
            center = gripper_center_w(robot, handles)
            cube_follow_pos = center + torch.tensor([[0.0, 0.0, -0.052]], device=sim.device)
            write_cube_pose(cube, cube_follow_pos)

        target_marker.visualize(goal)
        ee_marker.visualize(center)
        circular_camera(sim, step)

        if step % 30 == 0:
            height_gain = float(cube.data.root_pos_w[0, 2] - cube_initial_pos[0, 2])
            print(
                f"[INFO] phase={phase.value} step={step} err={err:.4f} "
                f"cube_z={float(cube.data.root_pos_w[0, 2]):.3f} gain={height_gain:.3f} attached={attached}",
                flush=True,
            )

        if phase == Phase.APPROACH and (err < 0.055 or phase_step > 150):
            phase = Phase.DESCEND
            phase_step = -1
            print("[INFO] DESCEND", flush=True)
        elif phase == Phase.DESCEND and (err < 0.040 or phase_step > 120):
            phase = Phase.CLOSE
            phase_step = -1
            print("[INFO] CLOSE", flush=True)
        elif phase == Phase.CLOSE and phase_step > 45:
            phase = Phase.ATTACH
            phase_step = -1
            grasp_anchor_pos = gripper_center_w(robot, handles).clone()
            smoothed_goal = phase_goal(phase, grasp_anchor_pos, sim.device)
            print("[INFO] ATTACH", flush=True)
        elif phase == Phase.ATTACH and phase_step > 20:
            phase = Phase.LIFT
            phase_step = -1
            print("[INFO] LIFT", flush=True)
        elif phase == Phase.LIFT:
            height_gain = float(cube.data.root_pos_w[0, 2] - cube_initial_pos[0, 2])
            if height_gain >= args_cli.success_height and not success_printed:
                cube_lifted = True
                success_printed = True
                print(
                    f"[INFO] SUCCESS cube_lifted=true height_gain={height_gain:.3f} "
                    f"gripper_cube_dist={torch.linalg.norm(center - cube.data.root_pos_w).item():.4f}",
                    flush=True,
                )
            if phase_step > 160:
                phase = Phase.HOLD
                phase_step = -1
                print("[INFO] HOLD", flush=True)
        elif phase == Phase.HOLD and phase_step > 180 and args_cli.max_steps == 0:
            phase = Phase.RESET
            phase_step = -1
            print("[INFO] RESET", flush=True)
        elif phase == Phase.RESET and phase_step > 80:
            attached = False
            success_printed = False
            write_cube_pose(cube, cube_initial_pos)
            grasp_anchor_pos = cube_initial_pos.clone()
            smoothed_goal = phase_goal(Phase.APPROACH, grasp_anchor_pos, sim.device)
            phase = Phase.APPROACH
            phase_step = -1
            print("[INFO] APPROACH", flush=True)

        step += 1
        phase_step += 1
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            print(f"[INFO] Reached max steps: {args_cli.max_steps} cube_lifted={cube_lifted}", flush=True)
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args_cli.max_steps > 0)
