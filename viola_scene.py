"""Shared IsaacLab helpers for the StarAI Viola demos."""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import subtract_frame_transforms


ARM_JOINT_EXPR = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
LEFT_GRIPPER_JOINT = "joint7_left"
RIGHT_GRIPPER_JOINT = "joint7_right"
EE_BODY = "link6"
LEFT_FINGER_BODY = "link7_left"
RIGHT_FINGER_BODY = "link7_right"


def add_lights_and_ground() -> None:
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    dome_cfg = sim_utils.DomeLightCfg(intensity=4500.0, color=(0.8, 0.82, 0.85))
    dome_cfg.func("/World/DomeLight", dome_cfg)

    key_cfg = sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.96, 0.9))
    key_cfg.func("/World/KeyLight", key_cfg, rotation=(0.6, 0.3, -0.2))


def make_viola(asset_path: str, prim_path: str = "/World/Viola") -> Articulation:
    robot_cfg = ArticulationCfg(
        prim_path=prim_path,
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
            "arm": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINT_EXPR,
                effort_limit_sim=120.0,
                velocity_limit_sim=8.0,
                stiffness=2500.0,
                damping=120.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[LEFT_GRIPPER_JOINT, RIGHT_GRIPPER_JOINT],
                effort_limit_sim=40.0,
                velocity_limit_sim=3.0,
                stiffness=900.0,
                damping=80.0,
            ),
        },
    )
    return Articulation(cfg=robot_cfg)


def make_cube(
    prim_path: str = "/World/Cube",
    pos: tuple[float, float, float] = (0.32, 0.0, 0.035),
    size: float = 0.07,
) -> RigidObject:
    cube_cfg = RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=3.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.22, 0.12)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )
    return RigidObject(cfg=cube_cfg)


def make_sphere_marker(prim_path: str, color: tuple[float, float, float], radius: float) -> VisualizationMarkers:
    cfg = SPHERE_MARKER_CFG.copy()
    cfg.markers["sphere"].radius = radius
    cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    return VisualizationMarkers(cfg.replace(prim_path=prim_path))


def resolve_viola_handles(robot: Articulation) -> dict[str, object]:
    arm_joint_ids, arm_joint_names = robot.find_joints(ARM_JOINT_EXPR, preserve_order=True)
    left_joint_ids, _ = robot.find_joints(LEFT_GRIPPER_JOINT)
    right_joint_ids, _ = robot.find_joints(RIGHT_GRIPPER_JOINT)
    ee_body_ids, ee_body_names = robot.find_bodies(EE_BODY)
    left_body_ids, _ = robot.find_bodies(LEFT_FINGER_BODY)
    right_body_ids, _ = robot.find_bodies(RIGHT_FINGER_BODY)
    if not arm_joint_ids or not left_joint_ids or not right_joint_ids or not ee_body_ids:
        raise RuntimeError(
            "Could not resolve Viola handles. "
            f"joints={robot.data.joint_names}, bodies={robot.data.body_names}"
        )
    return {
        "arm_joint_ids": arm_joint_ids,
        "arm_joint_names": arm_joint_names,
        "left_joint_id": left_joint_ids[0],
        "right_joint_id": right_joint_ids[0],
        "ee_body_id": ee_body_ids[0],
        "ee_body_name": ee_body_names[0],
        "left_body_id": left_body_ids[0],
        "right_body_id": right_body_ids[0],
    }


def reset_robot(robot: Articulation, handles: dict[str, object], gripper_open: bool = True) -> torch.Tensor:
    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    if gripper_open:
        set_gripper_positions(robot, handles, joint_pos, open_gripper=True)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    return joint_pos


def set_gripper_positions(
    robot: Articulation,
    handles: dict[str, object],
    target: torch.Tensor,
    open_gripper: bool,
) -> None:
    limits = robot.data.soft_joint_pos_limits
    left_id = int(handles["left_joint_id"])
    right_id = int(handles["right_joint_id"])
    if open_gripper:
        desired = {left_id: -0.025, right_id: 0.025}
    else:
        desired = {left_id: -0.002, right_id: 0.002}
    for joint_id, value in desired.items():
        lo = float(limits[0, joint_id, 0]) + 1.0e-4
        hi = float(limits[0, joint_id, 1]) - 1.0e-4
        target[:, joint_id] = max(lo, min(hi, value))


def gripper_center_w(robot: Articulation, handles: dict[str, object]) -> torch.Tensor:
    left = robot.data.body_pose_w[:, int(handles["left_body_id"]), 0:3]
    right = robot.data.body_pose_w[:, int(handles["right_body_id"]), 0:3]
    return 0.5 * (left + right)


def make_ik(device: str) -> DifferentialIKController:
    cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    return DifferentialIKController(cfg, num_envs=1, device=device)


def compute_ik_joint_target(
    ik: DifferentialIKController,
    robot: Articulation,
    handles: dict[str, object],
    desired_gripper_center_w: torch.Tensor,
    current_joint_target: torch.Tensor,
    max_joint_delta: float = 0.025,
) -> tuple[torch.Tensor, float]:
    arm_joint_ids = handles["arm_joint_ids"]
    ee_body_id = int(handles["ee_body_id"])
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    root_pose_w = robot.data.root_pose_w
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    desired_pos_b, _ = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        desired_gripper_center_w,
        ee_pose_w[:, 3:7],
    )
    ik.set_command(desired_pos_b, ee_quat=ee_quat_b)

    jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    arm_target = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids]
    arm_target = torch.clamp(arm_target, limits[:, :, 0] + 1.0e-3, limits[:, :, 1] - 1.0e-3)
    previous = current_joint_target[:, arm_joint_ids]
    delta = torch.clamp(arm_target - previous, -max_joint_delta, max_joint_delta)
    arm_target = previous + delta
    current_joint_target[:, arm_joint_ids] = arm_target

    err = torch.linalg.norm(gripper_center_w(robot, handles) - desired_gripper_center_w, dim=1).item()
    return current_joint_target, err


def write_cube_pose(cube: RigidObject, pos_w: torch.Tensor, quat_w: torch.Tensor | None = None) -> None:
    if quat_w is None:
        quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=pos_w.device)
    pose = torch.cat((pos_w, quat_w), dim=1)
    cube.write_root_pose_to_sim(pose)
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=pos_w.device))
    cube.update(0.0)


def setup_basic_sim(device: str) -> SimulationContext:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.15, -1.35, 0.95], [0.24, 0.0, 0.25])
    return sim


def circular_camera(sim: SimulationContext, frame: int, radius: float = 1.55) -> None:
    if frame % 180 == 0:
        angle = 0.45 + frame / 180 * 0.18
        eye = [radius * math.cos(angle), -radius * math.sin(angle), 1.0]
        sim.set_camera_view(eye, [0.22, 0.0, 0.24])
