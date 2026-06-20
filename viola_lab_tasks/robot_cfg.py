"""StarAI Viola articulation configuration for Isaac Lab tasks."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIOLA_USD = PROJECT_ROOT / "assets" / "viola.usd"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["joint7_left", "joint7_right"]
BASE_BODY = "base_link"
EE_BODY = "link6"

GRIPPER_OPEN = {"joint7_left": -0.025, "joint7_right": 0.025}
GRIPPER_CLOSED = {"joint7_left": -0.002, "joint7_right": 0.002}


def viola_usd_path() -> str:
    """Resolve the Viola USD path, allowing remote launch scripts to override it."""

    return os.environ.get("STARAI_VIOLA_ASSET", str(DEFAULT_VIOLA_USD))


VIOLA_HIGH_PD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=viola_usd_path(),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7_left": -0.025,
            "joint7_right": 0.025,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINTS,
            effort_limit_sim=120.0,
            velocity_limit_sim=8.0,
            stiffness=2500.0,
            damping=120.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=GRIPPER_JOINTS,
            effort_limit_sim=40.0,
            velocity_limit_sim=3.0,
            stiffness=900.0,
            damping=80.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""High-PD StarAI Viola configuration for task-space Isaac Lab demos."""
