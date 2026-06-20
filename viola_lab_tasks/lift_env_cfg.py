"""Isaac Lab Lift-Cube task variant for StarAI Viola."""

from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from .robot_cfg import (
    ARM_JOINTS,
    BASE_BODY,
    EE_BODY,
    GRIPPER_CLOSED,
    GRIPPER_JOINTS,
    GRIPPER_OPEN,
    TCP_OFFSET_POS,
    VIOLA_HIGH_PD_CFG,
)


@configclass
class ViolaCubeLiftEnvCfg(LiftEnvCfg):
    """Lift-Cube environment with the StarAI Viola arm."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = VIOLA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINTS,
            body_name=EE_BODY,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TCP_OFFSET_POS),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=GRIPPER_JOINTS,
            open_command_expr=GRIPPER_OPEN,
            close_command_expr=GRIPPER_CLOSED,
        )

        self.commands.object_pose.body_name = EE_BODY
        self.commands.object_pose.debug_vis = False
        self.commands.object_pose.ranges.pos_x = (0.25, 0.40)
        self.commands.object_pose.ranges.pos_y = (-0.16, 0.16)
        self.commands.object_pose.ranges.pos_z = (0.18, 0.34)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.32, 0.0, 0.055], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_BODY}",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{EE_BODY}",
                    name="end_effector",
                    offset=OffsetCfg(pos=TCP_OFFSET_POS),
                ),
            ],
        )


@configclass
class ViolaCubeLiftEnvCfg_PLAY(ViolaCubeLiftEnvCfg):
    """Small single-env Lift-Cube variant for WebRTC demos."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
