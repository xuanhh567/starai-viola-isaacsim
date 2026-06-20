"""Isaac Lab Reach task variant for StarAI Viola."""

from __future__ import annotations

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from .robot_cfg import ARM_JOINTS, EE_BODY, VIOLA_HIGH_PD_CFG


@configclass
class ViolaReachEnvCfg(ReachEnvCfg):
    """Reach pose-tracking environment with the StarAI Viola arm."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = VIOLA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [EE_BODY]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [EE_BODY]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [EE_BODY]

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINTS,
            body_name=EE_BODY,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        )

        self.commands.ee_pose.body_name = EE_BODY
        self.commands.ee_pose.ranges.pos_x = (0.22, 0.42)
        self.commands.ee_pose.ranges.pos_y = (-0.20, 0.20)
        self.commands.ee_pose.ranges.pos_z = (0.12, 0.36)
        self.commands.ee_pose.ranges.roll = (0.0, 0.0)
        self.commands.ee_pose.ranges.pitch = (0.0, 0.0)
        self.commands.ee_pose.ranges.yaw = (0.0, 0.0)


@configclass
class ViolaReachEnvCfg_PLAY(ViolaReachEnvCfg):
    """Small single-env Reach variant for WebRTC demos."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
