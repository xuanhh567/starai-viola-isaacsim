"""Local Isaac Lab task registrations for StarAI Viola."""

from __future__ import annotations

import gymnasium as gym


gym.register(
    id="Isaac-Reach-Viola-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "viola_lab_tasks.reach_env_cfg:ViolaReachEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-Viola-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "viola_lab_tasks.reach_env_cfg:ViolaReachEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-Viola-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "viola_lab_tasks.lift_env_cfg:ViolaCubeLiftEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-Viola-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "viola_lab_tasks.lift_env_cfg:ViolaCubeLiftEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
