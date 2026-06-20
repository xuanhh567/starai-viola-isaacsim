#!/usr/bin/env python3
"""Run StarAI Viola Isaac Lab task variants."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_REACH_TASK = "Isaac-Reach-Viola-IK-Abs-Play-v0"
DEFAULT_LIFT_TASK = "Isaac-Lift-Cube-Viola-IK-Abs-Play-v0"

parser = argparse.ArgumentParser(description="Run StarAI Viola Isaac Lab manipulation tasks.")
parser.add_argument("--task", type=str, default=None, help="Gym task id to run.")
parser.add_argument("--mode", choices=("zero", "random", "lift-sm"), default="random")
parser.add_argument("--num-envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max-steps", type=int, default=0, help="0 means run until the app exits.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric USD I/O.")
parser.add_argument("--visual-attach", action="store_true", help="Attach the cube visually after grasp for demos.")
parser.add_argument(
    "--screenshot-dir",
    type=str,
    default="",
    help="Directory for Isaac Sim viewport screenshots. Disabled when empty.",
)
parser.add_argument(
    "--screenshot-every",
    type=int,
    default=0,
    help="Capture a viewport screenshot every N simulation steps when --screenshot-dir is set.",
)
parser.add_argument(
    "--no-screenshot-on-success",
    action="store_false",
    dest="screenshot_on_success",
    help="Disable the automatic success screenshot when --screenshot-dir is set.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import viola_lab_tasks  # noqa: F401, E402
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


class ViewportScreenshotter:
    """Small wrapper around Isaac Sim viewport capture for run verification."""

    def __init__(self, output_dir: str, task_name: str, mode: str, every_steps: int, capture_on_success: bool):
        self.enabled = bool(output_dir)
        self.every_steps = max(0, every_steps)
        self.capture_on_success = capture_on_success
        self.task_name = self._safe_name(task_name)
        self.mode = self._safe_name(mode)
        self._warned_no_viewport = False
        self._capture_viewport_to_file = None
        self._get_active_viewport = None
        self._pending_captures = []

        self.output_dir = self._resolve_output_dir(output_dir) if self.enabled else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Viewport screenshots enabled: {self.output_dir}", flush=True)

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)

    @staticmethod
    def _resolve_output_dir(output_dir: str) -> Path:
        path = Path(output_dir).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parent / path

    def _load_viewport_api(self):
        if self._capture_viewport_to_file is None or self._get_active_viewport is None:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

            self._capture_viewport_to_file = capture_viewport_to_file
            self._get_active_viewport = get_active_viewport

    def capture(self, step: int, event: str, state: int | None = None) -> None:
        if not self.enabled or self.output_dir is None:
            return

        try:
            self._load_viewport_api()
            viewport = self._get_active_viewport()
            if viewport is None:
                if not self._warned_no_viewport:
                    print("[WARN] No active Isaac Sim viewport; screenshot skipped.", flush=True)
                    self._warned_no_viewport = True
                return

            state_suffix = f"_state_{state}" if state is not None else ""
            filename = f"{self.task_name}_{self.mode}_step_{step:05d}_{self._safe_name(event)}{state_suffix}.png"
            file_path = self.output_dir / filename
            capture_obj = self._capture_viewport_to_file(viewport, file_path=str(file_path))
            if capture_obj is not None:
                self._pending_captures.append(capture_obj)
            print(f"[INFO] Screenshot scheduled: {file_path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - screenshot failures should not stop the demo.
            print(f"[WARN] Screenshot capture failed: {exc}", flush=True)

    def maybe_capture_interval(self, step: int, state: int | None = None) -> None:
        if self.every_steps > 0 and step % self.every_steps == 0:
            self.capture(step, "interval", state=state)

    def capture_success(self, step: int, state: int | None = None) -> None:
        if self.capture_on_success:
            self.capture(step, "success", state=state)

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            import omni.kit.renderer_capture

            omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
            print("[INFO] Screenshot capture queue flushed.", flush=True)
        except Exception as exc:  # noqa: BLE001 - final cleanup should stay best effort.
            print(f"[WARN] Screenshot flush failed: {exc}", flush=True)


def resolve_task_name() -> str:
    if args_cli.task:
        return args_cli.task
    if args_cli.mode == "lift-sm":
        return DEFAULT_LIFT_TASK
    return DEFAULT_REACH_TASK


def zero_actions(env: gym.Env) -> torch.Tensor:
    return torch.zeros(env.action_space.shape, device=env.unwrapped.device)


def random_actions(env: gym.Env) -> torch.Tensor:
    return 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)
    APPROACH_TIMEOUT = wp.constant(2.5)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_lift_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    above_offset: wp.array(dtype=wp.transform),
    grasp_offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(above_offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ) or sm_wait_time[tid] >= PickSmWaitTime.APPROACH_TIMEOUT:
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(grasp_offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ) or sm_wait_time[tid] >= PickSmWaitTime.APPROACH_TIMEOUT:
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(grasp_offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """Simple task-space pick-and-lift state machine for Lift-Cube IK actions."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        self.above_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.above_offset[:, 2] = 0.12
        self.above_offset[:, -1] = 1.0
        self.grasp_offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.grasp_offset[:, 2] = 0.03
        self.grasp_offset[:, -1] = 1.0

        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.above_offset_wp = wp.from_torch(self.above_offset, wp.transform)
        self.grasp_offset_wp = wp.from_torch(self.grasp_offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(
        self,
        ee_pose: torch.Tensor,
        object_pose: torch.Tensor,
        des_object_pose: torch.Tensor,
        position_only: bool,
    ) -> torch.Tensor:
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        wp.launch(
            kernel=infer_lift_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                wp.from_torch(ee_pose.contiguous(), wp.transform),
                wp.from_torch(object_pose.contiguous(), wp.transform),
                wp.from_torch(des_object_pose.contiguous(), wp.transform),
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.above_offset_wp,
                self.grasp_offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        if position_only:
            return torch.cat([des_ee_pose[:, :3], self.des_gripper_state.unsqueeze(-1)], dim=-1)
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def run_zero_or_random(env: gym.Env, screenshotter: ViewportScreenshotter) -> None:
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = zero_actions(env) if args_cli.mode == "zero" else random_actions(env)
            env.step(actions)
            screenshotter.maybe_capture_interval(step)
        step += 1
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            print(f"[INFO] Reached max steps: {args_cli.max_steps}", flush=True)
            break


def run_lift_state_machine(env: gym.Env, env_cfg, screenshotter: ViewportScreenshotter) -> None:
    wp.init()

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    position_only = actions.shape[-1] == 4
    actions[:, -1] = 1.0

    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        position_threshold=0.015,
    )

    step = 0
    success_printed = False
    initial_object_z = None
    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            if initial_object_z is None:
                initial_object_z = float(object_position[0, 2])

            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
            actions = pick_sm.compute(
                torch.cat([tcp_position, tcp_orientation], dim=-1),
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
                position_only=position_only,
            )

            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

            if args_cli.visual_attach and int(pick_sm.sm_state[0]) >= int(PickSmState.LIFT_OBJECT):
                object_asset = env.unwrapped.scene["object"]
                attach_offset = torch.tensor([[0.0, 0.0, -0.015]], device=env.unwrapped.device)
                cube_pos_w = tcp_position + env.unwrapped.scene.env_origins + attach_offset
                cube_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)
                object_asset.write_root_pose_to_sim(torch.cat([cube_pos_w, cube_quat_w], dim=-1))
                object_asset.write_root_velocity_to_sim(torch.zeros((env.unwrapped.num_envs, 6), device=env.unwrapped.device))

            if step % 60 == 0:
                cube_height = env.unwrapped.scene["object"].data.root_pos_w[:, 2] - env.unwrapped.scene.env_origins[:, 2]
                print(
                    f"[INFO] lift-sm step={step} state={int(pick_sm.sm_state[0])} "
                    f"cube_z={float(cube_height[0]):.3f} "
                    f"gain={float(cube_height[0]) - initial_object_z:.3f} "
                    f"wait={float(pick_sm.sm_wait_time[0]):.2f} action_dim={actions.shape[-1]} "
                    f"visual_attach={args_cli.visual_attach}",
                    flush=True,
                )
                if args_cli.visual_attach and not success_printed and float(cube_height[0]) - initial_object_z > 0.05:
                    success_printed = True
                    print("[INFO] SUCCESS visual_attach_cube_lifted=true", flush=True)
                    screenshotter.capture_success(step, state=int(pick_sm.sm_state[0]))

            screenshotter.maybe_capture_interval(step, state=int(pick_sm.sm_state[0]))

        step += 1
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            print(f"[INFO] Reached max steps: {args_cli.max_steps}", flush=True)
            break


def main() -> None:
    task_name = resolve_task_name()
    print(f"[INFO] Running task={task_name} mode={args_cli.mode}", flush=True)

    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(task_name, cfg=env_cfg)

    print(f"[INFO] Gym observation space: {env.observation_space}", flush=True)
    print(f"[INFO] Gym action space: {env.action_space}", flush=True)
    env.reset()
    screenshotter = ViewportScreenshotter(
        args_cli.screenshot_dir,
        task_name,
        args_cli.mode,
        args_cli.screenshot_every,
        args_cli.screenshot_on_success,
    )

    try:
        if args_cli.mode == "lift-sm":
            run_lift_state_machine(env, env_cfg, screenshotter)
        else:
            run_zero_or_random(env, screenshotter)
    finally:
        screenshotter.flush()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args_cli.max_steps > 0)
