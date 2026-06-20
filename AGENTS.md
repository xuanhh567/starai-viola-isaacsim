# Agent Operating Guide

This repository is shared between a local Mac workspace and a remote RTX 5090
Isaac Sim machine. Codex and other agents must follow this guide before making
changes or running deployment commands.

## Core Rule

All source changes must be made in the local Mac repository first, then synced
to the RTX 5090 machine through GitHub.

Do not treat hand edits on the 5090 machine as the source of truth.

## Canonical Locations

- Local Mac repository:
  `/Users/wangjiaxuan/Robots/starai-viola-isaacsim`
- GitHub repository:
  `https://github.com/xuanhh567/starai-viola-isaacsim.git`
- RTX 5090 project directory:
  `/home/ubuntu/wjx/starai_isaac_viola`
- Preferred IsaacLab directory on 5090:
  `/home/ubuntu/wjx/IsaacLab`
- Read-only fallback IsaacLab directory on 5090:
  `/home/ubuntu/mcx/IsaacLab`

Historical notes may mention `/home/ubuntu/starai_isaac_viola`. For new work,
use `/home/ubuntu/wjx/starai_isaac_viola`.

## Required Workflow

1. Modify code, shell scripts, asset config, and documentation only in the local
   Mac repository.
2. Check local changes with:

   ```bash
   git status --short
   ```

3. Commit and push local changes:

   ```bash
   git add <changed-files>
   git commit -m "<message>"
   git push origin main
   ```

4. Sync the 5090 project directory from GitHub:

   ```bash
   ssh g5090 'cd /home/ubuntu/wjx/starai_isaac_viola && git pull --ff-only origin main'
   ```

5. Run demos or tests from `/home/ubuntu/wjx/starai_isaac_viola`, for example:

   ```bash
   ssh g5090 'cd /home/ubuntu/wjx/starai_isaac_viola && nohup ./start_ik_cube_grasp.sh > ik_cube_grasp.log 2>&1 & echo $!'
   ```

Future Isaac Lab task launchers should follow the same pattern, such as
`start_viola_lab_*.sh`.

## Remote Safety Boundaries

- Do not modify anything under `/home/ubuntu/mcx/*`.
- Do not edit `/home/ubuntu/mcx/IsaacLab`; it is a read-only dependency.
- Do not directly modify system-level IsaacLab installs, conda environments,
  GPU drivers, or Vulkan ICD files unless the user explicitly requests it.
- Do not use broad process killers such as:

  ```bash
  pkill -f starai_isaac_viola
  ```

  They may match the active SSH shell and interrupt the command itself.
- Prefer narrow process matching for a specific script, for example:

  ```bash
  ssh g5090 'pids=$(pgrep -f "python .*starai_isaac_viola/ik_cube_grasp_stream\\.py" || true); if [ -n "$pids" ]; then kill -9 $pids; fi'
  ```

## Dirty Worktree Handling

Before pulling or deploying on the 5090 machine, inspect the project state when
there is any doubt:

```bash
ssh g5090 'cd /home/ubuntu/wjx/starai_isaac_viola && git status --short'
```

If the 5090 worktree has uncommitted changes:

- Stop and ask the user before continuing.
- Do not run `git reset`, `git checkout --`, or overwrite files.
- Do not copy remote edits back as the final source of truth unless the user
  explicitly asks for that recovery flow.

## IsaacLab Integration Policy

When IsaacLab behavior needs to be adapted:

- Prefer wrappers, local task configs, local launch scripts, or patches inside
  this repository.
- Do not edit upstream IsaacLab files in `/home/ubuntu/mcx/IsaacLab`.
- If `/home/ubuntu/wjx/IsaacLab` exists and the user explicitly wants local
  IsaacLab changes, keep those changes separate from this repository's normal
  code flow and document them clearly.

## Verification Expectations

For code changes, prefer this sequence:

1. Local static checks where possible.
2. Commit and push to GitHub.
3. Pull on the 5090 machine with `git pull --ff-only origin main`.
4. Run the relevant script from `/home/ubuntu/wjx/starai_isaac_viola`.
5. Check logs and WebRTC status without modifying other users' directories.

Keep this file updated when the project workflow changes.
