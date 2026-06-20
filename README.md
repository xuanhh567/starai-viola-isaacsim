# StarAI Viola Isaac Sim Demos

Isaac Sim / IsaacLab demos for the StarAI Viola arm on the remote RTX 5090 host.

## Remote Host

- Host alias: `g5090`
- User: `ubuntu`
- Tailscale IP: `100.126.98.21`
- Runtime project path: `/home/ubuntu/wjx/starai_isaac_viola`
- IsaacLab path: set `ISAACLAB_PATH` if needed. Scripts prefer `/home/ubuntu/wjx/IsaacLab` and fall back to the existing read-only installation at `/home/ubuntu/mcx/IsaacLab`.

## Main Demo: IK Cube Grasp

Start the WebRTC demo:

```bash
ssh g5090 'cd /home/ubuntu/wjx/starai_isaac_viola && nohup ./start_ik_cube_grasp.sh > ik_cube_grasp.log 2>&1 & echo $!'
```

Open Isaac Sim WebRTC Streaming Client on the Mac:

```text
Server: 100.126.98.21
Port: 49100
```

Expected behavior:

- Viola opens the gripper.
- IK moves the gripper center above the cube.
- The arm descends to the cube.
- The gripper closes.
- The cube attaches to the gripper center and lifts.
- The log prints `SUCCESS cube_lifted=true`.

Check status:

```bash
ssh g5090 'pgrep -af "ik_cube_grasp_stream.py|isaaclab.sh -p" || true'
ssh g5090 'ss -tulpn | grep -E "49100|47998" || true'
ssh g5090 'tail -120 /home/ubuntu/wjx/starai_isaac_viola/ik_cube_grasp.log'
```

Stop demos:

```bash
ssh g5090 'pids=$(pgrep -f "python .*starai_isaac_viola/.*_stream\\.py" || true); if [ -n "$pids" ]; then kill -9 $pids; fi'
```

## Supporting Demos

Isaac Lab official Reach task variant:

```bash
ssh g5090 'cd /home/ubuntu/wjx/starai_isaac_viola && nohup ./start_viola_lab_reach.sh > viola_lab_reach.log 2>&1 & echo $!'
```

Isaac Lab official Lift-Cube task variant:

```bash
ssh g5090 'cd /home/ubuntu/wjx/starai_isaac_viola && nohup ./start_viola_lab_lift.sh > viola_lab_lift.log 2>&1 & echo $!'
```

Reach-only IK:

```bash
ssh g5090 'cd /home/ubuntu/wjx/starai_isaac_viola && nohup ./start_viola_reach.sh > reach_viola.log 2>&1 & echo $!'
```

Stable joint-space motion:

```bash
ssh g5090 'cd /home/ubuntu/wjx/starai_isaac_viola && nohup ./start_viola_demo.sh > demo_viola.log 2>&1 & echo $!'
```

Headless smoke test:

```bash
ssh g5090 '/home/ubuntu/wjx/starai_isaac_viola/smoke_test.sh 60'
```

## Notes

- The v1 cube grasp is a visual IK grasp demo. It attempts a deterministic pick-and-lift sequence and uses an attach/follow fallback after the gripper closes so the cube reliably lifts in WebRTC.
- This is not yet a validated pure friction/contact grasp.
- The Isaac Lab task variants register local Gym ids under `viola_lab_tasks` and use official Reach/Lift-Cube task configs with the Viola USD and joint names.
- Assets are included for reproducibility: `assets/viola.usd`, `assets/viola_isaac.urdf`, and `assets/configuration/*.usd`.
- ROS2 / MoveIt integration is not part of v1.
