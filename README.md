# F1Tenth_RL

This repository contains a reinforcement learning and ROS2 integration project for the F1TENTH autonomous racing platform.

## What is included

- `f1tenth_gym_ros_src/`
  - A ROS2 package that bridges `f1tenth_gym` into ROS2, including launch files, simulation configuration, map assets, and ROS publishers/subscribers.
  - Supports single- and two-agent racing setups using laser scan and Ackermann drive topics.
  - Includes containerized launch support via `Dockerfile` and `docker-compose.yml`.

- `Inference.py`
  - A ROS2 node that loads a Stable Baselines3 PPO model from a `.zip` file and publishes drive commands based on `/scan` input.

- `Engine_Inference.py`
  - A ROS2 TensorRT inference node that loads a serialized TensorRT engine and publishes drive commands from LiDAR input.

- `Conversion.py`
  - Exports a trained PPO policy to ONNX format for deployment and compatibility with optimized inference workflows.

- Model and engine artifacts
  - `2million_single_lidar.onnx`, `3million_robust_single_lidar.onnx`, `overtake_model.onnx`
  - `f1tenth_ppo_model.zip`, `f1tenth_ppo_model_2million.zip`, `f1tenth_ppo_overtake_model.zip`, `robust_f1tenth_3million.zip`

- `Demos/`
  - Example video recordings.
 

https://github.com/user-attachments/assets/dbbd1c31-a8f2-459d-b139-0b827e7736ee


https://github.com/user-attachments/assets/b92e9baf-3fbd-4521-b7ba-b5b728413431


https://github.com/user-attachments/assets/444fb01a-34ed-45e8-be6e-737dd16c61d7


https://github.com/user-attachments/assets/91ff4ca9-507b-4770-9c19-23fc28f4a0aa


## Main features

- ROS2 bridge for F1TENTH gym environment
- RL training and testing support for PPO agents
- Model conversion to ONNX
- TensorRT deployment example
- Containerized ROS2 simulation environment with display forwarding support

## How it is organized

- `f1tenth_gym_ros_src/`: ROS2 package source, launch files, maps, and package configuration.
- `Inference.py`: CPU-based PPO inference node.
- `Engine_Inference.py`: GPU-based TensorRT inference node.
- `Conversion.py`: ONNX export for a trained SB3 PPO model.
- `Demos/`: demonstration videos.

## Notes

- The ROS2 package inside `f1tenth_gym_ros_src/` already contains its own detailed README and package metadata.
- Use the ROS2 package for simulation launch and interaction.
- Use the inference scripts to run trained models with live `/scan` inputs and publish `/drive` commands.

## Suggested next steps

1. Review `f1tenth_gym_ros_src/README.md` for ROS2 simulation setup and launch instructions.
2. Verify your model file paths before running `Inference.py` or `Engine_Inference.py`.
3. Use `Conversion.py` when exporting a trained PPO model to ONNX.
