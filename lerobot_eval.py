#!/usr/bin/env python3

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluates a policy on a robot without creating a dataset.

Example usage:

```shell
python lerobot_eval.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0921581 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --robot.id=black \
    --policy.path=downloaded_models/pi0/local_my_pi0 \
    --task="Place the block in the bowl" \
    --episode_time_s=30 \
    --num_episodes=3 \
    --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import numpy as np
import rerun as rr

from lerobot.common.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.policies.factory import make_policy, get_policy_class
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.common.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class EvaluationConfig:
    robot: RobotConfig
    # Policy will be loaded from path in __post_init__
    policy: PreTrainedConfig | None = None
    # Task description for the policy
    task: str = "Complete the task"
    # Number of seconds for each evaluation episode
    episode_time_s: int | float = 60
    # Number of episodes to evaluate
    num_episodes: int = 1
    # FPS for control loop
    fps: int = 30
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events
    play_sounds: bool = True
    # Reset time between episodes (manual reset)
    reset_time_s: int | float = 30
    # Optional teleoperator for manual intervention
    teleop: TeleoperatorConfig | None = None

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("A policy must be provided for evaluation")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def evaluation_loop(
    robot: Robot,
    policy: PreTrainedPolicy,
    events: dict,
    fps: int,
    control_time_s: int | float,
    task: str,
    obs_features: dict,
    display_data: bool = False,
    teleop: Teleoperator | None = None,
):
    """
    Run a single evaluation episode.
    """
    if policy is not None:
        policy.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        observation = robot.get_observation()

        if policy is not None:
            # Build proper dataset frame from robot observation
            observation_frame = build_dataset_frame(obs_features, observation, prefix="observation")
            
            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        elif teleop is not None:
            # Allow manual intervention during evaluation
            action = teleop.get_action()
        else:
            logging.info("No policy or teleoperator available, skipping action generation.")
            continue

        # Send action to robot
        sent_action = robot.send_action(action)

        if display_data:
            # Log observation data to rerun
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation.{obs}", rr.Scalar(val))
                elif hasattr(val, 'shape'):  # numpy array or torch tensor
                    rr.log(f"observation.{obs}", rr.Image(val), static=True)
            
            # Log action data to rerun
            for act, val in sent_action.items():
                if isinstance(val, float):
                    rr.log(f"action.{act}", rr.Scalar(val))

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def evaluate(cfg: EvaluationConfig):
    """
    Main evaluation function.
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    if cfg.display_data:
        _init_rerun(session_name="policy_evaluation")

    # Initialize robot
    robot = make_robot_from_config(cfg.robot)
    
    # Initialize optional teleoperator
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Build dataset features for proper observation formatting
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=False)

    # Load policy directly using from_pretrained for evaluation
    policy_cls = get_policy_class(cfg.policy.type)
    policy = policy_cls.from_pretrained(cfg.policy.pretrained_path)
    policy.to(cfg.policy.device)

    # Connect to robot
    robot.connect()
    if teleop is not None:
        teleop.connect()

    # Initialize keyboard listener for manual control
    listener, events = init_keyboard_listener()

    try:
        # Run evaluation episodes
        for episode_idx in range(cfg.num_episodes):
            log_say(f"Starting evaluation episode {episode_idx + 1}/{cfg.num_episodes}", cfg.play_sounds)
            
            evaluation_loop(
                robot=robot,
                policy=policy,
                events=events,
                fps=cfg.fps,
                control_time_s=cfg.episode_time_s,
                task=cfg.task,
                obs_features=obs_features,
                display_data=cfg.display_data,
                teleop=teleop,
            )
            
            # Reset environment between episodes (except for the last one)
            if episode_idx < cfg.num_episodes - 1 and not events["stop_recording"]:
                log_say("Please reset the environment for the next episode", cfg.play_sounds)
                
                # Wait for manual reset
                evaluation_loop(
                    robot=robot,
                    policy=None,  # No policy during reset
                    events=events,
                    fps=cfg.fps,
                    control_time_s=cfg.reset_time_s,
                    task=cfg.task,
                    obs_features=obs_features,
                    display_data=cfg.display_data,
                    teleop=teleop,
                )

            if events["stop_recording"]:
                break

        log_say("Evaluation completed", cfg.play_sounds, blocking=True)

    finally:
        # Cleanup
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()

        if not is_headless() and listener is not None:
            listener.stop()

        log_say("Exiting", cfg.play_sounds)


if __name__ == "__main__":
    evaluate() 