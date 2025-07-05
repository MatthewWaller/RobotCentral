#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SO100 Robot Inference Script for GR00T models.

This script provides a specialized way to run inference on a finetuned GR00T model
with a real SO100 robot and camera, similar to how LeRobot handles SO100 integration.

The SO100 robot has:
- 6 motors: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
- 1 camera: webcam (single camera setup)
- State: single_arm (5 joints) + gripper (1 joint) = 6 total
- Actions: single_arm (5 joints) + gripper (1 joint) = 6 total

Usage examples:

1. Run inference with real SO100 robot and camera:
   python so100_inference.py \
       --model_path ./downloaded_models/gr00t/checkpoints/gr00t-n1.5-so100-finetuned \
       --robot_port /dev/tty.usbmodem123456789 \
       --camera_index 0 \
       --language_instruction "Pick up the red block"

2. Run episode with safety limits:
   python so100_inference.py \
       --model_path ./downloaded_models/gr00t/checkpoints/gr00t-n1.5-so100-finetuned \
       --robot_port /dev/tty.usbmodem123456789 \
       --camera_index 0 \
       --run_episode \
       --max_steps 100 \
       --max_relative_target 10 \
       --language_instruction "Pick up the block and place it in the bowl"
"""

import argparse
import logging
import time
import numpy as np
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SO100Camera:
    """Camera interface for SO100 robot."""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.camera = None
        
    def connect(self):
        """Connect to the camera."""
        try:
            import cv2
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(f"Could not open camera at index {self.camera_index}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"Camera connected successfully at index {self.camera_index}")
            logger.info(f"Resolution: {self.width}x{self.height}")
            
        except ImportError:
            raise ImportError("OpenCV (cv2) is required for camera support. Install with: pip install opencv-python")
    
    def get_image(self) -> np.ndarray:
        """Get image from camera."""
        if self.camera is None:
            raise RuntimeError("Camera not connected. Call connect() first.")
        
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")
        
        # Convert BGR to RGB
        frame_rgb = frame[:, :, ::-1]
        return frame_rgb
    
    def disconnect(self):
        """Disconnect from camera."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None


class RealSO100Robot:
    """Real SO100 robot interface using LeRobot."""
    
    def __init__(self, port: str, max_relative_target: Optional[int] = None):
        self.port = port
        self.max_relative_target = max_relative_target
        self.robot = None
        
    def connect(self, calibrate: bool = True):
        """Connect to the real SO100 robot."""
        try:
            from lerobot.common.robots.so100_follower import SO100Follower
            from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
            from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
            
            # Create robot configuration
            config = SO100FollowerConfig(
                port=self.port,
                max_relative_target=self.max_relative_target,
                cameras={
                    "webcam": OpenCVCameraConfig(
                        index_or_path=0,  # Will be overridden by camera class
                        fps=30,
                        width=640,
                        height=480,
                    )
                }
            )
            
            self.robot = SO100Follower(config)
            self.robot.connect(calibrate=calibrate)
            
            logger.info(f"Real SO100 robot connected on port {self.port}")
            
        except ImportError:
            raise ImportError("LeRobot is required for real SO100 robot support. Install with: pip install lerobot")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to SO100 robot: {e}")
    
    def get_observation(self) -> Dict[str, Any]:
        """Get real robot state."""
        if self.robot is None:
            raise RuntimeError("Robot not connected. Call connect() first.")
        
        return self.robot.get_observation()
    
    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Send action to real robot."""
        if self.robot is None:
            raise RuntimeError("Robot not connected. Call connect() first.")
        
        return self.robot.send_action(action)
    
    def disconnect(self):
        """Disconnect from robot."""
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None


class SO100InferenceRunner:
    """Main class for running SO100 robot inference with GR00T models."""
    
    def __init__(
        self,
        model_path: str,
        robot_port: Optional[str] = None,
        camera_index: int = 0,
        max_relative_target: Optional[int] = None,
        denoising_steps: Optional[int] = None,
        device: str = "cuda",
        language_instruction: str = "Please perform the task as instructed.",
        mock_robot: bool = False,
    ):
        """
        Initialize the SO100 inference runner.
        
        Args:
            model_path: Path to the model checkpoint or HuggingFace model ID
            robot_port: Serial port for SO100 robot (e.g., "/dev/tty.usbmodem123456789")
            camera_index: Camera index for webcam
            max_relative_target: Maximum relative target movement for safety
            denoising_steps: Number of denoising steps (optional)
            device: Device to run inference on
            language_instruction: Language instruction for the task
            mock_robot: Whether to use mock robot (for testing without hardware)
        """
        self.model_path = model_path
        self.robot_port = robot_port
        self.camera_index = camera_index
        self.max_relative_target = max_relative_target
        self.denoising_steps = denoising_steps
        self.device = device
        self.language_instruction = language_instruction
        self.mock_robot = mock_robot
        
        # Initialize components
        self._setup_policy()
        self._setup_robot()
        self._setup_camera()
    
    def _setup_policy(self):
        """Setup the GR00T policy for SO100."""
        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"Device: {self.device}")
        
        # Use SO100 data configuration
        data_config = DATA_CONFIG_MAP["so100"]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        
        # Create policy
        self.policy = Gr00tPolicy(
            model_path=self.model_path,
            embodiment_tag="new_embodiment",  # For finetuned models
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=self.denoising_steps,
            device=self.device,
        )
        
        logger.info("GR00T policy loaded successfully!")
        logger.info(f"Expected video key: {data_config.video_keys}")
        logger.info(f"Expected state keys: {data_config.state_keys}")
        logger.info(f"Expected action keys: {data_config.action_keys}")
    
    def _setup_robot(self):
        if self.robot_port is None:
            raise ValueError("robot_port must be specified when not using mock_robot")
            
        self.robot = RealSO100Robot(
            port=self.robot_port,
            max_relative_target=self.max_relative_target
        )
        logger.info(f"Using real SO100 robot on port {self.robot_port}")
    
    def _setup_camera(self):
        """Setup the camera."""
        self.camera = SO100Camera(camera_index=self.camera_index)
        logger.info(f"Camera setup for index {self.camera_index}")
    
    def connect(self):
        """Connect to robot and camera."""
        logger.info("Connecting to robot and camera...")
        
        # Connect robot
        self.robot.connect(calibrate=True)
        
        # Connect camera
        self.camera.connect()
        
        logger.info("All components connected successfully!")
    
    def create_observations(self) -> Dict[str, Any]:
        """Create observations dictionary from camera and robot state."""
        # Get camera image
        image = self.camera.get_image()
        
        # Get robot state
        robot_state = self.robot.get_observation()
        
        # Create observations dictionary for SO100
        observations = {
            # Video observation (single camera)
            "video.webcam": image[np.newaxis, :, :, :],  # Add batch dimension
            
            # State observations
            "state.single_arm": np.array([
                robot_state["shoulder_pan.pos"],
                robot_state["shoulder_lift.pos"],
                robot_state["elbow_flex.pos"],
                robot_state["wrist_flex.pos"],
                robot_state["wrist_roll.pos"],
            ], dtype=np.float64)[np.newaxis, :],  # Add batch dimension
            
            "state.gripper": np.array([
                robot_state["gripper.pos"]
            ], dtype=np.float64)[np.newaxis, :],  # Add batch dimension
            
            # Language instruction
            "annotation.human.task_description": [self.language_instruction]
        }
        
        return observations
    
    def run_single_inference(self) -> Dict[str, Any]:
        """Run a single inference step."""
        # Create observations
        observations = self.create_observations()
        
        # Print observation structure (only for first run)
        if not hasattr(self, '_printed_obs_structure'):
            logger.info("Observation structure:")
            for key, value in observations.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    logger.info(f"  {key}: {type(value)}")
            self._printed_obs_structure = True
        
        # Run inference
        start_time = time.time()
        with torch.inference_mode():
            actions = self.policy.get_action(observations)
        inference_time = time.time() - start_time
        
        logger.info(f"Inference completed in {inference_time:.3f}s")
        
        # Log action details
        logger.info("Predicted actions:")
        for key, value in actions.items():
            if isinstance(value, np.ndarray):
                logger.info(f"  {key}: {value.shape} ({value.dtype})")
                if value.size <= 10:  # Print small arrays
                    logger.info(f"    Values: {value.flatten()}")
            else:
                logger.info(f"  {key}: {type(value)}")
        
        return actions
    
    def execute_actions(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actions on the robot."""
        # Convert GR00T action format to SO100 format
        so100_actions = {}
        
        if "action.single_arm" in actions:
            arm_action = actions["action.single_arm"].flatten()
            so100_actions.update({
                "shoulder_pan.pos": float(arm_action[0]),
                "shoulder_lift.pos": float(arm_action[1]),
                "elbow_flex.pos": float(arm_action[2]),
                "wrist_flex.pos": float(arm_action[3]),
                "wrist_roll.pos": float(arm_action[4]),
            })
        
        if "action.gripper" in actions:
            gripper_action = actions["action.gripper"].flatten()
            so100_actions["gripper.pos"] = float(gripper_action[0])
        
        # Send actions to robot
        executed_actions = self.robot.send_action(so100_actions)
        
        logger.info(f"Executed actions: {executed_actions}")
        return executed_actions
    
    def run_episode(self, max_steps: int = 100, step_delay: float = 0.1):
        """Run inference for multiple steps (episode)."""
        logger.info(f"Starting episode with max {max_steps} steps...")
        logger.info(f"Language instruction: {self.language_instruction}")
        
        episode_actions = []
        
        for step in range(max_steps):
            logger.info(f"--- Step {step + 1}/{max_steps} ---")
            
            try:
                # Run inference
                actions = self.run_single_inference()
                episode_actions.append(actions)
                
                # Execute actions on robot
                executed_actions = self.execute_actions(actions)
                
                # Wait before next step
                if step_delay > 0:
                    time.sleep(step_delay)
                    
            except KeyboardInterrupt:
                logger.info("Episode interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                break
        
        logger.info(f"Episode completed. Total steps: {len(episode_actions)}")
        return episode_actions
    
    def disconnect(self):
        """Disconnect from robot and camera."""
        if hasattr(self, 'robot'):
            self.robot.disconnect()
        
        if hasattr(self, 'camera'):
            self.camera.disconnect()
        
        logger.info("All components disconnected")


def main():
    parser = argparse.ArgumentParser(description="SO100 robot inference script for GR00T models")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint or HuggingFace model ID"
    )
    parser.add_argument(
        "--robot_port",
        type=str,
        default=None,
        help="Serial port for SO100 robot (e.g., /dev/tty.usbmodem123456789)"
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Camera index for webcam"
    )
    parser.add_argument(
        "--max_relative_target",
        type=int,
        default=None,
        help="Maximum relative target movement for safety (degrees)"
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=None,
        help="Number of denoising steps (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default="Please perform the task as instructed.",
        help="Language instruction for the task"
    )
    parser.add_argument(
        "--mock_robot",
        action="store_true",
        help="Use mock robot (for testing without hardware)"
    )
    parser.add_argument(
        "--run_episode",
        action="store_true",
        help="Run inference for multiple steps (episode)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum number of steps for episode mode"
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.1,
        help="Delay between steps in episode mode (seconds)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save episode actions to a numpy file (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.mock_robot and args.robot_port is None:
        parser.error("--robot_port is required when not using --mock_robot")
    
    # Create inference runner
    runner = SO100InferenceRunner(
        model_path=args.model_path,
        robot_port=args.robot_port,
        camera_index=args.camera_index,
        max_relative_target=args.max_relative_target,
        denoising_steps=args.denoising_steps,
        device=args.device,
        language_instruction=args.language_instruction,
        mock_robot=args.mock_robot,
    )
    
    try:
        # Connect to hardware
        runner.connect()
        
        if args.run_episode:
            # Run episode
            episode_actions = runner.run_episode(
                max_steps=args.max_steps,
                step_delay=args.step_delay
            )
            
            # Save results if requested
            if args.output_file:
                np.save(args.output_file, episode_actions)
                logger.info(f"Episode actions saved to: {args.output_file}")
        else:
            # Run single inference
            actions = runner.run_single_inference()
            
            # Execute actions
            executed_actions = runner.execute_actions(actions)
            
            # Save results if requested
            if args.output_file:
                np.save(args.output_file, actions)
                logger.info(f"Actions saved to: {args.output_file}")
    
    finally:
        # Cleanup
        runner.disconnect()


if __name__ == "__main__":
    main() 