#!/usr/bin/env python3

"""
Optimized GR00T inference script that bypasses server-client architecture
for faster action generation by keeping the model directly in memory.
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from pathlib import Path

import draccus
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.utils.utils import init_logging, log_say

# Import GR00T components directly
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


class OptimizedGr00tRobotClient:
    """Direct GR00T policy inference client without server overhead"""

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "new_embodiment",
        data_config: str = "so100",
        denoising_steps: int = 2,  # Reduced from 4 for speed
        camera_keys: list = None,
        robot_state_keys: list = None,
        show_images: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the optimized GR00T client with direct model loading.
        
        Args:
            model_path: Path to the GR00T model
            embodiment_tag: Embodiment tag for the model
            data_config: Data configuration name
            denoising_steps: Number of denoising steps (reduced for speed)
            camera_keys: List of camera keys
            robot_state_keys: List of robot state keys
            show_images: Whether to show images
            device: Device to run the model on
        """
        self.camera_keys = camera_keys or []
        self.robot_state_keys = robot_state_keys or []
        self.show_images = show_images
        self.device = device
        
        print(f"Loading GR00T model directly in memory...")
        print(f"Model path: {model_path}")
        print(f"Denoising steps: {denoising_steps}")
        print(f"Device: {device}")
        
        # Load data config
        data_config_obj = DATA_CONFIG_MAP[data_config]
        modality_config = data_config_obj.modality_config()
        modality_transform = data_config_obj.transform()
        
        # Initialize policy directly (no server)
        self.policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
            device=device,
        )
        
        # Pre-warm the model with a dummy inference
        self._warm_up_model()
        
        print("✅ Model loaded and warmed up successfully!")
        
    def _warm_up_model(self):
        """Warm up the model with a dummy inference to avoid cold start penalty"""
        print("Warming up model...")
        
        # Get expected resolution for warm-up
        expected_resolution = self._get_expected_resolution()
        width, height = expected_resolution
        
        dummy_obs = {
            "video.webcam": np.zeros((1, height, width, 3), dtype=np.uint8),
            "state.single_arm": np.zeros((1, 5), dtype=np.float64),
            "state.gripper": np.zeros((1, 1), dtype=np.float64),
            "annotation.human.task_description": ["dummy task"],
        }
        
        # Run a dummy inference to warm up
        start_time = time.time()
        _ = self.policy.get_action(dummy_obs)
        warmup_time = time.time() - start_time
        print(f"Model warm-up completed in {warmup_time:.3f}s")

    def get_action(self, observation_dict: dict, lang: str) -> list:
        """
        Get actions directly from the model without server communication.
        
        Args:
            observation_dict: Robot observation dictionary
            lang: Language instruction
            
        Returns:
            List of action dictionaries
        """
        # Map camera keys to expected format
        obs_dict = {}
        
        # Use the first camera as the main camera
        if self.camera_keys:
            main_camera_key = self.camera_keys[0]
            camera_frame = observation_dict[main_camera_key]
            
            # Auto-resize camera input to match model expectations
            # Get expected resolution from model metadata
            expected_resolution = self._get_expected_resolution()
            current_resolution = (camera_frame.shape[1], camera_frame.shape[0])  # (width, height)
            
            if current_resolution != expected_resolution:
                print(f"🔄 Resizing camera input from {current_resolution} to {expected_resolution}")
                # Resize using OpenCV
                resized_frame = cv2.resize(
                    camera_frame, 
                    expected_resolution, 
                    interpolation=cv2.INTER_LINEAR
                )
                obs_dict["video.webcam"] = resized_frame
            else:
                obs_dict["video.webcam"] = camera_frame
        
        # Show images if requested
        if self.show_images:
            self._view_img(obs_dict)
        
        # Prepare state
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        obs_dict["state.single_arm"] = state[:5].astype(np.float64)
        obs_dict["state.gripper"] = state[5:6].astype(np.float64)
        obs_dict["annotation.human.task_description"] = [lang]
        
        # Add batch dimension
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            elif isinstance(obs_dict[k], list):
                pass  # Already in correct format
        
        # Direct model inference (no server communication)
        inference_start = time.time()
        action_chunk = self.policy.get_action(obs_dict)
        inference_time = time.time() - inference_start
        
        # Convert to lerobot format
        lerobot_actions = []
        for i in range(16):  # GR00T generates 16 actions
            action_dict = self._convert_to_lerobot_action(action_chunk, i)
            lerobot_actions.append(action_dict)
            
        print(f"⚡ Direct inference time: {inference_time:.3f}s")
        return lerobot_actions

    def _get_expected_resolution(self) -> tuple[int, int]:
        """Get the expected video resolution from model metadata"""
        try:
            # Try to get resolution from policy's modality transform
            modality_transform = self.policy.modality_transform
            if hasattr(modality_transform, '_original_resolutions') and modality_transform._original_resolutions:
                # Get the first video resolution (should be video.webcam)
                for key, resolution in modality_transform._original_resolutions.items():
                    if 'video' in key:
                        return resolution
            
            # Fallback to common GR00T resolution
            return (1280, 720)
        except Exception as e:
            print(f"⚠️  Could not determine expected resolution from metadata: {e}")
            print("Using default resolution (1280, 720)")
            return (1280, 720)

    def _convert_to_lerobot_action(self, action_chunk: dict, idx: int) -> dict:
        """Convert action chunk to lerobot format"""
        modality_keys = ["single_arm", "gripper"]
        concat_action = np.concatenate(
            [np.atleast_1d(action_chunk[f"action.{key}"][idx]) for key in modality_keys],
            axis=0,
        )
        return {key: concat_action[i] for i, key in enumerate(self.robot_state_keys)}

    def _view_img(self, img_dict: dict):
        """Display camera view"""
        if isinstance(img_dict, dict):
            img = np.concatenate([img_dict[k] for k in img_dict], axis=1)
        else:
            img = img_dict
        
        plt.imshow(img)
        plt.title("Camera View")
        plt.axis("off")
        plt.pause(0.001)
        plt.clf()


@dataclass
class OptimizedEvalConfig:
    robot: RobotConfig
    model_path: str = "./downloaded_models/gr00t/checkpoints/gr00t-n1.5-so100-finetuned"
    embodiment_tag: str = "new_embodiment"
    data_config: str = "so100"
    denoising_steps: int = 2  # Reduced from 4 for speed
    action_horizon: int = 8
    lang_instruction: str = "Place the block in the bowl"
    play_sounds: bool = False
    timeout: int = 60
    show_images: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "mps"


@draccus.wrap()
def eval_optimized(cfg: OptimizedEvalConfig):
    """Run optimized evaluation with direct model loading"""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # Initialize robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    camera_keys = list(cfg.robot.cameras.keys())
    robot_state_keys = list(robot._motors_ft.keys())
    
    print(f"Camera keys: {camera_keys}")
    print(f"Robot state keys: {robot_state_keys}")
    
    log_say("Initializing robot", cfg.play_sounds, blocking=True)
    
    # Initialize optimized policy client
    policy = OptimizedGr00tRobotClient(
        model_path=cfg.model_path,
        embodiment_tag=cfg.embodiment_tag,
        data_config=cfg.data_config,
        denoising_steps=cfg.denoising_steps,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
        show_images=cfg.show_images,
        device=cfg.device,
    )
    
    log_say(f"Starting optimized evaluation with: {cfg.lang_instruction}", cfg.play_sounds, blocking=True)
    
    try:
        # Main evaluation loop
        loop_count = 0
        while True:
            loop_start = time.time()
            
            # Get observation
            observation_dict = robot.get_observation()
            
            # Get action chunk
            action_chunk = policy.get_action(observation_dict, cfg.lang_instruction)
            
            # Execute actions
            for i in range(cfg.action_horizon):
                action_dict = action_chunk[i]
                robot.send_action(action_dict)
                time.sleep(0.015)  # Small delay for action execution
                
            loop_time = time.time() - loop_start
            loop_count += 1
            
            print(f"🔄 Loop {loop_count}: {loop_time:.3f}s total")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping optimized evaluation...")
    finally:
        # Clean shutdown
        try:
            robot.disconnect()
            print("✅ Robot disconnected safely")
        except Exception as e:
            print(f"⚠️  Error during robot disconnect: {e}")
        print("✅ Optimized evaluation stopped")


if __name__ == "__main__":
    eval_optimized() 