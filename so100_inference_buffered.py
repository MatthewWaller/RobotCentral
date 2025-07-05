#!/usr/bin/env python3

"""
Buffered GR00T inference script with smooth action execution.
Uses action buffering, overlapping inference/execution, and action smoothing
for smoother robot control inspired by lerobot techniques.
"""

import logging
import time
import threading
from collections import deque
from dataclasses import asdict, dataclass
from pprint import pformat
from pathlib import Path
from queue import Queue, Empty
import platform

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


def busy_wait(seconds):
    """Precise timing function adapted from lerobot"""
    if seconds <= 0:
        return
    if platform.system() == "Darwin":
        # On Mac, time.sleep is not accurate, use busy wait
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        time.sleep(seconds)


class ActionBuffer:
    """Thread-safe circular buffer for storing action chunks"""
    
    def __init__(self, max_size: int = 10):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        
    def put(self, action_chunk: list):
        """Add an action chunk to the buffer"""
        with self.lock:
            self.buffer.append(action_chunk)
            self.not_empty.notify()
            
    def get(self) -> list:
        """Get the next action chunk from buffer, blocking if empty"""
        with self.not_empty:
            while len(self.buffer) == 0:
                self.not_empty.wait()
            return self.buffer.popleft()
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self.lock:
            return len(self.buffer) == 0


class BufferedGr00tRobotClient:
    """Buffered GR00T policy inference client with smooth execution"""

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "new_embodiment",
        data_config: str = "so100",
        denoising_steps: int = 2,
        camera_keys: list = None,
        robot_state_keys: list = None,
        show_images: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "mps",
        buffer_size: int = 5,
    ):
        self.camera_keys = camera_keys or []
        self.robot_state_keys = robot_state_keys or []
        self.show_images = show_images
        self.device = device
        
        print(f"Loading GR00T model for buffered inference...")
        print(f"Model path: {model_path}")
        print(f"Denoising steps: {denoising_steps}")
        print(f"Device: {device}")
        print(f"Buffer size: {buffer_size}")
        
        # Load data config
        data_config_obj = DATA_CONFIG_MAP[data_config]
        modality_config = data_config_obj.modality_config()
        modality_transform = data_config_obj.transform()
        
        # Initialize policy directly
        self.policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
            device=device,
        )
        
        # Initialize action buffer
        self.action_buffer = ActionBuffer(max_size=buffer_size)
        
        # Threading control
        self.inference_thread = None
        self.inference_running = False
        self.observation_queue = Queue(maxsize=2)  # Small queue for latest observations
        
        # Performance tracking
        self.inference_times = deque(maxlen=10)
        self.total_inferences = 0
        
        # Pre-warm the model
        self._warm_up_model()
        
        print("✅ Buffered GR00T client ready!")
        
    def _warm_up_model(self):
        """Warm up the model with a dummy inference"""
        print("Warming up model...")
        
        expected_resolution = self._get_expected_resolution()
        width, height = expected_resolution
        
        dummy_obs = {
            "video.webcam": np.zeros((1, height, width, 3), dtype=np.uint8),
            "state.single_arm": np.zeros((1, 5), dtype=np.float64),
            "state.gripper": np.zeros((1, 1), dtype=np.float64),
            "annotation.human.task_description": ["dummy task"],
        }
        
        start_time = time.time()
        _ = self.policy.get_action(dummy_obs)
        warmup_time = time.time() - start_time
        print(f"Model warm-up completed in {warmup_time:.3f}s")

    def _get_expected_resolution(self) -> tuple[int, int]:
        """Get expected video resolution from model metadata"""
        try:
            modality_transform = self.policy.modality_transform
            if hasattr(modality_transform, '_original_resolutions') and modality_transform._original_resolutions:
                for key, resolution in modality_transform._original_resolutions.items():
                    if 'video' in key:
                        return resolution
            return (1280, 720)
        except Exception as e:
            print(f"⚠️  Could not determine expected resolution: {e}")
            return (1280, 720)

    def start_inference_thread(self, lang_instruction: str):
        """Start the background inference thread"""
        self.inference_running = True
        self.lang_instruction = lang_instruction
        self.inference_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True
        )
        self.inference_thread.start()
        print("🧠 Background inference thread started")
        
    def stop_inference_thread(self):
        """Stop the background inference thread"""
        self.inference_running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        print("🛑 Background inference thread stopped")

    def _inference_worker(self):
        """Background worker that continuously generates action chunks"""
        while self.inference_running:
            try:
                # Get latest observation from queue (non-blocking)
                try:
                    observation_dict = self.observation_queue.get_nowait()
                except Empty:
                    time.sleep(0.01)  # Short sleep to avoid busy waiting
                    continue
                
                # Prepare observation for inference
                obs_dict = self._prepare_observation(observation_dict, self.lang_instruction)
                
                # Run inference
                inference_start = time.time()
                action_chunk = self.policy.get_action(obs_dict)
                inference_time = time.time() - inference_start
                
                # Convert to lerobot format
                lerobot_actions = []
                for i in range(16):  # GR00T generates 16 actions
                    action_dict = self._convert_to_lerobot_action(action_chunk, i)
                    lerobot_actions.append(action_dict)
                
                # Add to buffer
                self.action_buffer.put(lerobot_actions)
                
                # Track performance
                self.inference_times.append(inference_time)
                self.total_inferences += 1
                avg_inference_time = np.mean(self.inference_times)
                
                print(f"🧠 Inference #{self.total_inferences}: {inference_time:.3f}s "
                      f"(avg: {avg_inference_time:.3f}s, buffer: {self.action_buffer.size()})")
                
            except Exception as e:
                print(f"❌ Inference error: {e}")
                time.sleep(0.1)

    def _prepare_observation(self, observation_dict: dict, lang: str) -> dict:
        """Prepare observation for model inference"""
        obs_dict = {}
        
        # Handle camera input with auto-resize
        if self.camera_keys:
            main_camera_key = self.camera_keys[0]
            camera_frame = observation_dict[main_camera_key]
            
            expected_resolution = self._get_expected_resolution()
            current_resolution = (camera_frame.shape[1], camera_frame.shape[0])
            
            if current_resolution != expected_resolution:
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
        
        return obs_dict

    def queue_observation(self, observation_dict: dict):
        """Queue an observation for inference (non-blocking)"""
        try:
            # Only keep latest observation in queue
            while not self.observation_queue.empty():
                try:
                    self.observation_queue.get_nowait()
                except Empty:
                    break
            self.observation_queue.put_nowait(observation_dict)
        except:
            pass  # Queue full, skip this observation

    def get_buffered_actions(self) -> list:
        """Get the next action chunk from the buffer"""
        return self.action_buffer.get()

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


class ActionSmoother:
    """Exponential moving average action smoother like lerobot"""
    
    def __init__(self, alpha: float = 0.3, robot_state_keys: list = None):
        self.alpha = alpha  # Smoothing factor (0 = no change, 1 = no smoothing)
        self.robot_state_keys = robot_state_keys or []
        self.previous_action = None
        
    def smooth(self, action: dict) -> dict:
        """Apply exponential moving average smoothing"""
        if self.previous_action is None:
            self.previous_action = action.copy()
            return action
        
        smoothed_action = {}
        for key in self.robot_state_keys:
            # Exponential moving average: new = alpha * new + (1-alpha) * old
            smoothed_action[key] = (
                self.alpha * action[key] + 
                (1 - self.alpha) * self.previous_action[key]
            )
        
        self.previous_action = smoothed_action.copy()
        return smoothed_action


@dataclass
class BufferedEvalConfig:
    robot: RobotConfig
    model_path: str = "./downloaded_models/gr00t/checkpoints/gr00t-n1.5-so100-finetuned"
    embodiment_tag: str = "new_embodiment"
    data_config: str = "so100"
    denoising_steps: int = 2
    action_horizon: int = 4  # Reduced for more frequent action updates
    lang_instruction: str = "Place the block in the bowl"
    play_sounds: bool = False
    timeout: int = 60
    show_images: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "mps"
    
    # Buffering and smoothing parameters
    buffer_size: int = 5
    execution_fps: int = 50  # Actions per second for smooth execution (increased)
    smoothing_alpha: float = 0.1  # Action smoothing factor (0-1, lower = smoother)
    enable_smoothing: bool = True


@draccus.wrap()
def eval_buffered(cfg: BufferedEvalConfig):
    """Run buffered evaluation with smooth action execution"""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # Initialize robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    camera_keys = list(cfg.robot.cameras.keys())
    robot_state_keys = list(robot._motors_ft.keys())
    
    print(f"Camera keys: {camera_keys}")
    print(f"Robot state keys: {robot_state_keys}")
    
    log_say("Initializing robot for buffered execution", cfg.play_sounds, blocking=True)
    
    # Initialize buffered policy client
    policy = BufferedGr00tRobotClient(
        model_path=cfg.model_path,
        embodiment_tag=cfg.embodiment_tag,
        data_config=cfg.data_config,
        denoising_steps=cfg.denoising_steps,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
        show_images=cfg.show_images,
        device=cfg.device,
        buffer_size=cfg.buffer_size,
    )
    
    # Initialize action smoother
    smoother = None
    if cfg.enable_smoothing:
        smoother = ActionSmoother(alpha=cfg.smoothing_alpha, robot_state_keys=robot_state_keys)
        print(f"🎯 Action smoothing enabled (alpha={cfg.smoothing_alpha})")
    
    # Start inference thread
    policy.start_inference_thread(cfg.lang_instruction)
    
    log_say(f"Starting buffered evaluation: {cfg.lang_instruction}", cfg.play_sounds, blocking=True)
    
    try:
        # Give inference thread time to fill buffer
        print("⏳ Filling action buffer...")
        time.sleep(2.0)
        
        loop_count = 0
        action_step = 0
        current_action_chunk = None
        
        while True:
            loop_start = time.perf_counter()
            
            # Get observation and queue for inference
            observation_dict = robot.get_observation()
            policy.queue_observation(observation_dict)
            
            # Get new action chunk if needed
            if current_action_chunk is None or action_step >= cfg.action_horizon:
                current_action_chunk = policy.get_buffered_actions()
                action_step = 0
                print(f"📦 New action chunk received (buffer size: {policy.action_buffer.size()})")
            
            # Get current action from chunk
            raw_action = current_action_chunk[action_step]
            
            # Apply smoothing if enabled
            if smoother:
                action_dict = smoother.smooth(raw_action)
            else:
                action_dict = raw_action
            
            # Execute action
            robot.send_action(action_dict)
            action_step += 1
            
            # Timing and stats
            loop_time = time.perf_counter() - loop_start
            loop_count += 1
            
            # Control execution FPS
            target_dt = 1.0 / cfg.execution_fps
            dt_s = time.perf_counter() - loop_start
            busy_wait(target_dt - dt_s)
            
            actual_loop_time = time.perf_counter() - loop_start
            actual_fps = 1.0 / actual_loop_time if actual_loop_time > 0 else 0
            
            if loop_count % 10 == 0:  # Print stats every 10 loops
                avg_inference_time = np.mean(policy.inference_times) if policy.inference_times else 0
                print(f"🔄 Loop {loop_count}: {actual_fps:.1f} FPS, "
                      f"inference: {avg_inference_time:.3f}s, "
                      f"buffer: {policy.action_buffer.size()}")
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping buffered evaluation...")
    finally:
        # Clean shutdown
        policy.stop_inference_thread()
        robot.disconnect()
        print("✅ Buffered evaluation stopped")


if __name__ == "__main__":
    eval_buffered() 