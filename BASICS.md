# Example of recording a dataset with lerobot
```
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0921581 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --robot.id=black \
    --dataset.repo_id=HelloCephalopod/block_pickup_17 \
    --dataset.num_episodes=10 \
    --dataset.single_task="Place the block in the bowl" \
    --dataset.reset_time_s=7 \
    --dataset.episode_time_s=15 \
    --dataset.tags='["so100","tutorial"]' \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58FA0919271 \
    --teleop.id=blue \
    --display_data=true \
    --resume=true
```

# Example of evaluating a dataset (basically, you add the policy.path at the bottom)
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0921581 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --robot.id=black \
    --dataset.single_task="Place the block in the bowl" \
    --dataset.repo_id=HelloCephalopod/eval_block_pickup_40 \
    --dataset.episode_time_s=15 \
    --dataset.num_episodes=1 \
    --dataset.push_to_hub=false \
    --display_data=true \
    --resume=false \
    --policy.path=downloaded_models/pi0/local_my_pi0


python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0921581 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --robot.id=black \
    --dataset.single_task="Place the block in the bowl" \
    --dataset.episode_time_s=15 \
    --dataset.num_episodes=1 \
    --dataset.push_to_hub=false \
    --display_data=true \
    --resume=false \
    --policy.path=downloaded_models/my_smolvla_local/

# Evaluate a locally finetuned GR00T policy
# First start the GR00T inference server:
python isaac-gr00t/scripts/inference_service.py --server --model_path ./downloaded_models/gr00t/checkpoints/gr00t-n1.5-so100-finetuned --embodiment-tag new_embodiment --data-config so100 --denoising-steps 4

# Then run the evaluation client:
python isaac-gr00t/getting_started/examples/eval_lerobot.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0921581 \
    --robot.id=black \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --policy_host=localhost \
    --policy_port=5555 \
    --lang_instruction="Place the block in the bowl" \
    --action_horizon=8

# 🚀 OPTIMIZED: Direct GR00T inference (RECOMMENDED!)
# This bypasses the server-client architecture for better performance
# It also automatically handles camera resolution mismatches
python so100_inference_optimized.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0921581 \
    --robot.id=black \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --model_path=./downloaded_models/gr00t/checkpoints/gr00t-n1.5-so100-finetuned \
    --embodiment_tag=new_embodiment \
    --data_config=so100 \
    --denoising_steps=2 \
    --lang_instruction="Place the block in the bowl" \
    --action_horizon=4

# 🎯 GENERAL EVALUATION: LeRobot Policy Evaluation (RECOMMENDED!)
# This is a clean evaluation script that works with any LeRobot policy without creating a dataset
# Compatible with pi0, gr00t, and other policies - no server needed!
python lerobot_eval.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0921581 \
    --robot.id=black \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --policy.path=downloaded_models/pi0/local_my_pi0 \
    --task="Place the block in the bowl" \
    --episode_time_s=30 \
    --num_episodes=3 \
    --display_data=true

# Note: If your camera doesn't support 1280x720, you can use any resolution
# The optimized script will automatically resize to match the model's expectations
# Example with different camera resolution:
# --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}"

# 🚀🎯 BUFFERED: Action buffering (experimental - has issues!)
# ⚠️  NOTE: This version can cause confusion due to temporal mismatches
# ⚠️  The robot executes actions computed on old observations, causing oscillation
# ⚠️  Use the OPTIMIZED version above instead for best results
# python so100_inference_buffered.py \
#     --robot.type=so100_follower \
#     --robot.port=/dev/tty.usbmodem58FA0921581 \
#     --robot.id=black \
#     --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
#     --model_path=./downloaded_models/gr00t/checkpoints/gr00t-n1.5-so100-finetuned \
#     --embodiment_tag=new_embodiment \
#     --data_config=so100 \
#     --denoising_steps=2 \
#     --lang_instruction="Place the block in the bowl" \
#     --action_horizon=4 \
#     --buffer_size=5 \
#     --execution_fps=50 \
#     --smoothing_alpha=0.1 \
#     --enable_smoothing=true

python -m lerobot.teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0921581 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58FA0919271 \
    --teleop.id=blue \
    --display_data=true