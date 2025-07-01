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
    --dataset.repo_id=HelloCephalopod/eval_block_pickup_37 \
    --dataset.episode_time_s=15 \
    --dataset.num_episodes=1 \
    --dataset.push_to_hub=false \
    --display_data=true \
    --resume=false \
    --policy.path=downloaded_models/pi0/local_my_pi0

# Evaluate a locally finetuned GR00T policy
# First start the GR00T inference server:
python scripts/inference_service.py --server --model_path ./checkpoints/gr00t-n1.5-so100-finetuned --embodiment-tag new_embodiment --data-config so100 --denoising-steps 4

# Then run the evaluation client:
python getting_started/examples/eval_lerobot.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0921581 \
    --robot.id=black \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --policy_host=localhost \
    --policy_port=5555 \
    --lang_instruction="Place the block in the bowl" \
    --action_horizon=8