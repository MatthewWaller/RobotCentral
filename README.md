# RobotCentral
A place for software engineers to learn the basics of robotics.

### Welcome to your robotics journey.

These educations resources are dedicated to exploring robotics. Train your own robot! Have it do specific tasks!

You may not have any robotics hardware experience, and that is no problem. These resources are for you. If you know basic Python, that should be enough to get you started.

We'll have our first robot be one of the most versitile: the robotic arm.

## Getting Started

### Clone the Repository

This repository includes the [LeRobot](https://github.com/huggingface/lerobot) framework and [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) as submodules. To get everything set up properly:

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/yourusername/RobotCentral.git

# OR if you already cloned without --recursive, initialize submodules:
git submodule update --init --recursive
```

### What's Included

- **LeRobot**: State-of-the-art machine learning for robotics in PyTorch
- **NVIDIA Isaac-GR00T**: World's first open foundation model for generalized humanoid robot reasoning and skills
- **Educational resources**: Step-by-step guides for robotics learning
- **Examples**: Ready-to-run robotics code examples
- **Pre-trained models**: Access to trained robotic policies

### Prerequisites

- Python 3.8+
- Basic Python knowledge
- Some folks might use robotics simulation, but here, we're going to focus on real, live robots. We'll use the SO-100. Will probably migrate to SO-101, but SO-100 is what I have right now.