# 🎯 Smooth Motion Tuning Guide

If your robot motion is **choppy or has tiny spurts**, here's how to fix it:

## 🚀 Quick Fix (Try This First)

```bash
python so100_inference_buffered.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0921581 \
    --robot.id=black \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --lang_instruction="Place the block in the bowl" \
    --execution_fps=50 \
    --smoothing_alpha=0.1 \
    --action_horizon=4 \
    --enable_smoothing=true
```

## 🔧 Parameter Tuning

### **For ULTRA SMOOTH motion:**
- `--execution_fps=50` (higher FPS = smoother)
- `--smoothing_alpha=0.05` (lower = more smoothing)
- `--action_horizon=2` (more frequent action updates)

### **For RESPONSIVE motion:**
- `--execution_fps=30`
- `--smoothing_alpha=0.3`
- `--action_horizon=6`

### **For BALANCED motion:**
- `--execution_fps=40`
- `--smoothing_alpha=0.15`
- `--action_horizon=4`

## 🎛️ Parameter Explanations

| Parameter | What It Does | Choppy Problem | Solution |
|-----------|--------------|----------------|----------|
| `execution_fps` | Actions per second | Too low FPS causes jerky motion | Increase to 40-60 |
| `smoothing_alpha` | How much to smooth actions | High values cause sharp movements | Decrease to 0.05-0.2 |
| `action_horizon` | Actions per chunk | Long chunks cause pauses | Decrease to 2-4 |
| `buffer_size` | Action chunks ready | Too small causes delays | Keep at 5-8 |

## 🔍 Diagnosing Motion Issues

**Symptoms** → **Likely Cause** → **Fix**

- **Jerky, robotic motion** → Low FPS → Increase `execution_fps` to 50+
- **Sharp direction changes** → Not enough smoothing → Decrease `smoothing_alpha` to 0.1
- **Pauses between movements** → Long action chunks → Decrease `action_horizon` to 2-4
- **Delayed responses** → Empty buffer → Increase `buffer_size` to 8
- **Slow overall motion** → Over-smoothing → Increase `smoothing_alpha` to 0.2

## 🎯 Motion Quality Levels

### **Level 1: Basic (Choppy)**
```bash
--execution_fps=20 --smoothing_alpha=0.5 --action_horizon=8
```

### **Level 2: Good**
```bash
--execution_fps=30 --smoothing_alpha=0.3 --action_horizon=6
```

### **Level 3: Smooth (Recommended)**
```bash
--execution_fps=50 --smoothing_alpha=0.1 --action_horizon=4
```

### **Level 4: Ultra Smooth**
```bash
--execution_fps=60 --smoothing_alpha=0.05 --action_horizon=2
```

## ⚠️ Trade-offs

- **Higher FPS** = Smoother motion BUT more CPU usage
- **Lower smoothing_alpha** = Smoother motion BUT slower response to changes
- **Lower action_horizon** = More responsive BUT more frequent inference calls

## 🚨 If Still Choppy

Try this extreme smoothness setting:

```bash
python so100_inference_buffered.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA0921581 \
    --robot.id=black \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}}" \
    --lang_instruction="Place the block in the bowl" \
    --execution_fps=60 \
    --smoothing_alpha=0.05 \
    --action_horizon=2 \
    --buffer_size=8 \
    --enable_smoothing=true
```

This should eliminate all choppiness but may be slightly slower to respond to new situations. 