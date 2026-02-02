# Basic Studying: Basic Reinforcement Learning Algorithms

## 🎯 Goal

- Understanding basic reinforcement learning (RL) algorithms  
- Gaining intuition on **how RL algorithms actually work through implementation**
- Studying learning behavior and performance differences using **toy problems**
- Understanding the **separation of environment and agent**, following the design philosophy of **Gymnasium**

---

## ✅ Completed

### Model-Free Algorithms
- A2C
- PPO
- DDPG
- SAC

---

## 🛠 TODO

- Add more **model-free algorithms**
- Add **model-based algorithms**
- Add more **tasks**

---

##  Environment Setup

### Python Version
```
Python 3.10.14
```

### Core Libraries
- PyTorch 2.5.1 (CUDA 12.1)
- NumPy 2.2.6
- Matplotlib 3.9.1



### Install PyTorch (CUDA 12.1)
```bash
pip install torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

---


## How to Use

### Train an Agent (example: PPO)
```bash
python train.py --algo ppo
```

---

### Plot Reward Curve (example: PPO)
```bash
python plot.py --algo ppo
```

---

### Play a Trained Agent (example: PPO)
```bash
python play.py --algo ppo
```


