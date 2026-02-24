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

### Model-Free Algorithms
- LQR-FLM

### Task 
- Inverted Pendulum
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
- gymnasium: 1.2.2
- mujoco: 3.4.0



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



### Plot Reward Curve (example: PPO)
```bash
python plot.py --algo ppo
```



### Play a Trained Agent (example: PPO)
```bash
python play.py --algo ppo
```
---

##  References

- Gymnasium (Basic Usage)  
  https://gymnasium.farama.org/introduction/basic_usage/

- Reinforcement-Learning-Book-Revision  
  https://github.com/pasus/Reinforcement-Learning-Book-Revision

- 수학으로 풀어보는 강화학습 원리와 알고리즘 
