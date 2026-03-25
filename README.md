🐍 AntiGravity Snake DQN
A Deep Q-Network (DQN) agent trained to master Snake in a dynamic, gravity-defying environment.

🚀 Overview

Unlike the classic Snake game, AntiGravity Snake introduces a constantly shifting gravity field that changes every 20 steps. The snake is pulled in one of five directions:

⬆️ Up
⬇️ Down
⬅️ Left
➡️ Right
🌀 Zero Gravity

The agent must learn not only how to reach food efficiently but also how to adapt to or exploit gravity drift.

🧠 Key Features
⚡ Dynamic Environment with changing physics
🤖 Deep Q-Learning (DQN) based agent
🎯 Reward shaping for better learning
📊 Training visualization tools
🎮 Playable AI simulation
🧩 Game Mechanics
🌍 Gravity Drift
Each timestep applies an extra movement in the current gravity direction
Forces strategic planning under dynamic physics
📡 State Representation

The agent observes:

Snake position
Relative food location
Current gravity vector
Immediate danger
🏆 Reward System
Action	Reward
Eat food	+10
Collision	-10
Move toward food	+1
Move away from food	-1
Move with gravity	+0.5
Move against gravity	-0.5
📁 Project Structure
AntiGravity-Snake-DQN/
│
├── env/               # Game logic + gravity system
├── agent/             # DQN model and agent
├── train.py           # Training loop
├── evaluate.py        # Run trained agent
├── plot_results.py    # Metrics visualization
├── config.py          # Hyperparameters
└── README.md
⚙️ Installation
pip install pygame torch numpy matplotlib
▶️ Usage
🏋️ Train
python train.py
📊 Visualize
python plot_results.py
🎮 Evaluate
python evaluate.py
📈 Learning Highlights
Learns to adapt under changing physics
Balances goal-seeking with environmental constraints
Exploits gravity for efficient movement
🛠️ Tech Stack
Python
PyTorch
Pygame
NumPy
Matplotlib
👨‍💻 Owner

Vedant Dubey

🔗 GitHub: https://github.com/vedantdubey19


Passionate about AI, Machine Learning, and building intelligent systems 🚀

🤝 Contributing

Contributions are welcome! Feel free to fork and submit a PR.
