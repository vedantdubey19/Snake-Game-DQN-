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
🎯 Reward Shaping for smarter learning
📊 Training visualization and evaluation tools
🎮 Playable AI simulation
🧩 Game Mechanics
🌍 Gravity Drift
Each timestep applies an extra movement in the current gravity direction.
Forces the agent to think ahead and adjust its trajectory.
📡 State Representation

The agent observes:

Snake position
Relative position of food
Current gravity vector
Immediate danger (collision awareness)
🏆 Reward System
Action	Reward
Eat food	+10
Collision (game over)	-10
Move toward food	+1
Move away from food	-1
Move with gravity	+0.5
Move against gravity	-0.5
📁 Project Structure
AntiGravity-Snake-DQN/
│
├── env/               # Snake environment + gravity mechanics
├── agent/             # DQN model and training logic
├── train.py           # Training loop
├── evaluate.py        # Run trained agent
├── plot_results.py    # Training visualization
├── config.py          # Hyperparameters & constants
└── README.md
⚙️ Installation

Make sure you have Python 3.8+ installed, then run:

pip install pygame torch numpy matplotlib
▶️ Usage
🏋️ Train the Agent
python train.py
📊 Visualize Training
python plot_results.py
🎮 Watch the Agent Play
python evaluate.py
📈 Learning Highlights
Learns to balance goal-seeking and physics adaptation
Develops strategies to ride gravity efficiently
Avoids dangerous moves influenced by drift
🛠️ Tech Stack
Python 🐍
PyTorch 🔥
Pygame 🎮
NumPy ➕
Matplotlib 📊
🌟 Future Improvements
Double DQN / Dueling DQN
Prioritized Experience Replay
Curriculum Learning for gravity complexity
Web-based visualization
🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

👨‍💻 Owner

Vedant Dubey