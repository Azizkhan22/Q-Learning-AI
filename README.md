# 🧠 Q-Learning Agent for FrozenLake (Gymnasium)

This project implements a **Q-learning**-based AI agent to solve the **FrozenLake-v1** environment from the **Gymnasium** library. The agent learns optimal actions to reach the goal without falling into holes, using tabular Q-learning.

## ❄️ Environment
- **Environment:** `FrozenLake-v1` from Gymnasium  
- **Mode:** Non-slippery (`is_slippery=False`) for deterministic behavior  
- **Render:** Visualized using `render_mode="human"`

## 🚀 Features
- Q-table initialized and updated using classic Q-learning  
- Epsilon-greedy exploration strategy  
- Training over 1000 episodes  
- Visual demonstration of the learned agent's performance

## 📦 Requirements
Install dependencies using pip:
```bash
pip install gymnasium pygame numpy
```

## 📁 Files
- `main.py` – Contains full training and visualization code

## ⚙️ Algorithm Details

| Parameter             | Value  |
|-----------------------|--------|
| Learning Rate (`α`)   | 0.9    |
| Discount Factor (`γ`) | 0.8    |
| Exploration Rate (`ε`)| 0.6    |
| Training Episodes     | 1000   |

## 🏃‍♂️ How to Run
Make sure you have all dependencies installed, then run the script:
```bash
python main.py
```

The script will:
1. Train the Q-learning agent on `FrozenLake-v1`.
2. Print the learned Q-table.
3. Run 3 episodes of the agent navigating the environment using the learned policy.
4. Render the agent's actions in a GUI window.

## 📊 Output
- Trained Q-table printed after training.
- Real-time environment visualization during test episodes.
- Terminal logs showing state, action, Q-values, and episode outcomes.

## 📷 Sample Output
```
Episode: 1
State: 0, Action: 1, Q-values: [0.  0.1 0.  0. ]
...
Reached the goal!
```

## 🧠 Concept
**Q-learning** is a reinforcement learning technique that learns the expected future rewards for each action in each state. The agent updates its Q-table using the Bellman equation during training.
