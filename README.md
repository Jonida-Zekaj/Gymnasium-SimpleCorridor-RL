# SimpleCorridor RL — Reinforcement Learning Project

A reinforcement learning project implementing a custom Gymnasium environment where an agent learns to navigate a corridor using the PPO (Proximal Policy Optimization) algorithm.

##  Table of Contents

- [Environment Overview](#-environment-overview)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dependencies](#-dependencies)
- [Running the Project](#-running-the-project)
- [Configuration](#-configuration)
- [Performance Metrics](#-performance-metrics)
- [Customization](#-customization)
- [Troubleshooting](#-troubleshooting)
- [File Descriptions](#-file-descriptions)
- [Learning Outcomes](#-learning-outcomes)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Author](#-author)
- [Support](#-support)

##  Environment Overview

**SimpleCorridor-v0** is a custom Gymnasium environment where:
- **Task**: Navigate from position 0 (left) to position 9 (right) in a 10-step corridor
- **Observations**: Current position (0-9)
- **Actions**: 0 = Move Left, 1 = Move Right
- **Rewards**: 
  - `+1.0` when reaching the goal
  - `-0.1` for each step taken
- **Rendering**: ASCII visualization of agent position and goal

##  Results

After training with PPO for **5000 timesteps**:
- **Success Rate**: 100%
- **Average Steps per Episode**: ~13.25
- **Episodes Evaluated**: 100

##  Project Structure

```
SimpleCorridor-RL/
├── env/
│   ├── __init__.py
│   └── corridor_env.py              # Custom SimpleCorridor environment
│
├── train_test/
│   ├── __init__.py
│   ├── train.py                     # PPO agent training
│   └── evaluate.py                  # Agent evaluation & metrics
│
├── visualization/
│   ├── __init__.py
│   ├── plot.py                      # Reward plotting
│   ├── video_recorder_imageio.py    # Video recording (imageio)
│   └── gym_wrapper_recorder.py      # Video recording (gym wrapper)
│
├── notebook/
│   └── SimpleCorridor_Walkthrough.ipynb  # Jupyter notebook walkthrough
│
├── configs/
│   └── ppo_config.yaml              # Configuration file
│
├── models/                          # Trained models (generated)
├── logs/                            # Training logs (generated)
├── videos/                          # Recorded videos (generated)
│
├── main.py                          # Main pipeline orchestration
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Jonida-Zekaj/Gymnasium-SimpleCorridor-RL.git
cd Gymnasium-SimpleCorridor-RL
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

- **gymnasium** (>=0.28.1) - RL environment framework
- **stable-baselines3** (>=2.0.0) - PPO algorithm
- **numpy** - Numerical computations
- **matplotlib** - Plotting
- **PyYAML** - Configuration management
- **pyglet** - Graphics rendering
- **moviepy** - Video processing
- **imageio** - Image/video I/O

## ▶️ Running the Project

### Full Pipeline

Execute the complete workflow (test → train → evaluate → plot):

```bash
python main.py
```

### Step-by-Step

Run individual components:

```python
# Just training
from train_test.train import train_agent
train_agent(total_timesteps=5000, path="models/PPOSimpleCorridor")

# Just evaluation
from train_test.evaluate import evaluate_agent
evaluate_agent("models/PPOSimpleCorridor")

# Just plotting
from visualization.plot import plot_rewards
plot_rewards("logs/")
```

## ⚙️ Configuration

Edit `configs/ppo_config.yaml` to customize:

```yaml
env_id: "SimpleCorridor-v0"          # Environment ID
log_dir: "logs/"                     # Training logs directory
model_path: "models/PPOSimpleCorridor"  # Where to save trained model
total_timesteps: 5000                # Training timesteps
video_file: "videos/simple_corridor_agent.mp4"
record_ep: 3                         # Episodes to record
```

## 📈 Performance Metrics

The evaluation script measures:
- **Success Rate**: % of episodes reaching the goal
- **Average Steps**: Mean steps per episode
- **Total Episodes**: Number of test episodes

Example output:
```
==================================================
Performance Evaluation - Trained Agent
==================================================
Success Rate: 100.00%
Average Steps per Episode: 13.250
Total Episodes: 100
Successful Episodes: 100
```

## 🔧 Customization

### Modify Environment Parameters

Update corridor parameters in `configs/ppo_config.yaml` and `env/corridor_env.py`

### Change Training Algorithm

Replace PPO with other stable-baselines3 algorithms:

```python
from stable_baselines3 import DQN, A2C
model = DQN("MlpPolicy", env)  # or A2C, PPO, etc.
```

### Adjust Training Duration

```yaml
total_timesteps: 10000  # Increase for longer training
```

## 🐛 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'gymnasium'`
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: `FileNotFoundError: models/ directory doesn't exist`
- **Solution**: Run `python main.py` which creates directories automatically

**Issue**: Plots not showing
- **Solution**: Ensure matplotlib backend is configured, or use `plt.savefig()`

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `env/corridor_env.py` | SimpleCorridor environment definition |
| `train_test/train.py` | PPO training script |
| `train_test/evaluate.py` | Evaluation metrics calculation |
| `visualization/plot.py` | Training reward visualization |
| `main.py` | Main pipeline orchestration |
| `configs/ppo_config.yaml` | Centralized configuration |

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Creating custom Gymnasium environments
- ✅ Training RL agents with PPO algorithm
- ✅ Evaluating agent performance
- ✅ Visualizing training progress
- ✅ Modular project structure for ML
- ✅ Configuration management with YAML

## 🔮 Future Improvements

- [ ] Add image-based rendering for video recording
- [ ] Implement curriculum learning (increasing corridor length)
- [ ] Multi-agent training
- [ ] Hyperparameter tuning & optimization
- [ ] Alternative algorithms (DQN, A3C)
- [ ] Web interface for visualization
- [ ] Tensorboard integration for training metrics

## 📄 License

This project is provided as-is for educational purposes.

## 👤 Author

Jonida Zekaj

## 📞 Support

For issues or questions, please create an issue in the repository.
