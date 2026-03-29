import gymnasium as gym 
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os 

def train_agent(total_timesteps : int = 5000,
                path : str = "PPOSimpleCorridor"):
    
    """
    Train a PPO agent on SimpleCorridor-v0.
    
    Args:
        total_timesteps: Number of environment steps to train
        model_path: Path to save trained model
        
    Returns:
        Trained PPO model
    """
        
    log_dir="./logs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    env=gym.make("SimpleCorridor-v0")
    env=Monitor(env,log_dir)

    model=PPO("MlpPolicy", env, verbose=1)            # creating a model/agent
    model.learn(total_timesteps)
    model.save(path)

    env.close() # close to clean up 
    print(f" Model was saved to {path}")
    return model

if __name__ == "__main__":
    train_agent()