import gymnasium as gym 
import env.corridor_env 
from train_test.train import train_agent
from train_test.evaluate import evaluate_agent
from visualization.plot import plot_rewards
from visualization.video_recorder_imageio import create_video_imageio
# from visualization.gym_wrapper_recorder import create_video_gym_wrapper
import yaml 
import argparse 

# Load configuration from YAML file
with open('configs/ppo_config.yaml', 'r') as file:
   config =yaml.safe_load(file)

env_id = config['env_id']
log_dir = config['log_dir']
model_path = config['model_path']
video_file = config['video_file']
total_timesteps = config['total_timesteps']
record_ep = config['record_ep']


# Step 1 :: Test with untrained agent to estimate baseline performance and establish a comperative framework for later.

print("\n"+"="*50)
print("Performance Evaluation - Untrained Agent")
print("="*50)

env = gym.make(env_id)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)              # how can agent move
 
obs, _ = env.reset()
print("Initial position:", obs)
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, New position: {obs}, Reward: {reward}, Done: {done}")
    env.render()
    if done:
      print("The goal was reached!")
      break
env.close()


# Step 2 :: Train the agent using PPO algorithm.
train_agent(
   total_timesteps = total_timesteps,
   path = model_path
   )

# Step 3 :: Plot the rewards
plot_rewards(log_dir)

# Step 4 :: Evaluate the trained agent's performance
evaluate_agent(model_path)

# Step 5 :: Record a video
from stable_baselines3 import PPO

model = PPO.load(model_path)
video_env = gym.make(env_id)
create_video_imageio(video_env,
                     model,
                     filename = video_file,
                     num_episodes=record_ep,
                     )
video_env.close()

# Step 6 :: Display video
#from recording.colab_display import display_video_in_colab
#display_video_in_colab(VIDEO_FILE)
 
print("\nFinished!")
 

