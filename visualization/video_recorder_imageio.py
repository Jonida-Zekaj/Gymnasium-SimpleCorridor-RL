import gymnasium as gym 
import imageio
import os

# Method 1: Using imageio to create MP4 video
def create_video_imageio(env, 
                         agent_or_policy, 
                         filename="agent_video.mp4", 
                         num_episodes=5, 
                         fps=30):
    """
    Create a video of the agent playing the game using imageio

    Args:
        env: The gymnasium environment
        agent_or_policy: Either a trained agent object or a policy function
        filename: Output video filename
        num_episodes: Number of episodes to record
        fps: Frames per second
    """
    frames = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_frames = []

        while not (done or truncated):
            # Get frame from environment
            frame = env.render()
            episode_frames.append(frame)
            action = get_action(env, agent_or_policy, obs)
            obs, reward, done, truncated, info = env.step(action)

        frames.extend(episode_frames)
        print(f"Episode {episode + 1} completed with {len(episode_frames)} frames")

    # Create directory if it doesn't exist
    video_dir = os.path.dirname(filename)
    if video_dir and not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)

    # Save video
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Video saved as {filename}")
    return filename

def get_action(env,
               agent_or_policy,
               obs):
    # Get action from your trained agent
    # Replace this with your agent's action selection method
    if hasattr(agent_or_policy, 'predict'):
        # For stable-baselines3 agents
        action, _ = agent_or_policy.predict(obs)
    elif hasattr(agent_or_policy, 'act'):
        # For custom agents with act method
        action = agent_or_policy.act(obs)
    elif callable(agent_or_policy):
        # For policy functions
        action = agent_or_policy(obs)
    else:
        # Fallback to random action
        action = env.action_space.sample()

    return action