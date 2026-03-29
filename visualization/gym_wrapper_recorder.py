import gymnasium as gym 
from gymnasium.wrappers import RecordVideo 


def create_video_gym_wrapper(env_name, 
                             agent_or_policy,
                            video_folder="./videos", 
                            num_episodes=5):
    """
    Create video using gym's video recording wrapper
    """

    # Create environment with video recording
    env = gym.make(env_name, render_mode = "rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False 

        while not done:
            action = get_action(env, agent_or_policy, obs)
            obs, reward, done, truncated, info = env.step(action)

    env.close()
    print(f"Videos saved in {video_folder}")

def get_action(env,
               agent_or_policy,
               obs):
    # Get action from your trained agent
    if hasattr(agent_or_policy, 'predict'):
        action, _ = agent_or_policy.predict(obs)
    elif hasattr(agent_or_policy, 'act'):
        action = agent_or_policy.act(obs)
    elif callable(agent_or_policy):
        action = agent_or_policy(obs)
    else:
        action = env.action_space.sample()
    return action 
