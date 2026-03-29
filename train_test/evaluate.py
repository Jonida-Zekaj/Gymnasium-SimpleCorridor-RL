import gymnasium as gym
from pyglet import env
from stable_baselines3 import PPO


def evaluate_agent(path: str = "PPOSimpleCorridor", 
                   num_episodes: int = 100,
                   max_steps_per_episode : int = 50):
    
    """
    Evaluate a trained PPO agent.
    
    Args:
        model_path: Path to trained model
        num_episodes: Number of episodes to test
        max_steps_per_episode: Max steps per episode
        
    Returns:
        Dictionary with success_rate and average_steps
    """
    eval_env=gym.make("SimpleCorridor-v0")
    model = PPO.load(path)

    successes =0
    total_steps=0

    for episode in range(num_episodes):
        obs, _ =eval_env.reset()
        episode_steps=0

        for step in range (max_steps_per_episode):
            action,_=model.predict(obs)
            obs, reward, done, truncated, _=eval_env.step(action)
            episode_steps+=1

            if done or truncated:
                if done:
                    successes+=1
                break
        total_steps+=episode_steps

    eval_env.close() # close to clean up

    success_rate=successes/num_episodes
    average_steps=total_steps/num_episodes

    print("\n" + "="*50)
    print("Performance Evaluation - Trained Agent")
    print("="*50)
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Steps per Episode: {average_steps:.3f}")
    print(f"Total Episodes: {num_episodes}")
    print(f"Successful Episodes: {successes}")
    
    return {"success_rate": success_rate, "average_steps": average_steps}

if __name__ == "__main__":
    evaluate_agent()
