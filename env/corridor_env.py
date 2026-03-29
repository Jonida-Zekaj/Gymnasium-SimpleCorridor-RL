class SimpleCorridorEnv(gym.Env):
    def __init__(self, corridor_length=10):
        super(SimpleCorridorEnv, self).__init__()
        self.corridor_length = corridor_length
        self.current_pos = 0                    # agents start position

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # movements: 0=Left/ 1=Right
        self.observation_space = spaces.Discrete(corridor_length)

    # At the begining of each episode environment is reset
    def reset(self, seed=None, options=None):
        self.current_pos = 0                     # returning to original position
        return self.current_pos, {}

    # Actions and updates
    def step(self, action):
        if action == 1:
            self.current_pos = min(self.current_pos + 1, self.corridor_length - 1)
        else:                                    # action == 0
            self.current_pos = max(self.current_pos - 1, 0)

        # Check if reached goal
        terminated = self.current_pos == self.corridor_length - 1
        reward = -0.1
        if terminated:
          reward=1.0                             # only if the goal is reached

        truncated = False

        return self.current_pos, reward, terminated, truncated, {}

    # Visual display of the agent's movements in the env.
    def render(self):
      corridor = ['_'] * self.corridor_length

      if self.current_pos == self.corridor_length - 1:
          corridor[self.current_pos] = 'AG'
      else:
          corridor[self.current_pos] = 'A'
          corridor[-1] = 'G'

      print(f"Agent's position: {self.current_pos}: {''.join(corridor)}")


# Registering the environment in GYM
gym.register(
    id="SimpleCorridor-v0",
    entry_point="__main__:SimpleCorridorEnv",
    kwargs={"corridor_length": 10}
)