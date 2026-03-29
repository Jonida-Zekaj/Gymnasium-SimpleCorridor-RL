import matplotlib.pyplot as plt 
from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_rewards(log_folder):
  x, y=ts2xy(load_results(log_folder), 'timesteps')
  plt.figure(figsize=(10,5))
  plt.plot(x, y, label="Reward per Episode")
  plt.xlabel("Timesteps")
  plt.ylabel("Reward per Episode")
  plt.title("Reward versus Timesteps")
  plt.grid()
  plt.legend()
  plt.show()