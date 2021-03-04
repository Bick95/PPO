from plot import plot_avg_trajectory_len
from ppo import ProximalPolicyOptimization as PPO

ppo = PPO(env='CartPole-v1')

train_stats = ppo.learn()
print(train_stats)

ppo.save()

plot_avg_trajectory_len(train_stats)
