from ppo import ProximalPolicyOptimization as PPO

ppo = PPO(env='CartPole-v1')

ppo.learn()

ppo.save()