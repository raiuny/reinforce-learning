import gym
from stable_baselines3 import PPO

# 创建环境
env = gym.make("CartPole-v1")

# 创建 PPO 模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 保存模型
model.save("ppo_cartpole")
