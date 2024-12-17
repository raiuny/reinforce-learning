import gym
from stable_baselines3 import PPO

# 创建环境
env = gym.make("CartPole-v1")
# 加载模型
loaded_model = PPO.load("ppo_cartpole")
# 在环境中测试模型
obs = env.reset()[0]
for _ in range(1000):
    action, _ = loaded_model.predict(obs)
    obs, reward, done, _, dic = env.step(action)
    print(obs)
    env.render()

    if done:
        obs = env.reset()[0]