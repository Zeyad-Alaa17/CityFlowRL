# noinspection PyUnresolvedReferences
import CityFlowRL
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

models_dir = "../models/"
env_kwargs = {'config': "hangzhou_1x1_bc-tyc_18041607_1h", 'steps_per_episode': 100, 'steps_per_action': 30}


def train():
    env = make_vec_env('CityFlowRL-v0', n_envs=12, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)

    model = PPO('MlpPolicy', env, verbose=2, tensorboard_log="./tensorboard/", batch_size=32, n_steps=24)
    # model = PPO.load("ppo_3", env=env)
    model.learn(total_timesteps=3000, reset_num_timesteps=False)
    model.save(os.path.join(models_dir, "ppo"))
    print("model saved")


def test():
    env = gym.make('CityFlowRL-v0', **env_kwargs)

    env = DummyVecEnv([lambda: env])

    model = PPO.load("ppo")
    episodes = 10
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = model.predict(obs)
            obs, reward, done, info = env.step(action)
        print(info)
    env.close()


if __name__ == "__main__":
    train()
    # test()
