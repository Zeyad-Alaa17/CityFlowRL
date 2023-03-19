# noinspection PyUnresolvedReferences
import CityFlowRL
import os
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

models_dir = "../models/"
env_kwargs = {'config': "hangzhou_1x1_bc-tyc_18041607_1h", 'steps_per_episode': 100, 'steps_per_action': 30}


def train():
    env = make_vec_env('CityFlowRL-v0', n_envs=12, vec_env_cls=SubprocVecEnv)

    model = A2C('MlpPolicy', env, verbose=2, tensorboard_log="./tensorboard/")
    # model = A2C.load("a2c", env=env)
    model.learn(total_timesteps=3000, reset_num_timesteps=False)
    model.save(os.path.join(models_dir, "a2c"))
    print("model saved")


def test():
    env = gym.make('CityFlowRL-v0', **env_kwargs)

    env = DummyVecEnv([lambda: env])

    model = A2C.load("a2c")
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
    # train()
    test()
