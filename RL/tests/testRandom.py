# noinspection PyUnresolvedReferences
import random
import CityFlowRL
import gym

env_kwargs = {'config': "hangzhou_1x1_bc-tyc_18041607_1h", 'steps_per_episode': 100, 'steps_per_action': 30}
env = gym.make('CityFlowRL-v0', **env_kwargs)


def test():
    # Check action space
    actions = 0
    while env.action_space.contains(actions):
        actions += 1
    print(env.action_space)
    print(env.observation_space)
    env.reset()

    # iterate environment a little bit to test env

    timeSteps = 100

    rewards = []
    for i in range(timeSteps):
        n = random.randint(0, actions - 1)
        observation, reward, done, info = env.step(action=n)
        rewards.append(reward)

        if done:
            break

    print(info)
    print("Episode reward: ", sum(rewards))

    observation = env.reset()
    print(observation)

    # replay.run(env_kwargs['config'])


if __name__ == "__main__":
    test()
    print("Test")
