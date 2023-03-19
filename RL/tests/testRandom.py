# noinspection PyUnresolvedReferences
import CityFlowRL
import random
import gym
import replay

env_kwargs = {'config': "hangzhou", 'steps_per_episode': 100, 'steps_per_action': 30}
env = gym.make('CityFlowRL-v0', **env_kwargs)

# Check action space
actions = 0
while env.action_space.contains(actions):
    actions += 1
print(env.action_space)
print(env.observation_space.shape)
env.reset()

# iterate environment a little bit to test env

timeSteps = 100
for i in range(timeSteps):

    n = random.randint(0, actions - 1)

    observation, reward, done, debug = env.step(action=n)
    print(observation)
    print(reward)
    print(debug)

    if done:
        break

observation = env.reset()
print(observation)

replay.run(env_kwargs['config'])
