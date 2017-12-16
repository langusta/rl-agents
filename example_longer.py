import gym


def cartAgent(observation):
    action = 0 if observation[2] < 0 else 1
    return action


def simulate(agent, render=False):
    observation = env.reset()
    cumulative_reward = 0
    for t in range(200):
        if render:
            env.render()
        action = agent(observation)
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
    return cumulative_reward, t+1


env = gym.make('CartPole-v0')
for episode in range(20):
    cumulative_reward, time = simulate(cartAgent)
    print(("Episode: {:3}, cumulative reward: {:4.0f}, timesteps: {:3}"
           ).format(episode, cumulative_reward, time))

simulate(cartAgent, True)
