import gym
import numpy as np
from collections import Counter
import math
import matplotlib.pyplot as plt
%matplotlib inline


# %% Parameter Class


class Parameter(object):
    """ Parameter for an Agent to use."""
    def __init__(self, begin, end, precision, weight):
        super(Parameter, self).__init__()
        self.precision = precision
        self.weights = np.array([weight]*precision)
        self.values = np.linspace(begin, end, precision)
        self.index = 0
        self.choose()

    def probabilities(self):
        return self.weights / sum(self.weights)

    def choose(self):
        self.index = np.random.choice(self.precision, p=self.probabilities())

    def updateWeightAt(self, index, by):
        self.weights[index] += by

    def updateCurrentWeight(self, by):
        self.updateWeightAt(self.index, by)

    def updateFuzzy(self, by, fuzziness=3, smoothing=lambda x: x):
        low = max(0, self.index - fuzziness)
        high = min(self.index + fuzziness, self.precision-1)
        for i in range(low, high+1):
            self.weights[i] += by / smoothing(abs(i-self.index)+1)

    def discountWeights(self, by):
        self.weights *= by

    def __call__(self):
        return self.values[self.index]


# Some tests:
# p = Parameter(1, 10, 10, 5)
# p.updateFuzzy(5)
# print(p.weights)
# print(p.index)
# print(p.precision)
# # p.updateWeightAt(5, 10)
# p.updateCurrentWeight(10)
# cc = Counter()
# for i in range(12000):
#     p.choose()
#     cc[p()] += 1
# cc

# %% CartAgent class


class CartAgent(object):
    """ Agent for CartPole-v0"""

    def __init__(self, precision=100, start_weight=20.0):
        self.velocity_treshold = Parameter(0.01, 1.0, precision, start_weight)
        self.angle_treshold = Parameter(0.005, 0.21, precision, start_weight)

    def make_decision(self, observation):
        if abs(observation[1]) < self.velocity_treshold():
            if abs(observation[2]) < self.angle_treshold():
                action = 0 if observation[3] < 0 else 1
            else:
                action = 0 if observation[2] < 0 else 1
        else:  # x velocity
            action = 1 if observation[1] < 0 else 0
        return action

    def learn(self, reward):
        self.velocity_treshold.updateCurrentWeight(reward)
        self.velocity_treshold.choose()
        self.angle_treshold.updateCurrentWeight(reward)
        self.angle_treshold.choose()

    def learn_fuzzy(self, reward, fuzziness=3, smoothing=lambda x: x):
        self.angle_treshold.updateFuzzy(reward, fuzziness, smoothing)
        self.angle_treshold.choose()
        self.velocity_treshold.updateFuzzy(reward, fuzziness, smoothing)
        self.velocity_treshold.choose()


# %% Plotting multilines example:


def plot_agent_history(agent, which, i_multiplyer):
        plt.subplots(figsize=(15, 10))
        for i in range(len(agent.hist_probs)):
            plt.plot(agent.angle_treshold_values, agent.hist_probs[i][which],
                     label="i=%d " % (i_multiplyer*i,))
        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True,
                         fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.show()


# %% Running and testing


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def simulate(agent, render=False, should_print=False):
    env = gym.make('CartPole-v0')
    observation = env.reset()
    cumulative_reward = 0
    while True:
        if render:
            env.render()
        action = agent.make_decision(observation)
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if should_print:
            print(observation)
        if done:
            break
    return cumulative_reward, observation


# %% tests

agent1 = CartAgent()
simulate(agent1)
plt.plot(agent1.angle_treshold.values, agent1.angle_treshold.probabilities())

# %% trials


def simulation_series(start_weight, learn_function, learn_paramteres={},
                      reward_function=lambda x: x, episodes=1000):
    agent = CartAgent(start_weight=start_weight)
    rewards = []
    probs = ([], [])
    for episode in range(episodes):
        reward, _ = simulate(agent)
        learn_function(agent, reward_function(reward), **learn_paramteres)
        rewards.append(reward)
        if episode % 100 == 0:
            probs[0].append(agent.angle_treshold.probabilities())
            probs[1].append(agent.velocity_treshold.probabilities())
    return rewards, probs, agent, episodes


def plot_avr_results(sim, angle=True, velocity=True, rewards=True,
                     smoothing=100):
    if angle:
        print("Angle treshold distribution")
        plt.plot(sim[2].angle_treshold.values, sim[1][0][9])
        plt.show()
    if velocity:
        print("Velocity treshold distribution")
        plt.plot(sim[2].velocity_treshold.values, sim[1][1][9])
        plt.show()
    if rewards:
        print("Rewards:")
        plt.plot(range(sim[3]-smoothing+1), moving_average(sim[0], smoothing))
        plt.show()


# %% first version learning:

sim1 = simulation_series(start_weight=200.0,
                         learn_function=CartAgent.learn, episodes=1000)
plot_avr_results(sim1, angle=False, velocity=False)

# %% learn only on maximal rewards

sim2 = simulation_series(start_weight=200.0,
                         learn_function=CartAgent.learn,
                         reward_function=lambda x: math.exp(x-198),
                         episodes=1000)
plot_avr_results(sim2, angle=False, velocity=False)

# %% learn only on maximal rewards but lower starting weight

sim3 = simulation_series(start_weight=20.0,
                         learn_function=CartAgent.learn,
                         reward_function=lambda x: math.exp(x-198),
                         episodes=1000)
plot_avr_results(sim3, angle=False, velocity=False)


# %% fuzzy learn plus on maximal rewards

sim4 = simulation_series(start_weight=20.0,
                         learn_function=CartAgent.learn_fuzzy,
                         learn_paramteres={'fuzziness': 5},
                         reward_function=lambda x: math.exp(x-198),
                         episodes=1000)
plot_avr_results(sim4, angle=False, velocity=False)

sim5 = simulation_series(start_weight=20.0,
                         learn_function=CartAgent.learn_fuzzy,
                         learn_paramteres={'fuzziness': 5},
                         reward_function=lambda x: math.exp(x-198),
                         episodes=5000)
plot_avr_results(sim5, angle=False, velocity=False, smoothing=200)
plot_avr_results(sim5, angle=True, velocity=True, smoothing=200)

#
# it works!! :)
#
