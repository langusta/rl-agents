import gym
import numpy as np
from collections import Counter
import math
import matplotlib.pyplot as plt
%matplotlib inline


# %% Agent

idea = """
This agent is supposed to solve environments where:
- there is finite number of actions: 0, 1, 2, ..., max_action
- actions are the same in each state
- observations are cubes of real values of given shape

We will use Monte Carlo off-policy learning on eps-greedy soft-policy.

TODO:
- implement and check performance on cart agent
- possible improvement - we can use the fact that our behaviour policy is of
  particular kind and get rid of weights - rethink if it is really possible and
  if is then implement and compar
- update policy pi inside the loop that learns on episode (like in sutton) and
  compare
- potentail improvement is: we could use whole paths (to learn) if we allow
  a little bootstrapping - the moment we notice action that was different then
  the one from policy then we can substitute the reward with its estimate
  Q(s,a) and continue - the question is what to do with weights then
  - probably reset path weight to 1


The algorithm:

pi - deterministic policy we want to learn
mi - eps-pi policy that we use to explore (it follows pi with 1-eps prob)

Q(s,a) - state-action value estimates
ro(s,a) - initiate with zeros, current sum of path weights for s
iterate up to max_iterations:
    generate episode:
        states = list of states
        actions_= on follwoing states use current soft policy
        rewards = list of following rewards # last one should be 0
    W = 1
    G = 0
    for t in range(len(states)-1, -1, -1):
        G += rewards[t+1]
        ro(s,a) += W
        Q(s,a) += W/ro(s,a)*(G-Q(s,a))
        # W *= pi(actions[t]|states[t]) / mi(actions[t]|states[t])
        # in our case pi(actions[t]|states[t]) is 0 or 1
        if pi(states[t]) == actions[t]:
            W *= 1/(1-ro.eps+ro.eps/states[t].number_of_actions)
        else:
            break
    if(pi == Greedy(Q)):
        break
    pi = Greedy(Q)
    mi = eps-Greedy(pi)
pi is optimal
"""


# %% Map interval to index-s
class IntervalToIndex:
    """
    Maps given interval to precision indexes starting with 0. Example:
        dig_int = IntervalToIndex(a=0, b=2, precision=4)
        list(map(dig_int, [0, 1, 2]))
        # [0, 1, 2]
        list(map(dig_int, [-0.5, 0.5, 1.5, 2.5]))
        # [0, 1, 2, 3]
        list(map(dig_int, [-100, 100]))
        # [0, 3]
    """
    def __init__(self, a=0.0, b=1.0, precision=1000):
        self.a = a
        self.b = b
        self.precision = precision - 2

    def __call__(self, value):
        if value <= self.a:
            return 0
        elif value > self.b:
            return self.precision + 1
        return math.ceil(self.precision * (value - self.a) / (self.b - self.a))


# %% super simple testing environment
# based on example 4.1 from Sutton
class SuttonEx4d1:
    def __init__(self):
        self.state = (2, 2)

    def reset(self):
        state = ()


class MinimalEnv:
    """
    We have three states:
    0
    1
    2
    and two actions: 0 - go up, 1 go down
    """
    def __init__(self):
        self.state = 0
        self.behaviour = {  # (state, action): (new state, reward)
            (0, 0): (0, -1),
            (0, 1): (1, -2),
            (1, 0): (0, -1),
            (1, 1): (2, 0),
            (2, 0): (2, 0),
            (2, 1): (2, 0)
        }

    def reset(self):
        self.state = 0
        return 0

    def step(self, action):
        if action not in [0, 1]:
            self.state = 0
            return (2, 0, True, 0)
        self.state, reward = self.behaviour[(self.state, action)]
        return (self.state, reward, self.state == 2, 0)


# minimal_env = MinimalEnv()
# minimal_env.reset()
# list(map(minimal_env.step, [0, 1, 0, 1, 1]))

# %% Silly^2 version


# %% Less silly version
# %% Policy
class IntegerCubePolicy:
    """
    A policy that can improve itself buahahahahaha!
    A state is of dimensions given by shape.
    """
    def __init__(self, shape, number_of_actions, eps=0.1):
        """Initialize by random integers."""
        self.shape = shape
        self.number_of_actions = number_of_actions
        self.eps = eps
        self.greedy_arm_prob_inv = 1 / (1 - eps + eps / number_of_actions)
        self.policy = np.random.randint(number_of_actions, size=shape)
        self.state_action_values = np.zeros(shape+(number_of_actions,))
        self.path_weights = np.zeros(shape+(number_of_actions,))

    def learn(self, episode):  # episode: [(state, action, reward), ...]
        # where reward follows (state, action)
        path_weight = 1
        path_return = 0
        for state, action, reward in reversed(episode):
            path_return += reward
            self.path_weights[state + (action,)] += path_weight
            self.state_action_values[state + (action,)] += \
                path_weight \
                * (path_return - self.state_action_values[state + (action,)]) \
                / self.path_weights[state + (action,)]
            # if self.policy[state] == action:
            if path_weight == 0:
                # careful here - if we do changes to policy within this
                # training then we might sometimes agree with non-greedy arm
                # and use different probability below
                # shit - below works for policy evaluation only
                # what if we change the policy along the way - then it might
                # not be correct
                path_weight *= (self.policy[state] == action) \
                               * self.greedy_arm_prob_inv
            else:
                break

    def am_i_greedy(self):
        return (self.policy == self.state_action_values.max(-1)).all()

    def let_me_be_greedy(self):
        """
        NOT NECESSARY - REMOVE
        What we want:
            self.policy = self.state_action_values.argmax(-1)
        BUT we also want ties to be broken randomly.
        """
        # sav = self.state_action_values
        # maxes = np.stack([sav.max(-1) for _ in range(sav.shape[-1])], axis=-1)
        # self.policy = (np.random.random(sav.shape) * (sav == maxes)).argmax(-1)
        self.policy = self.state_action_values.argmax(-1)

    def get_eps_greedy_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.choice(self.number_of_actions)
        else:
            return self.policy[state]


# %% tests
# let's test!
icp = IntegerCubePolicy((1, 3), 2, eps=0.15)
icp.state_action_values.shape
icp.path_weights
1/icp.greedy_arm_prob_inv
icp.policy
icp.get_eps_greedy_action((0, 1))

shape = (1, 3)
number_of_actions = 2
icp = IntegerCubePolicy(shape, number_of_actions, eps=0.1)
env = MinimalEnv()
for _ in range(5000):
    episode = []
    state = (0, env.reset())
    stop = False
    while not stop:
        action = icp.get_eps_greedy_action(state)
        new_state, reward, stop, _ = env.step(action)
        episode.append((state, action, reward))
        state = (0, new_state)
    icp.learn(episode)
    # icp.am_i_greedy()
    icp.let_me_be_greedy()

# although it should
icp.policy
icp.state_action_values
icp.path_weights
icp.let_me_be_greedy()

episode

[((0, 0), 1, -2),
 ((0, 1), 0, -1),
 ((0, 0), 1, -2),
 ((0, 1), 1, 0)]
# example_episode = episode

icp.state_action_values = np.zeros(shape+(number_of_actions,))
icp.path_weights = np.zeros(shape+(number_of_actions,))

path_weight = 1
path_return = 0
for state, action, reward in reversed(episode):
    state, action, reward = episode[-1]
    state, action, reward
    path_return += reward
    path_return
    icp.path_weights[state + (action,)] += path_weight
    icp.state_action_values[state + (action,)] += \
        path_weight \
        * (path_return - icp.state_action_values[state + (action,)]) \
        / icp.path_weights[state + (action,)]
    if icp.policy[state] == action:
        path_weight *= icp.greedy_arm_prob_inv
    else:
        break


# %% Agent
class SillyStateSpace:
    """
    It should 'translate' float intervals for each state coordinate
    to integer index of a hypercube.
    It should allow for learning ranges of parameters.
    """
    pass


class SillyRLAgent:
    """
    Silly-rl-agent-v1:
    Idea:
      - explore by acting randomly - remember ranges of state coordinates
      - once you do not see new states quantize state coordinates
      - do monte carlo off-policy iteration on eps-greedy version of currently
        considered policy
    For now let's work only with:
      - discrete action spaces
      - multidimensional observation spaces
      - the same actions at every state
    Possible improvements:
      - learning while exploring
    """
    def __init__(self, precision, number_of_actions, observation_dims):
        self.__precision = precision
        self.__number_of_actions = number_of_actions
        self.__observation_dims = observation_dims


# %% Tests

env = gym.make("CartPole-v0")

agent = SillyRLAgent(1000,
                     env.action_space.n,
                     env.observation_space.shape)
