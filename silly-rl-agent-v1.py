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

We sort of Q-off-policy learning on eps-greedy soft-policy.

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
- sensible terminate condition (rethink terminating state errors)
- rethink state_acxtion_verrs initation as zeros - then we have potentially
    stochastic stop condition


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
    def __init__(self, shape, number_of_actions, eps=0.1, alpha=0.05,
                 terminate_after=1000):
        """Initialize by random integers."""
        self.shape = shape
        self.number_of_actions = number_of_actions
        self.eps = eps
        self.policy = np.random.randint(number_of_actions, size=shape)
        self.terminal = np.zeros(shape)
        self.state_action_values = np.zeros(shape+(number_of_actions,))
        self.state_action_counters = np.zeros(shape+(number_of_actions,))
        self.state_action_verrs = np.zeros(shape+(number_of_actions,))
        self.state_values = np.zeros(shape)
        self.values_improvement = 1
        self.alpha = alpha
        self.terminate_after = terminate_after

    def learn(self, episode):  # episode: [(state, action, reward), ...]
        # where reward follows (state, action)
        next_state, _, _ = episode.pop()
        if self.terminal[next_state] > self.terminate_after:
            self.state_action_verrs[next_state] = 0
        elif self.terminal[next_state] > -1:
            self.terminal[next_state] += 1
        for state, action, reward in reversed(episode):
            self.terminal[state] = -1
            self.state_action_counters[state + (action,)] += 1
            q_diff = (reward + self.state_values[next_state]
                      - self.state_action_values[state + (action,)]) \
                * self.alpha
            #    / self.state_action_counters[state + (action,)]
            # compare the performance with the above (should be worse)
            self.state_action_values[state + (action,)] += q_diff
            self.state_action_verrs[state + (action,)] = abs(q_diff)
            # shit - above - terminal states never get updated!
            # IMPROVEMENT = check for convergence on each state-action
            # remember that:
            # - there might be loop-y actions (-inf == value)
            # - rewards may be probabilistic
            #       (you want remember some sort of confidence boud)
            next_state = state
        self.values_improvement = max(
            np.abs(self.state_values
                   - self.state_action_values.max(-1)
                   ).max(),
            self.state_action_verrs.max())
        self.state_values = self.state_action_values.max(-1)
        self.policy = self.state_action_values.argmax(-1)

    def am_i_greedy(self):
        return (self.policy == self.state_action_values.argmax(-1)).all()

    def let_me_be_greedy(self):
        self.policy = self.state_action_values.argmax(-1)

    def get_eps_greedy_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.choice(self.number_of_actions)
        else:
            return self.policy[state]


# %% tests
# let's test!
shape = (1, 3)
number_of_actions = 2
icp = IntegerCubePolicy(shape, number_of_actions, eps=0.1)
precision = 0.005
values_improvements = []
env = MinimalEnv()
counter = 0
while True:
    episode = []
    state = (0, env.reset())
    stop = False
    while not stop:
        action = icp.get_eps_greedy_action(state)
        new_state, reward, stop, _ = env.step(action)
        episode.append((state, action, reward))
        state = (0, new_state)
    episode.append((state, 0, 0))
    icp.learn(episode)
    values_improvements.append(icp.values_improvement)
    counter += 1
    if icp.values_improvement < precision or counter > 10000:
        break

plt.plot(range(len(values_improvements[-100:])), values_improvements[-100:])
len(values_improvements)
# although it should
icp.policy
print()
icp.state_action_values
print()
icp.state_values
print()
icp.state_action_counters
print()
icp.state_action_verrs


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
