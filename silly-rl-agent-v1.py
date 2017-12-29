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

Sort of Q-off-policy learning algorithm on eps-greedy soft-policy.

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


DEPRECATED The algorithm:

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

The above algorithm doesn't work on environmets with loops (potentially
infinite states). Example that breaks it:
 S0, A1 -> S1, reward == 0
 S0, A0 -> S0, reward == -1

While trying to solve it I changed the algorithm completely - it is no longer
Monte Carlo. Currently I go through entire episode and update Q function on
every state-action pair. After updating Q function I recompute values of
states.
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


class DummySpace:
    def __init__(self, n, shape):
        self.n = n
        self.shape = shape

    def sample(self):
        return np.random.randint(self.n)


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
        self.action_space = DummySpace(2, (1,))
        self.observation_space = DummySpace(1, (1,))

    def reset(self):
        self.state = 0
        return [0]

    def step(self, action):
        if action not in [0, 1]:
            self.state = 0
            return (2, 0, True, 0)
        self.state, reward = self.behaviour[(self.state, action)]
        return ([self.state], reward, self.state == 2, 0)


# minimal_env = MinimalEnv()
# minimal_env.reset()
# list(map(minimal_env.step, [0, 1, 0, 1, 1]))


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
        # self.terminal = np.zeros(shape)
        self.state_action_values = np.zeros(shape+(number_of_actions,))
        self.state_action_counters = np.zeros(shape+(number_of_actions,))
        self.state_action_verrs = np.zeros(shape+(number_of_actions,))
        self.state_values = np.zeros(shape)
        self.values_improvement = 1
        self.alpha = alpha
        # self.terminate_after = terminate_after

    def learn(self, episode):  # episode: [(state, action, reward), ...]
        # where reward follows (state, action)
        next_state, _, _ = episode.pop()
        # if self.terminal[next_state] > self.terminate_after:
        #     self.state_action_verrs[next_state] = 0
        # elif self.terminal[next_state] > -1:
        #     self.terminal[next_state] += 1
        for state, action, reward in reversed(episode):
            # self.terminal[state] = -1
            self.state_action_counters[state + (action,)] += 1
            q_diff = (reward + self.state_values[next_state]
                      - self.state_action_values[state + (action,)]) \
                * self.alpha
            #    / self.state_action_counters[state + (action,)]
            # compare the performance with the above (should be worse)
            self.state_action_values[state + (action,)] += q_diff
            self.state_action_verrs[state + (action,)] = abs(q_diff)
            # shit - above - terminal states never get updated!
            #   it's ok - they are zero from the start
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
env = MinimalEnv()
precision = 4
agent = SillyRLAgent(precision, env, eps=0.10, allow_determinism=False)
agent.learn_environment(episodes_num=1000)
agent.env_limits
agent.learn(0.005, 10000, 1000000)
agent.plot_all(1)
agent.policy.policy
agent.policy.state_action_values
len(agent.value_improvements)
print()
agent.policy.values_improvement


# %% moving average:
def moving_average(what, length):
    out = np.cumsum(what)
    out[length:] = out[length:] - out[:-length]
    return out[length - 1:] / length


# %% Agent
class SillyRLAgent:
    """
    Silly-rl-agent-v1:
    Idea:
      - explore by acting randomly - remember ranges of state coordinates
      - once you do not see new states quantize state coordinates
      - do sort of off-policy Q learning iteration on eps-greedy version of
        currently considered policy
    For now let's work only with:
      - discrete action spaces
      - multidimensional observation spaces
      - the same actions at every state
    Possible improvements:
      - learning while exploring state space (fastes way - store episodes,
        learn from them later)
      - using returns instead of rewards whenever feasible (it should speed
        up learning)
      - experience replay - remember some amount of previous episodes and
        do random changes from them when learning new episode
      - when in a state that didn't learn previously and neighbour states have
        learned - use their knowledge for start

    Design:
        - hyper-parameters:
            - precision - the number of discrete values given coordinate of
                state space can have
        - first version:
            - flatten and discretize observations
        - next versions:
            - flatten and discretize actions
        - expectations:
            - i can draw returns in the number of: episodes, steps
            - i can draw value improvements
            - i can tell it to draw moving average of the above
            - i can tell it to store every i-th agent (all its insides)
            - i can tell it to play any of stored agents
            - learn until some trashold on value improvement or number of steps
                - treshold and counter are arguments to learn method
            - learn more (until new treshold or counter)
            - [to think how] visualize policy, state values (even though state
                space can be huge, the states actually visited are probably
                small subset of the entire space)
            - evaluate rewards on currently best deterministic policy
    """
    def __init__(self, precision, env, return_freq=10,
                 allow_determinism=True, eps=0.1):
        self.precision = precision
        self.env = env
        self.number_of_actions = env.action_space.n
        self.eps = eps
        self.return_freq = return_freq
        self.allow_determinism = allow_determinism
        # TODO - correct so that observation space gets flattened (there might
        #   be some np functions to do this)
        #   - do the same for env_limits!!
        self.shape = (precision,) * env.observation_space.shape[0]
        self.reset()

    def reset(self):
        self.policy = IntegerCubePolicy(self.shape,
                                        self.number_of_actions,
                                        eps=self.eps)
        self.env_limits = np.zeros(env.observation_space.shape + (2,))
        self.value_improvements = []
        self.returns = []
        self.cum_steps = []
        self.policy_maps = None

    def learn_environment(self, episodes_num=1000):
        """
        Does random stuff for some time to learn the boundaries of state
        coordinates.

        Potential improvements:
            - do exploration until you stop seeing new values
            - add some observed step value to both limits
            - decide precision based on what you noticed
            - store episodes to learn from them later
            - learn while exploring [hard - probably involves some fundamental
                redesign]
        """
        state = self.env.reset()
        self.env_limits[:, 0] = state
        self.env_limits[:, 1] = state
        for _ in range(episodes_num):
            state = self.env.reset()
            stop = False
            while not stop:
                new_state, reward, stop, _ = env.step(
                                             self.env.action_space.sample())
                self.env_limits[:, 0] = np.minimum.reduce(
                                        [self.env_limits[:, 0],
                                         new_state])
                self.env_limits[:, 1] = np.maximum.reduce(
                                        [self.env_limits[:, 1],
                                         new_state])
                # state = new_state
        self.policy_maps = [IntervalToIndex(a=self.env_limits[i, 0],
                                            b=self.env_limits[i, 1],
                                            precision=self.precision)
                            for i in range(self.env_limits.shape[0])]

    def get_cube_state(self, env_state):
        return tuple([self.policy_maps[i](x)
                      for i, x in enumerate(env_state)])

    def learn(self, treshold=0.05, episodes_num=1000, steps_num=10000):
        episode_cnt = 0
        steps_cnt = 0
        while True:
            episode = []
            state = self.get_cube_state(self.env.reset())
            stop = False
            the_return = 0
            if episode_cnt % self.return_freq == 0:
                deterministic_run = True
            else:
                deterministic_run = False
            while not stop:
                if deterministic_run and self.allow_determinism:
                    action = self.policy.policy[state]
                else:
                    action = self.policy.get_eps_greedy_action(state)
                new_state, reward, stop, _ = env.step(action)
                new_state = self.get_cube_state(new_state)
                episode.append((state, action, reward))
                the_return += reward
                state = new_state
            episode.append((state, 0, 0))
            self.policy.learn(episode)
            self.value_improvements.append(self.policy.values_improvement)
            episode_cnt += 1
            steps_cnt += len(episode) - 1
            self.cum_steps.append(steps_cnt)
            if deterministic_run:
                self.returns.append(the_return)

            if self.policy.values_improvement < treshold \
               or episode_cnt > episodes_num or steps_cnt > steps_num:
                break

    def plot_value_improvements(self, episodes_on_x=False, average_over=1):
        value_improvements_movavg = moving_average(self.value_improvements,
                                                   average_over)
        plt.plot(range(len(value_improvements_movavg)),
                 value_improvements_movavg)
        plt.show()
        TODO
        return

    def plot_returns(self, episodes_on_x=False, average_over=1):
        returns_movavg = moving_average(self.returns,
                                        average_over)
        plt.plot(np.arange(len(returns_movavg)), returns_movavg)
        plt.show()
        TODO
        return

    def plot_all(self, average_over=1):
        print()
        len(self.value_improvements)
        print()
        self.plot_value_improvements(average_over=average_over)
        print()
        self.plot_returns(average_over=average_over)
        return

    # def animate(self):
    #     state = tuple([self.policy_maps[i](x)
    #                    for i, x in enumerate(env.reset())])
    #     stop = False
    #     the_return = 0
    #     while not stop:
    #         action = self.policy.policy[state]  # get greedy action
    #         new_state, reward, stop, _ = env.step(action)
    #         new_state = tuple([self.policy_maps[i](x)
    #                            for i, x in enumerate(new_state)])
    #         episode.append((state, action, reward))
    #         the_return += reward
    #         state = new_state


# %% tests:
env = gym.make("CartPole-v0")
precision = 8
treshold = 0.05
agent = SillyRLAgent(precision, env, eps=0.15)
agent.learn_environment(episodes_num=1000)
agent.env_limits
agent.learn(0.05, 2000, 200000)
agent.plot_all(5)


# other:
#   mountain car
#   acrobot
#   reacher
# - text environments
