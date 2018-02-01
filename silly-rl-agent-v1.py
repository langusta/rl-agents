import gym
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from IPython import get_ipython

get_ipython().magic(u'matplotlib inline')


# %% Agent

idea = """
This agent is supposed to solve environments where:
- there is finite number of actions: 0, 1, 2, ..., max_action
- actions are the same in each state
- observations are cubes of real values of given shape

I wanted to implement Monte Carlo algorithm, but during tests I discovered that
it may get stuck on loopy policies (for example going back and forth on some
two states). When I started thinking on how to modify the algorithm I ended up
with some sort of Q-learning. I won't modify it further. Instead, I will modify
the code so that further changes are treated as different agents and I will
comapre them.

Improvement ideas:
- implement and check performance on cart agent
- possible improvement - we can use the fact that our behaviour policy is of
  particular kind and get rid of weights - rethink if it is really possible and
  if is then implement and compare
- update policy pi inside the loop that learns on episode (like in sutton) and
  compare
- potentail improvement is: we could use whole paths (to learn) if we allow
  a little bootstrapping - the moment we notice action that was different then
  the one from policy then we can substitute the reward with its estimate
  Q(s,a) and continue - the question is what to do with weights then
  - probably reset path weight to 1
- sensible terminate condition (rethink terminating state errors)
- rethink state_action_verrs initation as zeros - then we have potentially
    stochastic stop condition


Inicial algorithm idea (Monte Carlo agent):

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
"""


# %% Map interval to index-s
class IntervalToIndex:
    """
    Maps given interval to array indexes starting with 0. Example:
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
    def __init__(self, shape, number_of_actions, eps=0.1, alpha=0.05):
        """Initialize by random integers."""
        self.shape = shape
        self.number_of_actions = number_of_actions
        self.eps = eps
        self.policy = np.random.randint(number_of_actions, size=shape)
        self.state_action_values = np.zeros(shape + (number_of_actions,))
        self.state_action_counters = np.zeros(shape + (number_of_actions,))
        self.state_action_verrs = np.zeros(shape + (number_of_actions,))
        self.state_values = np.zeros(shape)
        self.values_improvement = 1
        self.alpha = alpha

    def learn(self, episode):  # episode: [(state, action, reward), ...]
        # where reward follows (state, action)
        next_state, _, _ = episode.pop()
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

    def get_eps_greedy_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.choice(self.number_of_actions)
        else:
            return self.policy[state]


# Tests of IntegerCubePolicy are given below the Silly agent class

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
        up learning) - another option is to use reward + value estimate
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
            - look at minimal change in observations after an action when
                exploring, use it to compute precision.
            - after you learned something - decide action in new state,
                based on learned state that is close (do some value smoothing)
            - flatten and discretize actions
        - expectations:
            - [to think how] visualize policy, state values (even though state
                space can be huge, the states actually visited are probably
                small subset of the entire space)
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
        if len(env.observation_space.shape) > 0:
            self.shape = (precision,) * env.observation_space.shape[0]
            self.observation_box = True
        else:
            self.shape = (precision,)
            self.observation_box = False
        self.reset()

    def reset(self):
        self.episode_cnt = 0
        self.policy = IntegerCubePolicy(self.shape,
                                        self.number_of_actions,
                                        eps=self.eps)
        if self.observation_box:
            self.env_limits = np.zeros(self.env.observation_space.shape + (2,))
        else:
            self.env_limits = np.zeros((1, 2))
        self.value_improvements = []
        self.returns = []
        self.cum_steps = []
        self.policy_maps = None
        self.policy_store = []

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
        if not self.observation_box:
            state = np.array([state])
        self.env_limits[:, 0] = state
        self.env_limits[:, 1] = state
        for _ in range(episodes_num):
            state = self.env.reset()
            if not self.observation_box:
                state = np.array([state])
            stop = False
            while not stop:
                new_state, reward, stop, _ = self.env.step(
                    self.env.action_space.sample())
                if not self.observation_box:
                    new_state = np.array([new_state])
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
        if not self.observation_box:
            env_state = np.array([env_state])
        return tuple([self.policy_maps[i](x)
                      for i, x in enumerate(env_state)])

    def learn(self, treshold=0.05, episodes_num=1000, steps_num=10000):
        steps_cnt = 0 if len(self.cum_steps) == 0 else self.cum_steps[-1]
        while True:
            episode = []
            state = self.get_cube_state(self.env.reset())
            stop = False
            the_return = 0
            if self.episode_cnt % self.return_freq == 0:
                deterministic_run = True
            else:
                deterministic_run = False
            while not stop:
                if deterministic_run and self.allow_determinism:
                    action = self.policy.policy[state]
                else:
                    action = self.policy.get_eps_greedy_action(state)
                new_state, reward, stop, _ = self.env.step(action)
                new_state = self.get_cube_state(new_state)
                episode.append((state, action, reward))
                the_return += reward
                state = new_state
            episode.append((state, 0, 0))
            self.policy.learn(episode)
            self.value_improvements.append(self.policy.values_improvement)
            self.episode_cnt += 1
            steps_cnt += len(episode) - 1
            self.cum_steps.append(steps_cnt)
            if deterministic_run:
                self.returns.append(the_return)
                self.policy_store.append(copy.deepcopy(self.policy))

            if self.policy.values_improvement < treshold \
               or self.episode_cnt > episodes_num or steps_cnt > steps_num:
                break

    def play(self, which=-1):
        state = self.get_cube_state(self.env.reset())
        stop = False
        the_return = 0
        while not stop:
            self.env.render()
            action = self.policy_store[which].policy[state]
            new_state, reward, stop, _ = self.env.step(action)
            new_state = self.get_cube_state(new_state)
            the_return += reward
            state = new_state
        self.env.render(close=True)
        print("The return was: {}".format(the_return))
        print("The last state:")
        print(state)

    def plot_v_or_r(self, what, episodes_on_x=False, average_over=1):
        if what == 'v':
            movavg = moving_average(self.value_improvements, average_over)
            ylabel = "Value improvements"
        else:
            movavg = moving_average(self.returns, average_over)
            ylabel = "Returns"
        if episodes_on_x:
            xs = range(len(movavg))
            xlabel = "episodes"
        else:
            if what == 'v':
                xs = self.cum_steps[average_over - 1:]
            else:
                xs = range(0, self.episode_cnt + 1, self.return_freq)
                xs = xs[average_over - 1:]
                xs = [self.cum_steps[x] for x in xs]
            xlabel = "steps"
        plt.plot(xs, movavg)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        return

    def plot_all(self, average_over=1):
        print("Value improvements averaged over {}.".format(average_over))
        self.plot_v_or_r('v', average_over=average_over)
        print("Returns averaged over {}.".format(average_over))
        self.plot_v_or_r('r', average_over=average_over)
        return


# %% tests with MinimalEnv
# let's test!
env = MinimalEnv()
precision = 15
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
# list(map(agent.policy_maps[0], [0, 1, 2, 3]))

# %% tests with CartPole
envi = gym.make("CartPole-v0")
precision = 15
treshold = 0.05
agent = SillyRLAgent(precision, envi, eps=0.1)
agent.learn_environment(episodes_num=1000)
print()
agent.env_limits
agent.learn(0.05, 10000, 1000000)
agent.plot_all(5)
print("\nNumber of episodes: {}\nNumber of steps: {}".format(agent.episode_cnt,
      agent.cum_steps[-1]))
agent.play()

# other:
#   mountain car
#   acrobot
#   reacher
# - text environments


# %% MountainCar-v0
envi = gym.make("MountainCar-v0")
precision = 15
treshold = 0.05
agent = SillyRLAgent(precision, envi, eps=0.15)
agent.learn_environment(episodes_num=1000)
print()
agent.env_limits
agent.learn(0.05, 10000, 1000000)
agent.plot_all(5)
print("\nNumber of episodes: {}\nNumber of steps: {}".format(agent.episode_cnt,
      agent.cum_steps[-1]))
agent.play()

# %% LunarLander-v2
envi = gym.make("LunarLander-v2")
precision = 6  # carefull here - we can easily run out of ram!! should be <= 10
treshold = 0.05
agent = SillyRLAgent(precision, envi, eps=0.15)
agent.learn_environment(episodes_num=500)
print()
agent.env_limits
agent.learn(0.05, 1, 100)
agent.plot_all(5)
print("\nNumber of episodes: {}\nNumber of steps: {}".format(agent.episode_cnt,
      agent.cum_steps[-1]))
agent.play()

# %% FrozenLake
envi = gym.make("FrozenLake-v0")
precision = 17
treshold = 0.05
agent = SillyRLAgent(precision, envi, eps=0.15)
agent.learn_environment(episodes_num=1000)
print()
agent.env_limits
agent.learn(-1, 10000, 1000000)
agent.plot_all(1)
print("\nNumber of episodes: {}\nNumber of steps: {}".format(agent.episode_cnt,
      agent.cum_steps[-1]))
agent.play()
agent.policy.policy
agent.policy.state_action_values
len(agent.value_improvements)
print()
agent.policy.values_improvement

# %% CliffWalking
# doesn't really work - default policy  makes it almost impossible for the
# episode to end
envi = gym.make("CliffWalking-v0")
precision = 17
treshold = 0.05
agent = SillyRLAgent(precision, envi, eps=0.15)
agent.learn_environment(episodes_num=10)
print()
agent.env_limits
agent.learn(-1, 10, 1000)
agent.plot_all(1)
print("\nNumber of episodes: {}\nNumber of steps: {}".format(agent.episode_cnt,
      agent.cum_steps[-1]))
agent.play()
agent.policy.policy
agent.policy.state_action_values
len(agent.value_improvements)
print()
agent.policy.values_improvement

# %% Blackjack
envi = gym.make("Blackjack-v0")
envi.observation_space
# Tuple(Discrete(32), Discrete(11), Discrete(2))
# silly agent cannot handle this

# %% KellyCoinflip
envi = gym.make("KellyCoinflip-v0")
envi.observation_space
# Tuple(Box(1,), Discrete(301))
# again to hard to use..

# %% Taxi
envi = gym.make("Taxi-v2")
# envi.observation_space
#   500
precision = 501
treshold = 0.05
agent = SillyRLAgent(precision, envi, eps=0.15)
agent.learn_environment(episodes_num=100)
print()
agent.env_limits
agent.learn(-1, 100, 10000)
agent.plot_all(1)
print("\nNumber of episodes: {}\nNumber of steps: {}".format(agent.episode_cnt,
      agent.cum_steps[-1]))
agent.play()
agent.policy.policy
agent.policy.state_action_values
len(agent.value_improvements)
print()
agent.policy.values_improvement
