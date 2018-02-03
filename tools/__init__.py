import numpy as np
import math
import os
import pandas as pd


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


def moving_average(what, length):
    out = np.cumsum(what)
    out[length:] = out[length:] - out[:-length]
    return out[length - 1:] / length


def train(agent, env, debug=False):
    """
        train agent in given env
    """
    history = []
    while True:
        state = env.reset()
        the_return = 0
        stop = False
        while not stop:
            action = agent.act(state)
            new_state, reward, stop, _ = env.make_step(action)
            agent.learn(state, action, reward, stop)
            if debug:
                history.append((state, action, reward))
            the_return += reward
            state = new_state
        if env.finished():
            break
    return agent, history


def save_results(agent, environment,
                 force=False,
                 base='/home/langust/Research/AIS/rl-agents'):
    folder = base + "/results/" + environment.name
    file_name = environment.full_name + "__" + agent.full_name + ".csv"
    if not (os.path.exists(folder) and os.path.isdir(folder)):
        os.mkdir(folder)
    if not (os.path.exists(folder + "/" + file_name)) or force:
        value_improvements = agent.value_improvements
        if len(agent.value_improvements) < 1:
            value_improvements = [0] * len(agent.steps)
        data = pd.DataFrame({
            "steps": agent.steps,
            "returns": agent.returns,
            "value_improvements": value_improvements})
        data.to_csv(folder + "/" + file_name)


# to consider - should the original save function save the whole agent?
def load_results():
    pass


from envs.msc import MinimalEnv
from agents.sutton import QlearningCh6
from tools import train

agent, history = train(QlearningCh6(3, 2, 0.9, 0.1, 0.2), MinimalEnv(),
                       debug=True)
agent.steps
save_results(agent, MinimalEnv(), force=True)
stuf = pd.DataFrame.from_csv("results/MinimalEnv/MinimalEnv-v0__QlearningCh6-v0.csv")
stuf.head()
