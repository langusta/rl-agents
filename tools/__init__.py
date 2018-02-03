import numpy as np
import math


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
