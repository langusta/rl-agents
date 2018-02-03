from collections import namedtuple

version = "0.1.0"
available_envs = {
    "msc": ["MinimalEnv"],
    "openai": []
}
idea = """
States, actions, rewards:
    state:
        is a numpy array (one dimension, can be loooong)
        ex.:
        np.zeros(10)
        array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
    action:
        is a numpy array (one dimension)
        ex.:
        np.arange(5)
        array([0, 1, 2, 3, 4])
    reward:
        is a number (float)
        ex.:
        1
        3.14
        np.exp(1)

The interface is inspired by openai gym:
    env = MyEnv("version")
    # "version" is optional
    env.state.length
    # ex.: 5
    env.state.type
    # a list of length env.state.length:
    # [0, 0, 2, 2, 3]
    # meaning of its elements:
    #   0 - it's a continuous coordinate (and we don't know its limits)
    #   number > 0 - it's, a discrete coordinate
    #       taking values: 0, 1, ..., number - 1
    env.action.length
    # ex.: 1
    env.action.type
    # same as with env.state.type
    first_state = env.reset()
    action = agent(...)
    state, reward, done, info = env.make_step(action)

    # env should know if enough learning has been done:
    env.finished()
    # returns True or False
"""

SpaceDesc = namedtuple("SpaceDesc", ["length", "type"])


class BaseEnv:
    """
    All envs should implement this class.
    """

    def __init__(self, ver="v0"):
        self.name = "BaseEnv"
        self.version = ver
        self.full_name = self.name + "-" + ver
        self.state = SpaceDesc(0, [])
        self.action = SpaceDesc(0, [])
        pass

    def reset(self):
        pass

    def make_step(self, action):
        pass

    def finished(self):
        pass
