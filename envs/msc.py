import numpy as np
from envs import SpaceDesc


class MinimalEnv:
    """
    We have three states:
    0   - rewards: -1 for up, -2 for down
    1   - rewards: -1 for up, 0 for down
    2   - terminal state
    and two actions: 0 - go up, 1 go down
    """
    def __init__(self, ver="v0"):
        self.name = "MinimalEnv"
        self.version = ver
        self.full_name = self.name + "-" + ver
        self.state_desc = SpaceDesc(length=1, type=[3])
        self.action_desc = SpaceDesc(length=1, type=[2])
        self.state = 0
        self.step = 0
        self.episode = 0

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
        return np.array([0])

    def make_step(self, action):
        if action[0] not in [0, 1]:
            self.state = 0
            return (2, 0, True, 0)
        self.state, reward = self.behaviour[(self.state, action[0])]
        self.step += 1
        if self.state == 2:
            self.episode += 1
        return (np.array([self.state]), reward, self.state == 2, 0)

    def finished(self):
        if self.step > 1000 or self.episode > 100:
            return True
        return False


# %% some tests:
# minimal_env = MinimalEnv()
# minimal_env.reset()
# list(map(minimal_env.step, [[x] for x in [0, 1, 0, 1, 1]]))
# # [(array([0]), -1, False, 0),
# #  (array([1]), -2, False, 0),
# #  (array([0]), -1, False, 0),
# #  (array([1]), -2, False, 0),
# #  (array([2]), 0, True, 0)]
