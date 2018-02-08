import numpy as np
from envs import SpaceDesc
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class CliffEnv:
    """
     See sutton example 6.6 (page 140).
    """

    def __init__(self, ver="v0"):
        self.name = "CliffEnv"
        self.version = ver
        self.full_name = self.name + "-" + ver

        self.width = 6
        self.height = 3
        self.states = self.width * self.height
        self.state = SpaceDesc(1, [self.states])
        self.action = SpaceDesc(1, [self.states])
        self.start = (self.height - 1, 0)  # (row, column)
        self.state = self.start
        self.action_mod = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.step = 0
        self.episode = 0
        self.max_episode = 500

    def out_state(self):
        y, x = self.state
        return np.array([y * self.width + x])

    def reset(self):
        self.state = self.start
        return self.out_state()

    def make_step(self, action):
        """
            action[0]:
            0 - up
            1 - right
            2 - down
            3 - left
            ... - does nothing, reward -1
        """
        if action[0] not in [0, 1, 2, 3]:
            st = self.reset()
            return (st, -100, True, 0)

        dy, dx = self.action_mod[action[0]]
        y, x = self.state
        self.state = (y + dy, x + dx)

        reward = -1
        if self.state[0] < 0:
            self.state = (0, self.state[1])
        elif self.state[0] >= self.height:
            self.state = (self.height - 1, self.state[1])
        elif self.state[1] < 0:
            self.state = (self.state[0], 0)
        elif self.state[1] >= self.width:
            self.state = (self.state[0], self.width - 1)
        elif (self.state[0] == self.height - 1 and
              self.state[1] > 0 and self.state[1] < self.width - 1):
            reward = -100
            self.state = self.start

        stop = self.state == (self.height - 1, self.width - 1)
        self.step += 1
        if stop:
            self.episode += 1
        return self.out_state(), reward, stop, 0

    def finished(self):
        if self.episode > self.max_episode:
            return True
        return False

    def get_map(self):
        out = np.zeros((self.height, self.width))
        out[self.state] = 1
        return out

    def animate(self, actions):
        ims = []
        ims.append([plt.imshow(self.get_map(), animated=True)])
        for action in actions:
            _ , _, stop, _ = self.make_step(action)
            ims.append([plt.imshow(self.get_map(), animated=True)])
            if stop:
                self.reset()
                ims.append([plt.imshow(self.get_map(), animated=True)])
        return ims


# %% Tests:
#
# env = CliffEnv()
# state = env.reset()
# some_actions = [[a] for a in [3, 2, 1, 0, 1, 2, 0, 0, 1, 1, 1, 1, 1, 2, 2,
#                               0, 0, 3, 3, 3, 2, 1, 2, 2, 0]]
#
# # outs = [env.make_step(a) for a in some_actions]
# ims = env.animate(some_actions)
#
# # fig = plt.figure()
# ani = animation.ArtistAnimation(plt.gcf(), ims, interval=200, repeat_delay=500,
#                                 blit=True)
# # plt.show()
# ani.save('CliffWalkingTest.mp4')
