import numpy as np
from agents import BaseAgent


class Sutton(BaseAgent):
    """
        Q-learning algorithm from chapter 6
        Works for discrete state spaces and discrete actions.
        State and action is an array of length 1, containing
            respective number of state or action.
    """
    def __init__(self, max_states, max_actions, gamma, eps, alpha,
                 ver="v0", name="Sutton"):
        super().__init__(ver)
        self.name = name
        self.full_name = name + "-" + ver

        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.q = np.zeros((max_states, max_actions))
        self.prev = (None, None, None)  # (state, action, reward)

    def act(self, state, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if np.random.rand() < self.eps:
            return [np.random.randint(self.q.shape[1])]
        else:
            return [self.q[state[0], :].argmax()]

    def learn(self, state, action, reward, stop):
        super().learn(state, action, reward, stop)


class QlearningCh6(Sutton):
    def __init__(self, max_states, max_actions, gamma, eps, alpha,
                 ver="v0"):
        super().__init__(max_states, max_actions, gamma, eps, alpha, ver,
                         name="QlearningCh6")

    def learn(self, state, action, reward, stop):
        super().learn(state, action, reward, stop)
        if self.prev[0] is not None:
            pstate, paction, preward = self.prev  # p... == previous ...
            self.q[pstate[0], paction[0]] += self.alpha * \
                (preward + self.gamma
                 * self.q[state[0], :].max()
                 - self.q[pstate[0], paction[0]])
        if not stop:
            self.prev = (state, action, reward)
        else:
            self.prev = (None, None, None)


class SarsaCh6(Sutton):
    def __init__(self, max_states, max_actions, gamma, eps, alpha,
                 ver="v0"):
        super().__init__(max_states, max_actions, gamma, eps, alpha, ver,
                         name="SarsaCh6")

    def learn(self, state, action, reward, stop):
        super().learn(state, action, reward, stop)
        if self.prev[0] is not None:
            pstate, paction, preward = self.prev  # p... == previous ...
            self.q[pstate[0], paction[0]] += self.alpha * \
                (preward + self.gamma
                 * self.q[state[0], action[0]]
                 - self.q[pstate[0], paction[0]])
        if not stop:
            self.prev = (state, action, reward)
        else:
            self.prev = (None, None, None)
