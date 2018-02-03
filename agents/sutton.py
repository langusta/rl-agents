import numpy as np
from agents import BaseAgent


class QlearningCh6(BaseAgent):
    """
        Q-learning algorithm from chapter 6
        Works for discrete state spaces and discrete actions.
        State and action is an array of length 1, containing
            respective number of state or action.
    """
    def __init__(self, max_states, max_actions, gamma, eps, alpha):
        super().__init__()
        self.name = "QlearningCh6"

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
        if self.prev[0] is not None:
            self.q[self.prev[0][0], self.prev[1][0]] += self.alpha * \
                (self.prev[2] + self.gamma
                 * self.q[state[0], :].max()
                 - self.q[self.prev[0][0], self.prev[1][0]])
        if not stop:
            self.prev = (state, action, reward)
        else:
            self.prev = (None, None, None)
