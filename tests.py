import numpy as np
from envs.msc import MinimalEnv
from envs.sutton import CliffEnv
from agents.sutton import QlearningCh6, SarsaCh6
from tools import train, save_results, plot_agents
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# %% train some agent

agent, history = train(QlearningCh6(3, 2, 0.9, 0.1, 0.2), MinimalEnv(),
                       debug=True)

agent.q
agent.returns
history

# %% draw results

qlearner, history = train(QlearningCh6(3, 2, 0.9, 0.1, 0.2), MinimalEnv(),
                       debug=True)
sarsa, history = train(SarsaCh6(3, 2, 0.9, 0.2, 0.2), MinimalEnv(),
                        debug=True)
save_results(qlearner, MinimalEnv())
save_results(sarsa, MinimalEnv())
plot_agents([qlearner, sarsa], MinimalEnv(), what='r')

qlearner.q
sarsa.q
