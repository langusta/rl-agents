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
save_results(qlearner, MinimalEnv(), force=True)
save_results(sarsa, MinimalEnv(), force=True)
plot_agents([qlearner, sarsa], MinimalEnv(), what='r')

qlearner.q
sarsa.q

# %% let's test CliffWalking
cenv = CliffEnv()
cenv.max_episode = 50
qlearn, qhistory = train(QlearningCh6(18, 4, 1.0, 0.1, 0.3), cenv, debug=True)
cenv = CliffEnv()
cenv.max_episode = 50
sarsa, shistory = train(SarsaCh6(18, 4, 1.0, 0.1, 0.3), cenv, debug=True)
save_results(qlearn, CliffEnv(), force=True)
save_results(sarsa, CliffEnv(), force=True)
plot_agents([qlearn, sarsa], CliffEnv(), what='r')

# qlearn.q
# sarsa.q
len(qhistory)
len(shistory)

qactions = [a[1] for a in qhistory]
sactions = [a[1] for a in shistory]
env = CliffEnv()
qims = env.animate(qactions)
ani = animation.ArtistAnimation(plt.gcf(), qims, interval=50, repeat_delay=500,
                                blit=True)
ani.save('CliffWalkingQlerner.mp4')
env = CliffEnv()
sims = env.animate(sactions)
ani = animation.ArtistAnimation(plt.gcf(), sims, interval=50, repeat_delay=500,
                                blit=True)
ani.save('CliffWalkingSarsa.mp4')

qlearn.q.argmax(-1).reshape(3, 6)
print()
sarsa.q.argmax(-1).reshape(3, 6)
