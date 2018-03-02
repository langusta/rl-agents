import numpy as np
from envs.sutton import CliffEnv
from agents.sutton import QlearningCh6, SarsaCh6
from tools import train, save_results, plot_agents
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# %% let's test CliffWalking
# figure 6.5 from sutton
alpha = 0.5
eps = 0.1
del_ = False
for _ in range(20):
    cenv = CliffEnv()
    cenv.max_episode = 500
    qlearn, qhistory = train(QlearningCh6(48, 4, 1.0, eps, alpha), cenv, debug=True)
    cenv = CliffEnv()
    cenv.max_episode = 500
    sarsa, shistory = train(SarsaCh6(48, 4, 1.0, eps, alpha), cenv, debug=True)
    save_results(qlearn, CliffEnv(), del_if_exist=del_)
    save_results(sarsa, CliffEnv(), del_if_exist=del_)
plot_agents([qlearn, sarsa], CliffEnv(), what='r', x_ax='e', mov_av=10, ymin=-100, ymax=0)


# %% save videos:
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
