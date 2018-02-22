import numpy as np
import math
import os
import pandas as pd
import matplotlib.pyplot as plt


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
                 del_if_exist=False,
                 base='/home/langust/Research/AIS/rl-agents'):
    folder = base + "/results/" + environment.name
    file_name = environment.full_name + "__" + agent.full_name + ".csv"
    full_path = folder + "/" + file_name
    header = True
    if not (os.path.exists(folder) and os.path.isdir(folder)):
        os.mkdir(folder)
        run = 0
    else:
        if os.path.exists(full_path):
            if del_if_exist:
                run = 0
                os.remove(full_path)
            else:
                tmp = pd.read_csv(full_path)
                run = tmp.run.max() + 1
                header = False
        else:
            run = 0
    value_improvements = agent.value_improvements
    if len(agent.value_improvements) < 1:
        value_improvements = [0] * len(agent.steps)
    data = pd.DataFrame({
        "run": run,
        "episodes": range(len(agent.steps)),
        "steps": agent.steps,
        "returns": agent.returns,
        "value_improvements": value_improvements})
    data.to_csv(full_path, mode="a", index=False, header=header)


def load_results(agent, environment,
                 base='/home/langust/Research/AIS/rl-agents'):
    """returns Pandas DataFrame"""
    full_path = base + '/results/' + environment.name + \
        '/' + environment.full_name + '__' + \
        agent.full_name + '.csv'
    if os.path.exists(full_path):
        return pd.read_csv(full_path)
    else:
        print("No results found for: " + agent.full_name)
        return pd.DataFrame()


def plot_multiple(*args, figsize=(8, 6)):
    plt.subplots(figsize=figsize)
    for x, y, label in args:
        plt.plot(x, y, label=label)
    leg = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                     ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.7)
    plt.show()


def plot_agents(agents, environment, what='rv', x_ax='s', figsize=(8, 6)):
    rets = []
    vals = []
    for agent in agents:
        res = load_results(agent, environment)
        if x_ax == 's':
            res['x'] = res.steps
        else:
            res['x'] = res.episodes
        r_agg = res.groupby("episodes").agg({'x': 'mean',
                                             'returns': 'mean',
                                             'value_improvements': 'mean'})
        cnt = str(res.run.max() + 1)  # run starts with 0

        rets.append((r_agg.x, r_agg.returns, agent.full_name + " av" + cnt))
        vals.append((r_agg.x, r_agg.value_improvements, agent.full_name + " av" + cnt))
    if 'v' in what:
        plot_multiple(*vals, figsize=figsize)
    if 'r' in what:
        plot_multiple(*rets, figsize=figsize)



def save_agent(agent, environment, force=False,
               base='/home/langust/Research/AIS/rl-agents'):
    """Save entire agent"""
    pass
