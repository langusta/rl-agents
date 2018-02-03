
version = "0.1.0"
available_agents = {
    "sutton": ["QlearningCh6"],
    "my_agents": []
}
idea = """
What agents can do:

from agents import SomeAgent

agent = SomeAgent()
...
    agent.act(state)
    # state is np.array(...)
    agent.learn(state, action, reward, stop)
    # action is np.array
    # reward is a float number
    # stop is a flag (True/False) if the state is terminal

    plt.plot(agent.steps, agent.returns)
    # if the agent has been learning for:
    #   episodes: 0, 1, 2
    #   steps: 11, 15, 9  # in corresponding episodes
    # and got:
    #   returns: -11, -15, -9
    # then:
    #   agent.steps == [11, 26, 35]
    #       # 11, 11+15, 11+15+9
    #   agent.returns == [-11, -15, -9]
    plt.plot(np.arange(len(agent.returns)), agent.returns)
    # np.arange(len(agent.returns)) is array with episode numbers
    plt.plot(agent.steps, agent.value_improvements)

    agent.name  # is its name
    agent.step  # is the last step
    agent.episode  # is the last episode of learning

    # OPTIONAL:
    agent.save("some_file")
    agent.load("some_file")
    play(agent, environment)
"""


class BaseAgent:
    """
    All agents should implement this class.
    """
    def __init__(self, ver="v0"):
        self.name = "BaseAgent"
        self.version = ver
        self.full_name = self.name + "-" + ver
        self.step = 0
        self.steps = []
        self.episode = 0
        self.value_improvements = []
        self.returns = []
        self.the_return = 0

    def act(self, state):
        pass

    def learn(self, state, action, reward, stop):
        self.step += 1
        self.the_return += reward
        if stop:
            self.episode += 1
            self.returns.append(self.the_return)
            if len(self.steps) > 0:
                self.steps.append(self.step + self.steps[-1])
            else:
                self.steps.append(self.step)
            self.the_return = 0
