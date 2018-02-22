
Learning to train RL-agents.

The plan is to experiment with openai gym, implement some classic algorithms, try some of my own and some state of the art.

---

Some assumptions of how this works are given in the `__init__.py` files of `envs` and `agents` libraries. The heart is the `train` function from the `tools` library.

TODO:

- add option to plot episode lengths over episode number
- reproduce comparison of SARSA and QLearning from sutton
- for sutton.QlearningCh6, SarsaCh6 - maybe draw movements + changes in value function with time (to see how it updates)
- implement some simple agents (sutton)
- add some other agents and compare with silly one
- think of some ways of visualizing the insides of my current agents so that you can think of ways of improving them easier
...
- think of ways to parallel training different agents for comparisons (use all 4 threads)

DONE:
- check silly agent on some other env then CartPole
- look through my current silly agent
- think how to improve the code so that I can:
  - add new kinds of agents easily
  - compare different agents
- adjust training and plotting functions so that i can average results of many runs (and train many runs)
