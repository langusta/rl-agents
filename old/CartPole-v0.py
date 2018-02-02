import gym, time
import numpy as np

"""
TODO:
    learning - updating neighbours of given index (by half the reward or smth)

"""

class cartAgent(object):
    """Agent for CartPole-v0."""
    def __init__(self):
        super(cartAgent, self).__init__()
        self.learning_precision = 100
        self.x_velocity_treshold_values = np.linspace(0.01,1.0,self.learning_precision)
        self.x_velocity_treshold_weights = np.array([20.0]*self.learning_precision)
        self.angle_treshold_values = np.linspace(0.005,0.21,self.learning_precision)
        self.angle_treshold_weights = np.array([20.0]*self.learning_precision)
        self.gen_tresholds()

    def gen_tresholds(self):
        vel_probs = self.x_velocity_treshold_weights/sum(self.x_velocity_treshold_weights)
        angle_probs = self.angle_treshold_weights/sum(self.angle_treshold_weights)
        self.x_velocity_treshold_index = np.random.choice(len(self.x_velocity_treshold_values), p = vel_probs)
        self.x_velocity_treshold = self.x_velocity_treshold_values[self.x_velocity_treshold_index]
        self.angle_treshold_index = np.random.choice(len(self.angle_treshold_values), p = angle_probs)
        self.angle_treshold = self.angle_treshold_values[self.angle_treshold_index]
        return;

    def get_tresholds(self):
        return self.x_velocity_treshold, self.angle_treshold

    def print_probs(self):
        print("X Velocity weights: ", self.x_velocity_treshold_weights/sum(self.x_velocity_treshold_weights))
        print("Angle weights: ", self.angle_treshold_weights/sum(self.angle_treshold_weights))

    def learn(self, reward):
        self.x_velocity_treshold_weights[self.x_velocity_treshold_index] += reward
        self.angle_treshold_weights[self.angle_treshold_index] += reward
        self.gen_tresholds()

    def makeDecision(self, observation):
        if abs(observation[1]) < self.x_velocity_treshold:
            if abs(observation[2]) < self.angle_treshold:
                action = 0 if observation[3] < 0 else 1
            else:
                action = 0 if observation[2] < 0 else 1
        else: # x velocity
            action = 1 if observation[1] < 0 else 0
        return action


def simulate(env, agent, render = False, should_print = False):
    observation = env.reset()
    cumulative_reward = 0
    for t in range(200):
        if render:
            env.render()
        action = agent.makeDecision(observation)
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if should_print:
            print(observation)
        if done:
            break
    return cumulative_reward, t+1, observation

envir = gym.make('CartPole-v0')
cart = cartAgent()
for episode in range(5000):
    cumulative_reward, time, observation = simulate(envir, cart)
    cart.learn(cumulative_reward)
    vel_tresh, angle_tresh = cart.get_tresholds()
    if episode % 50 == 0:
        print("Episode: {:3}, cumulative reward: {:4.0f}, timesteps: {:3}, last observation: {}, v_tresh: {:0.4f}, a_tresh: {:0.4f}".format(episode, cumulative_reward, time, observation, vel_tresh, angle_tresh))

simulate(envir, cart, True)
cart.print_probs()
