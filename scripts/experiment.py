"""
This file contains implementations for the Experiment and MultiProcessExperiment
classes.
"""


from multiprocessing import Process
import numpy as np

import utils

class Experiment(object):
    """
    Simulates a single experiment with a single agent updating shared weights.
    """

    def __init__(self, mdp, agent, num_episodes, max_steps, rewards, start_state_values):
        self.mdp = mdp
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.rewards = rewards
        self.start_state_values = start_state_values

    def run(self, agent_id):
        print 'running experiment with agent number {}...'.format(agent_id)

        total_rewards = []
        total_reward = 0

        for episode in range(self.num_episodes):

            if utils.have_converged(self.start_state_values):
                break

            if episode % 100 == 0:
                print 'running episode {} for agent {}...'.format(episode, agent_id)
            state = self.mdp.START_STATE
            action = self.agent.get_action(state)

            for step in range(self.max_steps):
                transitions = self.mdp.succAndProbReward(state, action)

                if len(transitions) == 0:
                    reward = 0
                    new_state = None
                    break

                new_state, prob, reward = transitions[0]
                total_reward += reward
                action = self.agent.incorporateFeedback(state, action, reward, new_state)
                state = new_state

            self.agent.incorporateFeedback(state, action, reward, new_state)
            total_rewards.append(total_reward)
            self.rewards.append(total_reward)
            self.start_state_values.append(self.agent.value_weights[((0,0), None)])
            total_reward = 0
        
        print 'average reward of agent {}: {}'.format(agent_id, np.mean(total_rewards))

class MultiProcessExperiment(object):
    """
    Runs experiments in parallel on different processes.
    """

    def __init__(self, experiment, num_agents):
        self.experiment = experiment
        self.num_agents = num_agents

    def run(self):

        processes = []
        for idx in range(self.num_agents):
            p = Process(target=self.run_experiement, args=(self.experiment, idx))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    @staticmethod
    def run_experiement(experiment, agent_id):
        experiment.run(agent_id)