"""
filename: async_rl.py
author: blake wulfe
date: 4/27/16

Discussion:
This file contains a process-based implementation of tabular, 1-step,
asynchronous actor critic. It's pretty different from the A3C 
algorithm from the paper: http://arxiv.org/abs/1602.01783
in that it does not use a function approximator and in that it
does not incorporate (the forward view) of eligibility traces.

It is similar in that it implements actor critic
with multiple agents updating weights in parallel. 
Whether or not running with muiltiple processes helps
seems to depend upon the specific maze MDP you use.

For example:
single process:
2x5 maze: 5.4
2x10 maze: 21.9
1x30 maze: 49.2
5x3 maze: 11.9

two processes:
2x5 maze: 3.5 seconds
2x10 maze: 22.1 seconds
1x30 maze: 79.2
5x3 maze: 18.5

Implementation Description:
This implementation does the following
1. instantiates an agent, a mdp, and an experiment
2. instantiates a multiexpirement, passing in the experiment from (1)
3. calls run on the multiexpirement, which forks NUM_PROCESSES processes,
    that simulate runs in parallel and update shared weights managed by 
    pythons multiprocessing.Manager class. This class doesn't implement 
    locks by default, so the weight values are basically randomly updated. 
    The reason this uses processes instead of threads as in the original 
    paper is that the python GIL makes it so that only a single python 
    thread can execute at a time, which defeats the purpose of the algorithm.
4. each of the agents being simulated stops when the value of the
    start state converges
5. plots of the rewards and start state values over time are displayed
"""

from multiprocessing import Manager, Process
import numpy as np
import time

import utils
import maze_mdp

# number of agents to simulate
# should equal number of cores
NUM_PROCESSES = 2

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

class AsyncActorCritic(object):
    """
    An actor critic agent. 
    """

    def __init__(self, actions, discount, weights, value_weights, tau, learning_rate):
        self.actions = actions
        self.discount = discount
        self.weights = weights
        self.value_weights = value_weights
        self.tau = tau
        self.learning_rate = learning_rate
        self.num_iters = 0

    def feature_extractor(self, state, action=None):
        return [((state, action), 1)]

    def getV(self, state):
        score = 0
        for f, v in self.feature_extractor(state):
            score += self.value_weights[f] * v
        return score

    def getQ(self, state, action):
        score = 0
        for f, v in self.feature_extractor(state, action):
            score += self.weights[f] * v
        return score

    def get_action(self, state):
        """
        Softmax action selection.
        """
        self.num_iters += 1
        q_values = np.array([self.getQ(state, action) for action in self.actions])
        q_values = q_values - max(q_values)
        exp_q_values = np.exp(q_values / (self.tau + 1e-2))
        weights = dict()
        for idx, val in enumerate(exp_q_values):
            weights[idx] = val
        action_idx = utils.weightedRandomChoice(weights)
        action = self.actions[action_idx]
        return action

    def incorporateFeedback(self, state, action, reward, new_state):
        """
        Update both actor and critic weights.
        """
        prediction = self.getV(state)
        target = reward
        new_action = None

        if new_state != None:
            new_action = self.get_action(new_state)
            target += self.discount * self.getV(new_state)

        update = self.learning_rate * (target - prediction)
        for f, v in self.feature_extractor(state):
            self.value_weights[f] = self.value_weights[f] + 2 * update

        for f, v in self.feature_extractor(state, action):
            self.weights[f] = self.weights[f] + update * 1

        return new_action

def run():
    # build the mdp
    start = time.time()
    room_size = 5
    num_rooms = 2
    mdp = maze_mdp.MazeMDP(room_size=room_size, num_rooms=num_rooms)

    # build the agent
    m = Manager()
    init_dict = {(s, a): 0 for s in mdp.states for a in mdp.ACTIONS + [None]}
    shared_weights = m.dict(init_dict)
    shared_value_weights = m.dict(init_dict)
    agent = AsyncActorCritic(actions=mdp.ACTIONS, discount=mdp.DISCOUNT, 
        weights=shared_weights, value_weights=shared_value_weights, tau=.3, learning_rate=.5)

    # build a single experiment
    rewards = m.list()
    start_state_values = m.list()
    max_steps = (2 * room_size * num_rooms) ** 2
    experiment = Experiment(mdp=mdp, agent=agent, num_episodes=800, max_steps=max_steps,
        rewards=rewards, start_state_values=start_state_values)

    # run the experiment
    multiexperiment = MultiProcessExperiment(experiment=experiment, num_agents=NUM_PROCESSES)
    multiexperiment.run()

    # report results
    end = time.time()
    print 'took {} seconds to converge'.format(end - start)
    mdp.print_state_values(shared_value_weights)
    optimal = mdp.EXIT_REWARD + (2 * room_size * num_rooms * mdp.MOVE_REWARD)
    utils.plot_values(rewards, optimal, 'rewards')
    utils.plot_values(start_state_values, optimal, 'start state value')

if __name__ =='__main__':
    run()

