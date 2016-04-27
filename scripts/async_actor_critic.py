"""
This file contains the implementation of asynchronous actor critic.
"""

import numpy as np

import utils

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
