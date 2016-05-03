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
        # should really be returning probs[action_idx]
        # sum_exp_q_values = np.sum(exp_q_values)
        # probs = exp_q_values / sum_exp_q_values
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
        # prediction = V(s)
        prediction = self.getV(state)
        target = reward
        new_action = None

        if new_state != None:
            new_action = self.get_action(new_state)
            # target = r + yV(s')
            target += self.discount * self.getV(new_state)
        
        # advantage actor critic because we use the td error
        # as an unbiased sample of the advantage function
        update = self.learning_rate * (target - prediction)
        for f, v in self.feature_extractor(state):
            # update critic weights
            self.value_weights[f] = self.value_weights[f] + 2 * update

        for f, v in self.feature_extractor(state, action):
            # update actor weights
            # this update should actually be:
            # self.weights[f] += update * (v - prob(v))
            # since (v - prob(v)) is, in this case, equal to
            # the gradient of the log of the policy
            # however, that seems to work way worse than simply
            # multiplying by v (i.e., 1) instead, though it's likely
            # this version loses convergence gaurantees
            # and / or would work poorly with a neural net
            self.weights[f] = self.weights[f] + update * v

        return new_action
