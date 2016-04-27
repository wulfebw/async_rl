"""
Utilities used by the algorithm and for plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import random

CONVERGENCE_THRESHOLD = 1e-4
MIN_CONVERGENCE_VALUES = 15

# credit: stanford.cs221.problem_set_6
# http://web.stanford.edu/class/cs221/
def weightedRandomChoice(weightDict):
    weights = []
    elems = []
    inf_elems = []
    for elem in weightDict:
        w = weightDict[elem]
        if w == float('inf'):
            inf_elems.append(elem)
        else:
            weights.append(w)
            elems.append(elem)

    # because this learning algorithm updates the weights without 
    # locking them first, they can occasionally be corrupted
    # if this happens, then replace corrupted values with 
    # average remaining probability
    if len(inf_elems) > 0:
        total = sum([weightDict[elem] for elem in elems if elem not in inf_elems])
        remaining_prob = (1. - total) / len(inf_elems)

        for elem in inf_elems:
            weights.append(remaining_prob)
            elems.append(elem)

    total = sum(weights)
    key = random.uniform(0, total)
    runningTotal = 0.0
    chosenIndex = None
    for i in range(len(weights)):
        weight = weights[i]
        runningTotal += weight
        if runningTotal > key:
            chosenIndex = i
            return elems[chosenIndex]

def have_converged(values):
    if len(values) < MIN_CONVERGENCE_VALUES:
        return False
    max_v = max(values[-MIN_CONVERGENCE_VALUES:])
    min_v = min(values[-MIN_CONVERGENCE_VALUES:])
    if max_v - min_v < CONVERGENCE_THRESHOLD:
        return True
    return False

def plot_values(values, optimal, ylabel):
    values = values[:(len(values) / 4) * 4]
    values = np.mean(np.reshape(values, (-1, 4)), axis=1).reshape(-1)
    plt.scatter(range(len(values)), values)
    plt.axhline(optimal, c='k', linestyle='dashed')
    plt.xlabel('episodes (1 per actor-learner)')
    plt.ylabel(ylabel)
    plt.show()