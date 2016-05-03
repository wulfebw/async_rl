"""
Script for running an experiment.
"""

from multiprocessing import Manager
import time

import async_actor_critic
import experiment
import maze_mdp
import utils

# number of agents to simulate
# should equal number of cores
NUM_PROCESSES = 2

def run():
    # build the mdp
    start = time.time()
    room_size = 3
    num_rooms = 5
    mdp = maze_mdp.MazeMDP(room_size=room_size, num_rooms=num_rooms)

    # build the agent
    m = Manager()
    init_dict = {(s, a): 0 for s in mdp.states for a in mdp.ACTIONS + [None]}
    shared_weights = m.dict(init_dict)
    shared_value_weights = m.dict(init_dict)
    agent = async_actor_critic.AsyncActorCritic(actions=mdp.ACTIONS, discount=mdp.DISCOUNT, 
        weights=shared_weights, value_weights=shared_value_weights, tau=.3, learning_rate=.5)

    # build a single experiment
    rewards = m.list()
    start_state_values = m.list()
    max_steps = (2 * room_size * num_rooms) ** 2
    exp = experiment.Experiment(mdp=mdp, agent=agent, num_episodes=800, max_steps=max_steps,
        rewards=rewards, start_state_values=start_state_values)

    # run the experiment
    multiexperiment = experiment.MultiProcessExperiment(experiment=exp, num_agents=NUM_PROCESSES)
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