## Summary
This repo contains a process-based implementation of tabular, 1-step,
asynchronous actor critic. It's pretty different from the A3C 
algorithm from the Asynchronous Methods for Deep 
Reinforcement Learning paper: http://arxiv.org/abs/1602.01783
in that it does not use a function approximator and in that it
does not incorporate (the forward view) of eligibility traces.

It is similar in that it implements actor critic
with multiple agents updating weights in parallel.

There's also a simple test maze markov decision process (MDP).
Whether or not running with muiltiple processes is beneficial
seems to depend upon the specific maze MDP you use.

## Results
### single process:
- 2x5 maze: 5.4 seconds
- 2x10 maze: 21.9 seconds
- 1x30 maze: 49.2 seconds
- 5x3 maze: 11.9 seconds

### two processes:
- 2x5 maze: 3.5 seconds
- 2x10 maze: 22.1 seconds
- 1x30 maze: 79.2 seconds
- 5x3 maze: 18.5 seconds

## File Descriptions
- run_experiment.py: Script for running an experiment.
- async_actor_critic.py: Contains the implementation of asynchronous actor critic.
- experiment.py: Contains implementations for the Experiment and MultiProcessExperiment
classes.
- maze_mdp.py: Defines a simple MDP on which to test the actor critic implementation.
- utils.py: Utilities used by the algorithm and for plotting.
