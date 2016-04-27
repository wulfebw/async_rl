This repo contains a process-based implementation of tabular, 1-step,
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