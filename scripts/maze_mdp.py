"""
This file defines a simple MDP on which to test the a3c implementation.
"""

import numpy as np

class MazeMDP(object):
    """
    A MDP specifying a maze, where that maze is a square, 
    consists of num_rooms and each room having room_size discrete squares
    in it. So can have 1x1, 2x2, 3x3, etc size mazes. Rooms are separated
    by walls with a single entrance between them. The start state is 
    always the bottom left of the maze. The end state is always in the top 
    right room of the maze. So the 1x1 maze looks like:

     _______
    |      E|
    |       |
    |       |
    |S      |
     -------

    the 2x2 maze would be

     _______ _______   
    |       |      E|
    |               |
    |               |
    |       |       |
     --   -- --   --
     __   __ __   __
    |       |       |
    |               |
    |               |
    |S      |       |
     ------- -------


    state is represented in absolute terms, so the bottom left corner 
    of all mazes is (0,0) and to top right corner of all mazes is 
    (room_size * num_rooms - 1, room_size * num_rooms - 1). In other 
    words, the state ignores the fact that there are rooms or walls 
    or anything, it's just the coordinates.

    actions are N,E,S,W movement by 1 direction. No stochasticity.
    moving into a wall leaves agent in place. Rewards are nothing 
    except finding the exit is worth a lot 

    room_size must be odd
    """
   
    EXIT_REWARD = 1
    MOVE_REWARD = -.01
    ACTIONS = [(1,0),(-1,0),(0,1),(0,-1)] 
    DISCOUNT = 1
    START_STATE = (0,0)

    def __init__(self, room_size, num_rooms):
        self.room_size = room_size
        self.num_rooms = num_rooms
        self.max_position = self.room_size * self.num_rooms - 1
        self.end_state = (self.max_position, self.max_position) 
        self.computeStates() 

    def calculate_next_state(self, state, action):
        return state[0] + action[0], state[1] + action[1]

    def runs_into_wall(self, state, action):
        next_state = self.calculate_next_state(state, action)

        # 1. check for leaving the maze
        if next_state[0] > self.max_position or next_state[0] < 0 \
                            or next_state[1] > self.max_position or next_state[1] < 0:
            return True

        # 2. check if movement was through doorway and if so return false
        doorway_position = (self.room_size) / 2
        # check horizontal movement through doorway
        if next_state[0] != state[0]:
            if next_state[1] % self.room_size == doorway_position:
                return False

        # check vertical movement through doorway
        if next_state[1] != state[1]:
            if next_state[0] % self.room_size == doorway_position:
                return False

        # 3. check if movement was through a wall
        room_size = self.room_size
        # move right to left through wall
        if state[0] % room_size == room_size - 1 and next_state[0] % room_size == 0:
            return True

        # move left to right through wall
        if next_state[0] % room_size == room_size - 1 and state[0] % room_size == 0:
            return True

        # move up through wall
        if state[1] % room_size == room_size - 1 and next_state[1] % room_size == 0:
            return True

        # move down through wall
        if next_state[1] % room_size == room_size - 1 and state[1] % room_size == 0:
            return True

        # if none of the above conditions meet, then have not passed through wall
        return False

    def succAndProbReward(self, state, action): 

        # if we reach the end state then the episode ends
        if np.array_equal(state, self.end_state):
            return []

        if self.runs_into_wall(state, action):
            # if the action runs us into a wall do nothing
            next_state = state
        else:
            # o/w determine the next position
            next_state = self.calculate_next_state(state, action)

        # if next state is exit, then set reward
        reward = self.MOVE_REWARD
        if np.array_equal(next_state, self.end_state):
            reward = self.EXIT_REWARD

        return [(next_state, 1, reward)]

    # credit: stanford.cs221.problem_set_6
    # http://web.stanford.edu/class/cs221/
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.START_STATE)
        queue.append(self.START_STATE)
        while len(queue) > 0:
            state = queue.pop()
            for action in self.ACTIONS:
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)

    def print_state_values(self, value_weights):
        print """\nThe following are the learned values of the states in the maze.\n Note that the agent really only learns a single path from the start state\n in the bottom left to the end state in the top right.\n"""
        V = {}
        for state in self.states:
            state_value = value_weights[(state, None)]
            V[state] = state_value

        for ridx in reversed(range(self.max_position + 1)):
            for cidx in range(self.max_position + 1):
                if (ridx, cidx) in V:
                    print '{0:.4f}'.format(V[(ridx, cidx)]),
            print('\n')