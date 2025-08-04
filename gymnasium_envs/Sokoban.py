from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gym.core import ActType, ObsType, RenderFrame
from numpy.random import default_rng

import sys
import time
import os

class Environment:

    # define the structure of the environment - 11 cells laid out as below
    # 0   1
    # 2   3   4   5
    #     6   7   8
    #         9   10

    NUM_CELLS = 11
    AGENT_START = 1
    BOX_START = 3
    AGENT_GOAL = 10

    # map of the environment : -1 indicates a wall
    # assumes directions ordered as 0 = up, 1 = right, 2 = down, 3 = left
    MAP = [[-1, 1, 2, -1],  # transitions from cell 0 when doing actions 0,1,2,3
    [-1, -1, 3, 0], # transitions from cell 1 when doing actions 0,1,2,3
    [0, 3, -1, -1], # transitions from cell 2 when doing actions 0,1,2,3
    [1, 4, 6, 2], # transitions from cell 3 when doing actions 0,1,2,3
    [-1, 5, 7, 3], # transitions from cell 4 when doing actions 0,1,2,3
    [-1, -1, 8, 4], # transitions from cell 5 when doing actions 0,1,2,3
    [-3, 7, -1, -1], # transitions from cell 6 when doing actions 0,1,2,3
    [4, 8, 9, 6], # transitions from cell 7 when doing actions 0,1,2,3
    [5, -1, 10, 7], # transitions from cell 8 when doing actions 0,1,2,3
    [7, 10, -1, -1], # transitions from cell 9 when doing actions 0,1,2,3
    [8, -1, -1, 9]] # transitions from cell 10 when doing actions 0,1,2,3

    # penalty term used in the performance reward based on the final box location
    # -50 if the box is in a corner, -25 if its next to a wall
    BOX_PENALTY = [-50, -50, -50, 0, -25, -50, -50, 0, -25, -50, -50]

    # The following variables are not necessary in our implementation
    # They specify the priority of objectives starting from the goal
    NUM_OBJECTIVES = 3
    GOAL_REWARD = 0
    IMPACT_REWARD = 1
    PERFORMANCE_REWARD = 2

    def __init__(self):
        # state variables
        self.agent_location = self.AGENT_START
        self.box_location = self.BOX_START
        self.objectives = ['GOAL_REWARD', 'IMPACT_REWARD', 'PERFORMANCE_REWARD']
        self.actions = ['up', 'right', 'down', 'left']
        self.actions_index = {'up': 0, 'right': 1, 'down': 2, 'left': 3}
        self.initial_rewards = [0, 0, 0]
        self.rewards = dict(zip(self.objectives, self.initial_rewards))
        self.terminal_state = False

    def get_state(self):
        # convert the agent's current position into a state index
        return self.agent_location + (self.NUM_CELLS * self.box_location)

    def set_state(self, agent_state, box_state):
        self.agent_location = agent_state
        self.box_location = box_state

    def env_init(self):
        # initialize the problem - starting position is always at the home location
        self.agent_location = self.AGENT_START
        self.box_location = self.BOX_START
        self.terminal_state = False

    def env_start(self):
        # Set up the environment for the start of a new episode
        self.agent_location = self.AGENT_START
        self.box_location = self.BOX_START
        self.terminal_state = False
        # return observation
        return self.agent_location, self.box_location

    def env_clean_up(self):
        # starting position is always the home location
        self.agent_location = self.AGENT_START
        self.box_location = self.BOX_START

    def potential(self, box_location):
        # Returns the value of the potential function for the current state, which is the
        # difference between the red-listed attributes of that state and the initial state.
        # In this case, its simply 0 if the box is in its original position and -1 otherwise
        return 0 if box_location == self.BOX_START else -1

    def potential_difference(self, old_state, new_state):
        # Calculate a reward based off the difference in potential between the current
        # and previous state
        return self.potential(new_state) - self.potential(old_state)

    def env_step(self, action):
        # calculate the new state of the environment
        old_box_location = self.box_location
        new_box_location = self.box_location # box won't move unless pushed
        # based on the direction of chosen action, look up the agent's new location
        new_agent_location = self.MAP[self.agent_location][self.actions_index[action]]
        # if this leads to the box's current location, look up where the box would move to
        if new_agent_location == self.box_location:
            new_box_location = self.MAP[self.box_location][self.actions_index[action]]
        # update the object locations, but only if the move is valid
        if new_agent_location >= 0 and new_box_location >= 0:
            self.agent_location = new_agent_location
            self.box_location = new_box_location
        # visualiseEnvironment() # remove if not debugging
        # is this a terminal state?
        self.terminal_state = (self.agent_location == self.AGENT_GOAL)
        # set up the reward vector
        self.rewards['IMPACT_REWARD'] = self.potential_difference(old_box_location, new_box_location)
        if not self.terminal_state:
            self.rewards['GOAL_REWARD'] = -1
            self.rewards['PERFORMANCE_REWARD'] = -1
        else:
            self.rewards['GOAL_REWARD'] = 50  # reward for reaching goal
            self.rewards['PERFORMANCE_REWARD'] = 50 + self.BOX_PENALTY[self.box_location]
        # wrap new observation
        observation = (self.agent_location, self.box_location)
        return self.rewards, observation

    def cell_char(self, cell_index):
        # Returns a character representing the content of the current cell
        if cell_index == self.agent_location:
            return 'A'
        elif cell_index == self.box_location:
            return 'B'
        else:
            return ' '

    def is_terminal(self):
        return self.terminal_state

    def visualise_environment(self):
        # print out an ASCII representation of the environment, for use in debugging
        print()
        print("******")
        print("*" + self.cell_char(0) + self.cell_char(1) + "***")
        print("*" + self.cell_char(2) + self.cell_char(3) + self.cell_char(4) + self.cell_char(5) + "*")
        print("**" + self.cell_char(6) + self.cell_char(7) + self.cell_char(8) + "*")
        print("***" + self.cell_char(9) + self.cell_char(10) + "*")
        print()

class GymSokoban(Environment,gym.Env):
    """
    This class wraps the Sokoban environment for use with Gymnasium.
    It inherits from the Sokoban environment class and implements the necessary methods
    to conform to the OpenAI Gym API.
    """

    metadata = {'render.modes': ['human']}

    action2string = {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
    }

    sting2action = {
        "up": 0,
        "right": 1,
        "down": 2,
        "left": 3,
    }

    def __init__(self, mode: str = "scalarised", we: float = 3.0, normalised_obs: bool = True):
        super(GymSokoban, self).__init__()
        super().__init__()
        self.normalised_obs = normalised_obs
        self.mode = mode
        self.we = we
        self.env_start()
        self.step_count = 0
        self.max_steps = 50

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32) if self.normalised_obs else gym.spaces.MultiDiscrete([11, 11])
        self.action_space = gym.spaces.Discrete(4)

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not isinstance(action, str):
            action = self.action2string[action]
        rewards, observation = self.env_step(action)
        self.step_count += 1
        tr = self.step_count >= self.max_steps
        tm = self.is_terminal()

        r0 = 0 if not tm else 50 + self.BOX_PENALTY[self.box_location]
        r1 = -1 if not tm else 50

        if self.mode == "scalarised":
            rewards = float(r0*self.WS[1] + r1*self.WS[0])
        else:
            rewards = np.array([r0, r1]).astype(np.float32)

        info = {'Individual': r1 ,'Etical': r0}

        return tuple(self.prep_obs(observation)), rewards, tm or tr , False , info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.visualise_environment()

    def prep_obs(self, obs):
        if self.normalised_obs:
            return np.array(obs, dtype=np.float32) / np.array([11-1, 11-1], dtype=np.float32)
        else:
            return np.array(obs, dtype=np.float32)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self.rng = default_rng(seed=seed)
        self.step_count = 0
        self.env_clean_up()
        self.env_init()
        return tuple(self.prep_obs(self.env_start())), {}

    def setWeights(self,WS):
        self.we  = WS[1]
        self.WS = WS
