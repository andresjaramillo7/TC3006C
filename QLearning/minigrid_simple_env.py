from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Floor
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper,RGBImgPartialObsWrapper
from collections import defaultdict
import numpy as np
import random

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=19,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.size = size
        self.key_positions = []
        self.lava_positions = []

        self.start_agent_pos=(1,1)
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            max_steps = 4 * size**2
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # Place walls in straight lines
        # Vertical walls
        # for y in range(1, height-1):
        #     self.put_obj(Wall(), width // 2, y)
        
        # # Horizontal walls
        # for x in range(1, width-1):
        #     self.put_obj(Wall(), x, height//2)
        
        # Create openings in the walls
        openings = [(width//2,5),(width//2,15),(5,height//2),(15,height//2),]
        
        for x, y in openings:
            self.grid.set(x, y, None)

        
        # Place a goal square in the bottom-right corner
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)
        
        self._place_agent()
        
        self.mission = "Reach the goal"

    def _place_agent(self):
        print("Executing place agent")
        self.agent_pos = (1, 1)
        self.agent_dir = 0  # 0 = right, 1 = down, 2 = left, 3 = up

    def reset(self, **kwargs):
        print("resetting")
        self.stepped_floors = set()
        obs, info = super().reset(**kwargs)
        # self._place_agent()  # Place the agent in a new random position
        return obs, info

    def step(self, action):
        prev_pos=self.agent_pos
        prev_dir=self.agent_dir
        obs, reward, terminated, truncated, info = super().step(action)

        if(self.grid.get(*self.agent_pos) is None):
            # print("this is normal floor. i.e., None")
            reward += -0.2

        if(self.agent_pos[0]>17//2 ):
            reward += -0.1

        if(self.agent_pos[1]>17//2 ):
            reward += -0.1

        if(self.agent_pos[0]>17//2 and self.agent_pos[1]>17//2):
            reward += 0.1

        if isinstance(self.grid.get(*self.agent_pos), Goal):
            reward += 10
            terminated = True
            # self.agent_pos=(1,1)
        return obs, reward, terminated, truncated, info

# Agent's directions
N_DIRECTIONS = 4

# Agent's actions
ACTION_FORWARD = 0   # move forward
ACTION_LEFT    = 1   # turn left
ACTION_RIGHT   = 2   # turn right
N_ACTIONS      = 3

# Actions mapping to MiniGrid's actions 
ACTION_MAP = {
    ACTION_LEFT:    0,  # MiniGrid: left  = 0
    ACTION_RIGHT:   1,  # MiniGrid: right = 1
    ACTION_FORWARD: 2   # MiniGrid: forward = 2
}

def get_state(env):
    x, y = env.agent_pos
    d = env.agent_dir  # 0:right, 1:down, 2:left, 3:up
    return (int(x), int(y), int(d))

def q_learning_formula(Q, a, r, x, y, d, s2, alpha, gamma, done):
    Q[x, y, d, a] = Q[x, y, d, a] + alpha * (r + (0 if done else gamma * np.max(Q[s2])) - Q[x, y, d, a])
    return Q

def epsilon_greedy(q_table, x, y, d, steps, eps_start = 1.0, eps_end = 0.05, eps_decay = 1000):
    # Calculate epsilon dynamically based on steps
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1 * steps / eps_decay)
    
    # Explore or exploit
    if random.random() >  eps_threshold:
        return int(np.argmax(q_table[x, y, d])) # Exploit: best action from Q-table
    else:
        return np.random.randint(N_ACTIONS) # Explore: random action

def q_training(env, epochs, alpha, gamma, eps_start = 1.0, eps_end = 0.05, eps_decay = 1000):
    epoch = 0
    ep_return = 0.0
    total_returns = []

    observation, info = env.reset()
    x, y, d = get_state(env)

    # Initialize Q-table with zeros
    q_table = np.zeros((env.size, env.size, N_DIRECTIONS, N_ACTIONS))

    while epoch < epochs:
        # Select action using epsilon-greedy policy
        action = epsilon_greedy(q_table, x, y, d, epoch, eps_start, eps_end, eps_decay)
        
        # Execute action
        observation2, reward, terminated, truncated, info = env.step(ACTION_MAP[action])
        done = terminated or truncated
        
        # Update Q-table
        x2, y2, d2 = get_state(env)
        s2 = (x2, y2, d2)
        q_learning_formula(q_table, action, reward, x, y, d, s2, alpha, gamma, done)
        
        # Move to next state
        x, y, d = x2, y2, d2
        ep_return += reward
        epoch += 1
        
        if done:
            total_returns.append(ep_return)
            observation, info = env.reset()
            x, y, d = get_state(env)
            ep_return = 0.0

    return q_table, total_returns

def q_testing(env, q_table, epochs):
    for epoch in range(epochs):
        observation, info = env.reset()
        x, y, d = get_state(env)
        done = False
        
        while not done:
            # Select best action from Q-table
            action = int(np.argmax(q_table[x, y, d]))
            
            # Execute action
            observation2, reward, terminated, truncated, info = env.step(ACTION_MAP[action])
            done = terminated or truncated
            
            # Move to next state
            x, y, d = get_state(env)

if __name__ == "__main__":
    env = SimpleEnv()
    env_test = SimpleEnv(render_mode='human')
    q_table, returns = q_training(env, epochs=100, alpha=0.1, gamma=0.9)
    q_testing(env_test, q_table, epochs=5)
    env.close()