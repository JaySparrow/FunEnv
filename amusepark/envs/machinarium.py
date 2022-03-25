import gym
from gym import spaces

import numpy as np

from amusepark.configs.machinarium_configs import *
from amusepark.utils.text_attr import Background

class TraverseMazeEnv(gym.Env):
    """A mini-puzzle in the greenhouse of the game Machinarium"""
    metadata = {'render.modes': ['terminal']}

    def __init__(self, maze_idx: int=-1):
        super(TraverseMazeEnv, self).__init__()

        self.maze_idx = maze_idx

        # observation/state space
        #   {-1: obstacle; 0: empty; 1: traversed; 2: start}
        self.observation_space = spaces.Box(low=-1, high=2, shape=(H, W), dtype=int)

        # action space
        #   {UP: 0; RIGHT: 1; DOWN: 2; LEFT: 3}
        self.action_space = spaces.Discrete(4)

        # init the maze & current position
        self.maze = np.zeros((H, W), dtype=int)
        self.cur_pos = (0, 0)
        self.maze[self.cur_pos] = 2

        # init step counter
        self.step_counter = 0

    def reset(self):
        # init the maze & current position
        self.maze, self.cur_pos = self.__load_maze()

        # init step counter
        self.step_counter = 0

        return self.maze

    def step(self, action):
        assert self.action_space.contains(action)

        # move in the direction until blocked
        is_valid = True
        while is_valid:
            is_valid = self.__move2next(action)

        # get valid moving directions from the new position
        valid_directions = self._get_valid_directions()

        # done if no further valid moves 
        if len(valid_directions) == 0:
            if not 0 in self.maze: # success: every cell of the maze has been traversed
                reward = 1
            else: # failure: some cell hasn't been traversed
                reward = -1
            done = True
        else:
            reward = 0
            done = False

        # info
        info = {}

        # step counter
        self.step_counter += 1

        return self.maze, reward, done, info

    def render(self, mode='terminal'):
        if mode != 'terminal':
            raise NotImplementedError

        print(">>>>>> STEP %i <<<<<<"%(self.step_counter))

        for i in range(H):
            for j in range(W):
                cell = self.maze[i, j]
                if (i, j) == self.cur_pos: 
                    # current position
                    print(Background.BROWN + " ", end="")
                elif cell == -1: 
                    # obstacle
                    print(Background.RED + " ", end="")
                elif cell == 0: 
                    # empty
                    print(Background.LIGHT_GRAY + " ", end="")
                elif cell == 1: 
                    # traversed
                    print(Background.GREEN + " ", end="")
                elif cell == 2: 
                    # start
                    print(Background.BLUE + " ", end="")
                else: 
                    # unknown
                    print(" ", end="")
                print(Background.RESET, end="|")
            print(Background.RESET + "\n" + "-"*W*2)

    def close(self):
        pass

    def _get_valid_directions(self):
        valid_directions = set()

        for move_dir in DIRECTIONS:
            di, dj = DIRECTIONS[move_dir]
            next_pos = (self.cur_pos[0] + di, self.cur_pos[1] + dj)

            if 0 <= next_pos[0] < H and 0 <= next_pos[1] < W and self.maze[next_pos] == 0:
                valid_directions.add(move_dir)

        return valid_directions

    def __load_maze(self):
        num = len(MAZES)

        # get the maze configuration
        if not (0 <= self.maze_idx < num): # randomly sample a maze
            maze_idx = np.random.randint(num)
        else: # always pick the selected one
            maze_idx = self.maze_idx
        maze = MAZES[maze_idx]

        # get the start
        assert 2 in maze, f"maze {maze_idx} has no starting position!"
        pos = np.where(maze == 2)
        start = (pos[0][0], pos[1][0])

        return maze, start

    def __move2next(self, move_dir: int) -> bool:
        assert move_dir in DIRECTIONS, f"invalid direction: {move_dir}!"

        # next position
        di, dj = DIRECTIONS[move_dir]
        next_pos = (self.cur_pos[0] + di, self.cur_pos[1] + dj)

        # check if valid
        is_valid = False
        if 0 <= next_pos[0] < H and 0 <= next_pos[1] < W and self.maze[next_pos] == 0:
            is_valid = True

        # update current position & the maze
        if is_valid:
            self.cur_pos = next_pos
            self.maze[next_pos] = 1

        return is_valid

if __name__ == '__main__':
    maze_idx = 5
    env = TraverseMazeEnv(maze_idx)
    obs = env.reset()
    env.render()
    for act in OPT_ACTIONS[maze_idx]:
        obs, r, done, info = env.step(act)
        env.render()
        print("reward = ", r)
        if done:
            print("done")
            break