import gym
from gym import spaces

import numpy as np

from amusepark.utils.text_attr import Background
from amusepark.configs.isoland_configs import *

class MoveArrowEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['terminal']}

    def __init__(self, env_config: dict=ENV_CONFIG_0):
        super(MoveArrowEnv, self).__init__()

        self.env_config = env_config

        ### init ###
        # load env config as a state
        self.H, self.W, self.state = self.__load_config(env_config)

        # get arrows dict
        #   arrow idx -> [arrow dir, arrow pos]
        self.arrows = self._get_arrows()
        num_arrows = len(self.arrows)

        # init step counter
        self.step_counter = 0

        ### define spaces ###
        # observation/state space (see meta's of isoland_configs.py)
        #   1st layer: fixed map landmarks (i.e. env direction signs + arrow goal markers)
        #   2nd layer: current arrow positions and directions
        self.observation_space = spaces.Box(low=0, high=5+ARROW_FEATURE_NUM*num_arrows-1, shape=(self.H, self.W, 2), dtype=int)

        # action space
        #   pick which arrow to move
        self.action_space = spaces.Discrete(num_arrows)

    def __load_config(self, config: dict):
        ## map shape
        H, W = config['shape']
        # create an empty map
        state = np.zeros((H, W, 2), dtype=int)

        ## env direction signs
        for (i, j, direction) in config['signs']:
            assert direction in DIRECTIONS
            state[i, j, 0] = direction

        ## movable arrows
        for (idx, arrow) in enumerate(config['arrows']):
            goal, start = arrow

            # 1st layer
            i, j = goal
            state[i, j, 0] = self._feature2meta((idx, GOAL))

            # 2nd layer
            i, j, direction = start
            state[i, j, 1] = self._feature2meta((idx, direction))

        return H, W, state

    def reset(self):

        # load env config as a state
        self.H, self.W, self.state = self.__load_config(self.env_config)

        # get arrows dict
        #   arrow idx -> [arrow dir, arrow pos]
        self.arrows = self._get_arrows()

        # init step counter
        self.step_counter = 0
    
        return self.state
    
    def step(self, action):
        assert self.action_space.contains(action)
        arrow_dir, _ = self.arrows[action]

        # dynamics
        self._move(action, arrow_dir)

        # done  : True if all arrows are placed on their corresponding goal positions
        # reward: terminal, success or failure
        done = self._is_done()
        if done:
            reward = 1
        else:
            reward = 0

        # info
        info = {}

        # step counter
        self.step_counter += 1
    
        return self.state, reward, done, info
    
    def render(self, mode='terminal'):
        if mode == 'terminal':

            print(">>>>>> STEP %i <<<<<<"%(self.step_counter))

            for i in range(self.H):
                print(end="|")
                for j in range(self.W):

                    # get meta's
                    landmark_meta, arrow_meta = self.state[i, j, :]

                    # draw each grid cell
                    if arrow_meta > 0: 
                        # draw movable arrow
                        is_arrow, arrow_idx, arrow_dir = self._meta2feature(arrow_meta)
                        assert is_arrow and arrow_dir in DIRECTIONS, f"invalid value in layer 2: {arrow_meta}!"
                        print(COLORS[arrow_idx] + SYMBOLS[arrow_dir], end="")
                    else: 
                        # draw landmark
                        landmark_is_arrow, landmark_idx, landmark_feature = self._meta2feature(landmark_meta)
                        assert not landmark_is_arrow, f"invalid value in layer 1: {landmark_meta}!"
                        if landmark_feature in DIRECTIONS: 
                            # direction landmark
                            print(Background.LIGHT_WHITE + SYMBOLS[landmark_feature], end="")
                        elif landmark_idx == -1: 
                            # empty cell
                            print(Background.LIGHT_WHITE + " ", end="")
                        else: 
                            # arrow goal
                            print(COLORS[landmark_idx] + " ", end="")

                    print(Background.RESET, end="|")
                        
                print(Background.RESET + "\n" + "-"*self.W*2)

        else:
            raise NotImplementedError

    def close(self):
        pass
    
    def _meta2feature(self, meta: int):

        # check if the meta represents a landmark or a movable arrow
        is_arrow = True
        if meta < 5: # env direction signs
            is_arrow = False
            return is_arrow, -1, meta
        if meta % ARROW_FEATURE_NUM == 0: # arrow goal markers
            is_arrow = False
        
        # recover the arrow's index and feature
        arrow_idx = (meta - 5) // ARROW_FEATURE_NUM
        arrow_feature = (meta - 5) % ARROW_FEATURE_NUM

        return is_arrow, arrow_idx, arrow_feature

    def _feature2meta(self, feature: tuple) -> int:

        assert len(feature) == 2, f"invalid feature dimension: {len(feature)}!"

        arrow_idx, arrow_feature = feature[0], feature[1]

        assert arrow_feature < ARROW_FEATURE_NUM, f"invalid feature value: {arrow_feature}!"

        meta = 5 + ARROW_FEATURE_NUM * arrow_idx + arrow_feature

        return meta

    def _get_arrows(self) -> dict:
        # arrow idx -> [arrow dir, arrow pos]
        arrows = dict()

        # get 2nd layer
        layer2 = self.state[:, :, 1]

        # get movable arrow positions
        arrow_pos = np.where(layer2 > 0)

        # decode meta's
        for i, j in zip(arrow_pos[0], arrow_pos[1]):

            arrow_meta = layer2[i, j]
            is_arrow, arrow_idx, arrow_dir = self._meta2feature(arrow_meta)

            assert is_arrow and arrow_dir in DIRECTIONS, f"invalid value in layer 2: {arrow_meta}!"

            arrows[int(arrow_idx)] = [arrow_dir, (i, j)]
        
        return arrows

    def _update_arrow_states(self):
        # create an empty 2nd layer of state
        layer2 = np.zeros(self.state[:, :, 1].shape)

        for arrow_idx in self.arrows.keys():
            # get arrow feature and position
            arrow_dir, arrow_pos = self.arrows[arrow_idx]

            # encode arrow feature
            arrow_meta = self._feature2meta((arrow_idx, arrow_dir))

            # update 2nd layer of state
            layer2[arrow_pos] = arrow_meta

        self.state[:, :, 1] = layer2

    def _is_done(self):
        # get layers
        layer1 = self.state[:, :, 0]

        # check if done
        for arrow_idx in self.arrows.keys():
            # get arrow feature and position
            arrow_dir, arrow_pos = self.arrows[arrow_idx]

            # get landmark meta
            landmark_meta = layer1[arrow_pos]

            # get landmark feature
            landmark_is_arrow, landmark_idx, landmark_feature = self._meta2feature(landmark_meta)
            assert not landmark_is_arrow, f"invalid value in layer 1: {landmark_meta}!"

            if (landmark_feature == GOAL) and (arrow_idx == landmark_idx): # an arrow is placed on its goal
                continue
            else:
                return False  
        return True

    def _get_next_pos(self, cur_pos: tuple, move_dir: int):
        assert move_dir in DIRECTIONS, f"invalid direction: {move_dir}!"

        di, dj = DIRECTIONS[move_dir]
        next_pos = (cur_pos[0] + di, cur_pos[1] + dj)

        is_valid = False
        if 0 <= next_pos[0] < self.H and 0 <= next_pos[1] < self.W:
            is_valid = True

        return next_pos, is_valid

    def _move(self, arrow_idx: int, move_dir: int):

        # get arrow direction and current position
        assert arrow_idx in self.arrows, f"invalid arrow index: {arrow_idx}!"
        _, arrow_pos = self.arrows[arrow_idx]

        # get next position
        next_pos, is_valid = self._get_next_pos(arrow_pos, move_dir)

        if not is_valid: # out of boundary: keeps unchanged
            return

        ## check the occupancy of the next position
        # get meta's
        layer1 = self.state[:, :, 0]
        layer2 = self.state[:, :, 1]
        next_landmark_meta = layer1[next_pos]
        next_arrow_meta = layer2[next_pos]

        # get features
        _, _, next_landmark_feature = self._meta2feature(next_landmark_meta)
        next_is_arrow, next_arrow_idx, _ = self._meta2feature(next_arrow_meta)

        if next_is_arrow: 
            # another arrow is placed on the next position
            # recursively move the next arrow
            self._move(next_arrow_idx, move_dir)
        if next_landmark_feature in DIRECTIONS: 
            # the next position is a direction landmark
            # update the current arrow's direction and position
            self.arrows[arrow_idx] = [next_landmark_feature, next_pos]
        else:
            # the next position is EMPTY/GOAL
            # update the current arrow's position
            self.arrows[arrow_idx][1] = next_pos

        # update the states
        self._update_arrow_states()

if __name__ == '__main__':
    env = MoveArrowEnv(env_config=ENV_CONFIG_0)
    obs = env.reset()
    env.render()
    for act in OPT_ACTIONS_0:
        obs, r, done, info = env.step(act)
        env.render()
        print("reward = ", r)
        if done:
            print("done")
            break