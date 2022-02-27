import gym
from gym import spaces

import os
import numpy as np

from amusepark.utils.text_attr import Background

GUESS_NUM = 6

def load_words(filename: str) -> list:

    if not os.path.isfile(filename):
        print("Word file unfound! Use default!")
        return ['default']

    with open(filename, 'r') as f:
        lines = f.readlines()
    words = [l.strip('\n') for l in lines]
    return words

def compare_words(word1: str, word2: str) -> np.ndarray:
    assert len(word1) == len(word2), "length not equal"
    word_len = len(word1)

    word1 = list(word1.lower())
    word2 = list(word2.lower())
    # init color
    color = -np.ones(word_len, dtype=int)

    for i in range(word_len):

        # green
        if word1[i] == word2[i]:
            color[i] = 2
            word1[i] = '-'
            word2[i] = '-'
    
    for i in range(word_len):
        char1 = word1[i]
        if char1 == '-': 
            continue

        # orange
        if char1 in word2:
            color[i] = 1
            word2[word2.index(char1)] = '-'
        # gray
        else:
            color[i] = 0
        
    return color


class WordleEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, word_filename: str, guess_num: int=6):
        super(WordleEnv, self).__init__()

        self.guess_num = guess_num

        # load words
        self.__word_pool = load_words(word_filename)

        # init hidden target word
        self.hidden_word = np.random.choice(self.__word_pool)
        self.word_len = len(self.hidden_word)
        
        # action space \in {A, B, ..., Z}^self.word_len
        self.action_space = spaces.MultiDiscrete([26] * self.word_len) # ndarray of size (self.word_len, )

        # observation/state space 
        #   color: {not guessed: -1, gray: 0, orange: 1, green: 2}
        #   guess: {not guessed: -1, A: 0, B: 1, ..., Z: 25}
        self.observation_space = spaces.Dict({
            'color': spaces.Box(low=-1, high=2, shape=(self.guess_num, self.word_len), dtype=int),
            'guess': spaces.Box(low=-1, high=25, shape=(self.guess_num, self.word_len), dtype=int)
        })

        # init state
        self.color = -np.ones((self.guess_num, self.word_len), dtype=int)
        self.guess = -np.ones((self.guess_num, self.word_len), dtype=int)

        # init guess counter
        self.guess_counter = 0

    def _array2str(self, action: np.ndarray) -> str:
        return ''.join([chr(v + 97) for v in action])
    
    def _str2array(self, word: str) -> np.ndarray:
        return np.array([ord(c) - 97 for c in word], dtype=int)

    def step(self, action):
        assert self.action_space.contains(action)

        action_str = self._array2str(action)

        # done
        done = False
        if (self.guess_counter >= self.guess_num-1) or (action_str == self.hidden_word):
            done = True

        # state
        self.color[self.guess_counter, :] = compare_words(action_str, self.hidden_word)
        self.guess[self.guess_counter, :] = action
        observation = {'color': self.color, 'guess': self.guess}

        # reward
        if done:
            if action_str == self.hidden_word:
                reward = self.guess_num - self.guess_counter
            else:
                reward = -1
        else:
            reward = 0

        # info
        info = {'hidden_word': self.hidden_word}

        # guess counter
        self.guess_counter += 1

        return observation, reward, done, info
    
    def reset(self):

        # init state
        self.color = -np.ones((self.guess_num, self.word_len), dtype=int)
        self.guess = -np.ones((self.guess_num, self.word_len), dtype=int)

        obs = {'color': self.color, 'guess': self.guess}

        # init hidden target word
        self.hidden_word = np.random.choice(self.__word_pool)
        self.word_len = len(self.hidden_word)

        # init guess counter
        self.guess_counter = 0

        return obs
    
    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError
        
        print(">>>>>> GUESS %i <<<<<<"%(self.guess_counter))
        for i in range(self.guess_num):
            word = self._array2str(self.guess[i, ])
            color = self.color[i, :]
            for (w, c) in zip(word, color):
                if c == 2: prefix = Background.GREEN
                elif c == 1: prefix = Background.BROWN
                elif c == 0: prefix = Background.LIGHT_GRAY
                else: prefix = Background.BLACK
                print(prefix + w, end='')
            print(Background.RESET)
        
    def close (self):
        pass

if __name__ == '__main__':
    from amusepark.utils.path import data_path
    env = WordleEnv(os.path.join(data_path, 'wordle-hidden.txt'), guess_num=GUESS_NUM)
    obs = env.reset()
    env.render()
    for i in range(env.guess_num):
        act = env.action_space.sample()
        obs, r, done, info = env.step(act)
        env.render()
        print(r, done, info)
