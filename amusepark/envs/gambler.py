from gym import spaces

import numpy as np

class CoinGameEnv:
    """ 2-Armed Bandit Env
    The banker and the player each has a coin. In one play, they need to show either face of their own coin simultaneously, and the result is dependent on the face combination.
    From the player's perspective:
    +--------------------------+
    banker\player | head | tail
    +--------------------------+
    head          | -3   | +2
    +--------------------------+
    tail          | +2   | -1
    +--------------------------+
    The banker shows the "head" according to a (deterministic or nondeterministic) probability p.
    The player wants to maximize the expected value earned.
    """
    metadata = {'render.modes': ['terminal']}
    # (banker, player): value
    values = {
        (0, 0): -3,
        (0, 1): +2,
        (1, 0): +2,
        (1, 1): -1,
        (-1, -1): 0
    }
    # action: label
    labels = {
        0: 'head',
        1: 'tail',
        -1: 'none'
    }

    def __init__(self, deterministic=True):
        super(CoinGameEnv, self).__init__()

        # variables
        self.deterministic = deterministic
        self.deterministic_p = np.random.rand()
        self.banker = -1
        self.player = -1

        # observation/state space: placeholder
        self.observation_space = spaces.Discrete(1)

        # action space: head (0) or tail (1)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        return 0

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}!"

        self.player = action

        ## banker's strategy
        # sample the probability p for "head"
        if self.deterministic:
            p = self.deterministic_p
        else:
            p = np.random.rand()
        # sample the face according to p above
        if np.random.rand() < p:
            self.banker = 0
        else:
            self.banker = 1

        ## reward
        reward = self.values[(self.banker, action)]

        ## info
        info = {
            'banker_p': p,
            'banker_a': self.banker
        }

        return 0, reward, True, info

    def render(self, mode='terminal'):
        if mode != 'terminal':
            raise NotImplementedError

        print(">>")
        print("{:6s}|{:6s}|{:6s}".format("banker", "player", "value"))
        print("-"*20)
        print("{:6s}|{:6s}|{:<6d}".format(self.labels[self.banker], self.labels[self.player], self.values[(self.banker, self.player)]))
        print()

    def close(self):
        pass

if __name__ == '__main__':
    for mode in {True, False}:
        print(f"\n>>>>>> Deterministic Mode: {mode} <<<<<<")
        env = CoinGameEnv(deterministic=mode)

        for _ in range(5):
            obs = env.reset()
            done = False
            while not done:
                act = env.action_space.sample()
                obs, r, done, info = env.step(act)
                env.render()
                print(r, done, info)