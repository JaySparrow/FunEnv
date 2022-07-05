import gym
from gym import spaces

import numpy as np

from amusepark.utils.text_attr import Foreground

class TicTacToeEnv(gym.Env):
    """ Tic-Tac-Toe

    - Observation:
    A 3x3 grid board. 1: pieces of Player 1; -1 pieces of Player 2.

    - Action:
    The position is encoded from left to right and top to bottom as 0 to 8. Actions take in turn.
    
    """
    metadata = {'render.modes': ['terminal']}
    def __init__(self) -> None:
        super(TicTacToeEnv, self).__init__()

        # variables
        self.board = np.zeros((3, 3), dtype=int)
        self.turn_piece = 1 # 1: player 1; -1: player 2
        self.step_counter = 0

        # observation/state space
        # 0: empty, 1: player 1, -1: player 2
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=int)

        # action space
        # left to right, top to bottom in the 3x3 grid
        self.action_space = spaces.Discrete(9)

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.turn_piece = 1 # 1: player 1; -1: player 2
        self.step_counter = 0
        return self.board

    def is_win(self, i, j, turn_piece, board):
        # horizontal
        if board[(i+1)%3, j] == turn_piece and board[(i+2)%3, j] == turn_piece:
            return True
        elif board[i, (j+1)%3] == turn_piece and board[i, (j+2)%3] == turn_piece:
            return True
        elif (i+j)%2==0 and board[(i+1)%3, (j+1)%3] == turn_piece and board[(i+2)%3, (j+2)%3] == turn_piece:
            return True
        else:
            return False

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}!"

        # decode coordinates
        i, j = action // 3, action % 3

        # check if the cell has been occupied
        if self.board[i, j] == 0: # empty
            self.board[i, j] = self.turn_piece

            # check if game is over
            if self.is_win(i, j, self.turn_piece, self.board):
                done = True
                reward = float(self.turn_piece)
                msg = "Win: 3 pieces in a row!"
            elif (self.board != 0).all():
                done = True
                reward = -1.
                msg = "Tie: no feasible move! (Player 2 wins by default.)"
            else:
                done = False
                reward = 0.
                msg = ""
            
        else: # occupied
            done = True
            reward = float(self.turn_piece*-1)
            msg = f"Postion ({i}, {j}) is occupied! "

        # info
        info = {"message": msg}

        # change turn
        self.turn_piece *= -1

        # increment step counter
        self.step_counter += 1

        return self.board, reward, done, info
    
    def render(self, mode='terminal'):
        if mode != 'terminal':
            raise NotImplementedError

        if (self.board == 0).all(): print(">>>Init")
        elif self.turn_piece == -1: print(">>>" + Foreground.RED + "Player 1" + Foreground.RESET + f" Step {self.step_counter}")
        elif self.turn_piece == 1: print(">>>" + Foreground.BLUE + "Player 2" + Foreground.RESET + f" Step {self.step_counter}")
        else: raise NotImplementedError

        print("-"*7)
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 1:
                    print("|" + Foreground.RED + "o" + Foreground.RESET, end="")
                elif self.board[i, j] == -1:
                    print("|" + Foreground.BLUE + "x" + Foreground.RESET, end="")
                else:
                    print("| ", end="")
            print("|\n" + "-"*7)

    def close(self):
        pass


class GobbletEnv(gym.Env):
    """ Simplified Version of the 2-Player Board Game Gobblet
    Simplification: All pieces are available at the beginning (no external stacks).

    - Mode:
        - Static(0) : Once a piece is placed, it can no longer be moved again.
        - Dynamic(1): All uncoverd pieces can be moved as wish.

    - Observation:
    A 4x4 grid board. Each entry number represents a piece on board (1 to 12 for Player 1; -1 to -12 for Player 2) or empty (0).

    - Action:
    A 2-tuple of (piece[int], position[int]). The piece is an absolute number in 1 to 12. The position is encoded from left to right and top to bottom as 0 to 15. Actions take in turn.

    - Note: An invalid action terminates the game and the other player automatically wins.
    """
    metadata = {'render.modes': ['terminal']}
    mode_dict = {
        0: 'Static',
        1: 'Dynamic'
    }

    # ranks of each piece (a higher can cover a lower)
    p_ranks = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4] # 1st: dummy
    p_symbols = ['\u25cb', '\u25d4', '\u25d1', '\u25d5', '\u25cf']

    def __init__(self, mode: int=0) -> None:
        super(GobbletEnv, self).__init__()

        ## variables
        # mode
        assert mode in self.mode_dict, f"Invalid mode: {mode}!"
        self.mode = mode
        # availability of each piece for both players (1st: dummy)
        self.p1_avail = [False] + [True for _ in range(12)]
        self.p2_avail = [False] + [True for _ in range(12)]
        # board of the history (pieces in a channel with the smaller index cover that with the larger index)
        # pieces are labeled 1 to 12 for player 1 and -1 to -12 for player 2; 0 stands for empty
        self.board = np.zeros((5, 4, 4), dtype=int) # last channel: dummy
        # ranks of the pieces of history
        # 1 to 4 for player 1 and -1 to -4 for player 2; 0 stands for empty
        self.rank = np.zeros((5, 4, 4), dtype=int) # last channel: dummy
        # who's turn: 1 for player 1; -1 for player 2
        self.player = 1
        # step counter
        self.step_counter = 0
        
        ## observation/state space
        self.observation_space = spaces.Box(low=-12, high=12, shape=(4, 4), dtype=int)

        ## action space: (piece, position)
        self.action_space = spaces.Tuple(
            (spaces.Discrete(12, start=1), spaces.Discrete(16))
        )
    
    def reset(self):
        ## init variables
        # availability of each piece for both players (1st: dummy)
        self.p1_avail = [False] + [True for _ in range(12)]
        self.p2_avail = [False] + [True for _ in range(12)]
        # board of the history (pieces in a channel with the smaller index cover that with the larger index)
        # pieces are labeled 1 to 12 for player 1 and -1 to -12 for player 2; 0 stands for empty
        self.board = np.zeros((5, 4, 4), dtype=int) # last channel: dummy
        # ranks of the pieces of history
        # 1 to 4 for player 1 and -1 to -4 for player 2; 0 stands for empty
        self.rank = np.zeros((5, 4, 4), dtype=int) # last channel: dummy
        # who's turn: 1 for player 1; -1 for player 2
        self.player = 1
        # step counter
        self.step_counter = 0

        return self.board[0, :, :]

    def step(self, action: tuple):
        assert self.action_space.contains(action)

        # decode action
        piece, pos = action
        pos_i, pos_j = pos // 4, pos % 4

        # decide player
        if self.player == 1: p_avail = self.p1_avail
        elif self.player == -1: p_avail = self.p2_avail
        else: raise NotImplementedError

        # piece not available: the other player wins
        if not p_avail[piece]:
            self.player *= -1
            return self.board[0, :, :], float(self.player), True, {'board': self.board, 'rank': self.rank, 'message': f"Piece {piece} unavailable!"}

        # pos is taken up by a higher rank piece: the other player wins
        if self.p_ranks[piece] <= self.p_ranks[abs(self.board[0, pos_i, pos_j])]:
            self.player *= -1
            return self.board[0, :, :], float(self.player), True, {'board': self.board, 'rank': self.rank, 'message': f"Position ({pos_i}, {pos_j}) unavailable for piece {piece}!"}

        # remove the piece from the board if any (dynamic mode)
        revealed_piece = 0
        if (self.player * piece == self.board[0, :, :]).any():
            prev_pos = np.where(self.player * piece == self.board[0, :, :])
            prev_pos_i, prev_pos_j = int(prev_pos[0]), int(prev_pos[1])
            for c in range(3, -1, -1):
                self.board[c, prev_pos_i, prev_pos_j] = self.board[c+1, prev_pos_i, prev_pos_j]
                self.rank[c, prev_pos_i, prev_pos_j] = self.rank[c+1, prev_pos_i, prev_pos_j]
            revealed_piece = self.board[0, prev_pos_i, prev_pos_j]
        
        # place the piece
        covered_piece = self.board[0, pos_i, pos_j]
        for c in range(1, 5, 1):
            self.board[c, pos_i, pos_j] = self.board[c-1, pos_i, pos_j]
            self.rank[c, pos_i, pos_j] = self.rank[c-1, pos_i, pos_j]
        self.board[0, pos_i, pos_j] = self.player * piece
        self.rank[0, pos_i, pos_j] = self.player * self.p_ranks[piece]

        # update availability
        if covered_piece > 0: self.p1_avail[covered_piece] = False
        elif covered_piece < 0: self.p2_avail[-covered_piece] = False
        if self.mode == 0: # static
            p_avail[piece] = False
        elif self.mode == 1: # dynamic
            if revealed_piece > 0: self.p1_avail[revealed_piece] = True
            elif revealed_piece < 0: self.p2_avail[-revealed_piece] = True
        else:
            raise NotImplementedError

        msg = ""
        # check if the player wins or there is no valid move left (the other wins)
        if (self.player * self.board[0, pos_i, :] > 0).all() or \
            (self.player * self.board[0, :, pos_j] > 0).all() or \
                (self.player * self.board[0, (0,1,2,3), (0,1,2,3)] > 0).all() or \
                    (self.player * self.board[0, (0,1,2,3), (3,2,1,0)] > 0).all(): # horizontal, vertical, or diagonals
            done, reward = True, float(self.player)
            msg = "Win: 4 pieces in a row!"
        elif sum(p_avail) == 0: # no piece available
            done, reward = True, float(-self.player)
            msg = "No piece available!"
        elif self.mode == 0: # all pieces on board ranked higher than available pieces
            p_avail_ = p_avail.copy()
            p_avail_.reverse()
            avail_max_rank = self.p_ranks[12-p_avail_.index(True)]
            on_board_min_rank = abs(self.rank[0, :, :]).min()
            if avail_max_rank <= on_board_min_rank:
                done, reward = True, float(-self.player)
                msg = f"No position available! (max rank of available piece: {avail_max_rank}, min rank on board: {on_board_min_rank})"
            else:
                done, reward = False, 0
        else:
            done, reward = False, 0

        # switch turn
        self.player *= -1

        # increment step counter
        self.step_counter += 1

        return self.board[0, :, :], reward, done, {'board': self.board, 'rank': self.rank, 'message': msg}

    def render(self, mode='terminal'):
        if mode != 'terminal':
            raise NotImplementedError

        if (self.rank == 0).all(): print(">>>Init")
        elif self.player == 1: print(">>>" + Foreground.BLUE + "Player 2" + Foreground.RESET + f" Step {self.step_counter}")
        elif self.player == -1: print(">>>" + Foreground.RED + "Player 1" + Foreground.RESET + f" Step {self.step_counter}")
        else: raise NotImplementedError

        _, m, n = self.board.shape
        for i in range(m):
            s1 = ""
            s2 = ""
            for j in range(n):
                r = self.rank[0, i, j] # rank
                p = self.board[0, i, j] # piece
                # rank
                if r > 0: s1 += "|" + Foreground.RED + self.p_symbols[r] + Foreground.RESET
                elif r < 0: s1 += "|" + Foreground.BLUE + self.p_symbols[-r] + Foreground.RESET
                else: s1 += "| "
                # piece
                if p > 0: s2 += "|" + Foreground.RED + "%2d"%(p) + Foreground.RESET
                elif p < 0: s2 += "|" + Foreground.BLUE + "%2d"%(-p) + Foreground.RESET
                else: s2 += "|  "
            s1 += "|"
            s2 += "|"
            print(s1 + " "*5 + s2)
            print("-"*(2*n+1) + " "*5 + "-"*(3*n+1))
        
    def close(self):
        pass

def TicTacToe_example():
    print("\n====== Tic-Tac-Toe ======\n")
    # env
    env = TicTacToeEnv()
    # action sequence
    actions = [6, 2, 3, 0, 1, 5, 8, 7, 4]

    obs = env.reset()
    env.render()
    done = False
    while not done:
        act = actions[env.step_counter]
        obs, r, done, info = env.step(act)
        env.render()
        print(r, done)
    print("---\n" + info["message"])

def Gobblet_example(mode=0):
    # env
    env = GobbletEnv(mode)
    print("====== Gobblet: " + env.mode_dict[mode] + " Mode ======\n")
    # action sequence
    actions = [
        [
            (12, 15), (12, 12), 
            (11, 5), (11, 9), 
            (9, 4), (10, 0), 
            (8, 7), (9, 6), 
            (10, 6)
        ],
        [
            (12, 9), (12, 0),
            (11, 4), (11, 15),
            (10, 13), (10, 3),
            (11, 1), (9, 5),
            (12, 5), (8, 9), 
            (10, 9), (7, 13),
            (11, 13), (10, 1),
            (12, 6), (11, 3),
            (11, 2), (6, 10),
            (9, 10), (12, 14),
            (8, 0), (5, 15),
            (11, 12), (3, 2),
            (6, 2), (4, 11),
            (12, 11), (11, 10),
            (7, 2), (10, 0)
        ]
    ]

    obs = env.reset()
    env.render()
    done = False
    while not done:
        act = actions[mode][env.step_counter]
        obs, r, done, info = env.step(act)
        env.render()
        print(r, done)
    print("---\n" + info["message"])

if __name__ == '__main__':
        Gobblet_example(mode=1)
        # TicTacToe_example()