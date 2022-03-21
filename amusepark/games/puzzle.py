import numpy as np
from amusepark.games.puzzle_configs import PIECES, BOARD, CALENDAR, COLORS, MONTH
from amusepark.utils.text_attr import Background

class Piece:
    r"""
    config(2darray)  : the piece's shape
        1: occupid; 0: empty
    origin(2tuple)   : the coordinates (on board) of the piece's upper left corner
    coord (2darray)  : the coordinates of the piece's blocks (relative to the origin)
    index (int)      : the piece's index/name
    """
    def __init__(self, config: np.ndarray, index: int=0):
        self.coord = self.config2coord(config)
        self.origin = (0, 0)
        self.index = index

    def config2coord(self, config: np.ndarray):
        coord = np.vstack(np.where(config > 0)).T
        return coord
    
    def coord2config(self, coord: np.ndarray):
        h = coord[:, 0].max() + 1
        w = coord[:, 1].max() + 1

        # create new config
        config = np.zeros((h, w), dtype=int)

        # mark occupied blocks
        config[coord[:, 0], coord[:, 1]] = 1

        return config
        
    def get_dim(self):
        h = self.coord[:, 0].max() + 1
        w = self.coord[:, 1].max() + 1
        return h, w

    def rotate90(self, counter_clockwise: bool=True, in_place: bool=True):
        # define rotation matrix
        if counter_clockwise:
            rot_mat = np.array([
                [0, -1],
                [1, 0]
            ], dtype=int)
        else:
            rot_mat = np.array([
                [0, 1],
                [-1, 0]
            ], dtype=int)

        # apply rotation
        new_coord = self.coord @ rot_mat.T

        # shift
        dh = np.abs(new_coord[:, 0].min())
        dw = np.abs(new_coord[:, 1].min())
        new_coord += np.array([dh, dw])

        if in_place:
            self.coord = new_coord
        
        return new_coord

    def flip(self, horizontal: bool=True, in_place: bool=True): # horizontal: flip in horizontal direction (i.e. flip w.r.t. the vertical axis)
        h, w = self.get_dim()

        # define flip operators
        if horizontal:
            flip_vec = np.array([1, -1], dtype=int)
            bias = np.array([0, w-1], dtype=int)
        else:
            flip_vec = np.array([-1, 1], dtype=int)
            bias = np.array([h-1, 0], dtype=int)

        # apply flip operation
        new_coord = flip_vec * self.coord + bias

        if in_place:
            self.coord = new_coord
        
        return new_coord

    def get_config(self):
        return self.coord2config(self.coord)

class Board:
    r"""
    attributes:
    status/config(2darray)  : the board's status/shape
        >0: occupid; 0: empty; -1: invalid/wall
    pieces      (set[int]): the indices of pieces which are placed on the board

    methods:
    place: place 'piece' at 'origin'. if suceeded, return True; otherwise, return False.
    """
    def __init__(self, config: np.ndarray):
        # track the status of the board
        self.status = config.copy().astype(int)
        self.status[self.status>0] = 0 # clear the board

        # track the indices of pieces which are placed on the board
        self.pieces = set()

    def get_empty_cells(self) -> np.ndarray:
        empty_cells = np.vstack(np.where(self.status == 0)).T # (N, 2)
        return empty_cells

    def place(self, piece: Piece, origin: tuple) -> bool:
        # check if piece has been placed
        if piece.index in self.pieces:
            print(f"[place]: piece {piece.index} has already been placed on the board!")
            return False

        # check if piece index is > 0
        if piece.index <= 0:
            print(f"[place]: piece index should be > 0, got {piece.index}!")
            return False

        i, j = origin

        # check if valid
        if (self.status[piece.coord[:, 0]+i, piece.coord[:, 1]+j] != 0).any():
            print(f"[place]: piece {piece.index} invalid place!")
            return False

        # place piece on board
        self.status[piece.coord[:, 0]+i, piece.coord[:, 1]+j] = piece.index
        self.pieces.add(piece.index)
        piece.origin = origin

        return True   

class Calendar(Board):
    calendar = CALENDAR
    colors = COLORS
    month = MONTH
    def __init__(self):
        super().__init__(config=BOARD)
        self.set_date(date=(3, 20))

    def month2coord(self, month: int) -> tuple:
        assert month in list(range(1, 13)), f"Month {month} is invalid!"

        i = (month-1) // 6
        j = (month-1) % 6

        return (i, j)

    def day2coord(self, day: int) -> tuple:
        assert day in list(range(1, 32)), f"Day {day} is invalid!"
    
        i = (day-1) // 7 + 2
        j = (day-1) % 7

        return (i, j)

    def set_date(self, date: tuple):
        month, day = date

        # mark month cell
        mcoord = self.month2coord(month)
        self.status[mcoord] = -1

        # mark day cell
        dcoord = self.day2coord(day)
        self.status[dcoord] = -1

        # record date
        self.date = date
    
    def render(self):
        # get date coordinates
        m, d = self.date
        date_coord = {self.month2coord(m), self.day2coord(d)}

        h, w = self.status.shape

        for i in range(h):
            for j in range(w):

                # render texts
                if i < 2 and self.calendar[i, j] > 0: # month
                    s = "{:>3}".format(self.month[self.calendar[i, j]])
                elif i >= 2 and self.calendar[i, j] > 0: # day
                    s = "{:>3}".format(str(self.calendar[i, j]))
                else:
                    s = " " * 3

                # render color
                if (i, j) in date_coord or self.status[i, j] == 0: # month, day, or empty
                    print(s, end="")
                elif self.status[i, j] < 0: # wall
                    print(Background.BLACK + s, end="")
                else: # pieces
                    print(self.colors[self.status[i, j]] + s, end="")

                print(Background.RESET, end="")

            print()

class APuzzleADay:
    def __init__(self):
        self.calendar = Calendar()
        self.pieces = self.__load_pieces(PIECES)
    
    def __load_pieces(self, piece_configs: list[np.ndarray]):
        pieces = dict()

        for i, piece_config in enumerate(piece_configs):
            idx = i + 1
            pieces[idx] = Piece(piece_config, idx)

        return pieces


if __name__ == '__main__':
    apad = APuzzleADay()
    cal = apad.calendar
    ps = apad.pieces

    cal.place(ps[1], (0, 0))
    cal.place(ps[2], (0, 1))
    cal.place(ps[3], (0, 3))

    ps[4].rotate90(counter_clockwise=False)
    cal.place(ps[4], (1, 5))

    cal.place(ps[5], (3, 0))

    ps[6].flip(horizontal=False)
    cal.place(ps[6], (3, 2))

    cal.place(ps[7], (5, 0))
    cal.place(ps[8], (4, 3))

    print(cal.month[cal.date[0]], cal.date[1])
    cal.render()
