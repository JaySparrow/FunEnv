import numpy as np

## directions ##
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
DIRECTIONS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

## maze dimensions ##
H = 5
W = 5

## mazes ##
MAZES = [
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
    ], dtype=int),
    np.array([
        [0, 0, 0,  0,  0],
        [0, 0, 0, -1,  0],
        [0, 0, 0,  0,  0],
        [0, 0, 0,  0,  0],
        [0, 0, 0,  2, -1],
    ], dtype=int),
    np.array([
        [0, 0,  0, 0,  0],
        [0, 0,  0, 0,  0],
        [0, 0, -1, 0, -1],
        [0, 0,  0, 0,  2],
        [0, 0,  0, 0,  0],
    ], dtype=int),
    np.array([
        [0,  0,  0,  0,  0],
        [0, -1,  2, -1,  0],
        [0,  0,  0,  0,  0],
        [0, -1, -1,  0, -1],
        [0,  0,  0,  0, -1],
    ], dtype=int),
    np.array([
        [ 0, 0,  0,  0, -1],
        [ 0, 2,  0,  0,  0],
        [ 0, 0,  0, -1,  0],
        [-1, 0,  0,  0,  0],
        [ 0, 0, -1,  0,  0],
    ], dtype=int),
    np.array([
        [ 0,  0, -1, -1, -1],
        [ 0,  0,  0,  0,  0],
        [ 0,  2,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [-1, -1,  0,  0,  0]
    ], dtype=int)
]

## optimal actions ##
OPT_ACTIONS = [
    [UP, RIGHT, DOWN, LEFT, UP, RIGHT, DOWN, LEFT, UP],
    [LEFT, UP, RIGHT, DOWN, LEFT, UP, RIGHT, DOWN, RIGHT],
    [DOWN, LEFT, UP, RIGHT, DOWN, LEFT, DOWN, RIGHT, UP],
    [DOWN, LEFT, DOWN, RIGHT, UP, RIGHT, UP, LEFT, DOWN],
    [UP, LEFT, DOWN, RIGHT, UP, RIGHT, DOWN, RIGHT, DOWN, LEFT, UP, LEFT, DOWN, LEFT],
    [DOWN, LEFT, UP, RIGHT, DOWN, RIGHT, DOWN, LEFT, UP, RIGHT, DOWN]
]