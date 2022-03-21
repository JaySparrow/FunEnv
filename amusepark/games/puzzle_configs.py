import numpy as np
from amusepark.utils.text_attr import Background

### pieces ###
PIECES = [
    np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1]
    ]),
    np.array([
        [1, 0, 1],
        [1, 1, 1]
    ]),
    np.array([
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0]
    ]),
    np.array([
        [0, 1, 1, 1],
        [1, 1, 0, 0]
    ]),
    np.array([
        [1, 1, 0],
        [1, 1, 1]
    ]),
    np.array([
        [0, 0, 1, 0],
        [1, 1, 1, 1]
    ]),
    np.array([
        [1, 1, 1],
        [1, 1, 1]
    ]),
    np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 1]
    ])
]

### board ###
BOARD = np.array([
    [0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, -1, -1, -1]
])

CALENDAR = np.array([
    [1, 2, 3, 4, 5, 6, -1],
    [7, 8, 9, 10, 11, 12, -1],
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19, 20, 21],
    [22, 23, 24, 25, 26, 27, 28],
    [29, 30, 31, -1, -1, -1, -1]
], dtype=int)

MONTH = [None, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

### colors ###
COLORS = [Background.RESET, Background.LIGHT_BLUE, Background.BLUE, Background.BROWN, Background.CYAN, Background.GREEN, Background.LIGHT_PURPLE, Background.YELLOW, Background.RED]