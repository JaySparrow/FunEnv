from amusepark.utils.text_attr import Background

###### meta's: symbolic indicators on maps ######

GOAL = EMPTY = 0

### directions
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
DIRECTIONS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
SYMBOLS = {UP: u'\u2191', RIGHT: u'\u2192', DOWN: u'\u2193', LEFT: u'\u2190'}

### arrow's encoding: 
#   definitions start from 5
#   {arrow_i goal_pos, arrow_i cur_pos_UP, arrow_i cur_pos_RIGHT, arrow_i cur_pos_DOWN, arrow_i cur_pos_LEFT} 
# = {5 + 5i          , 6 + 5i            , 7 + 5i               , 8 + 5i              , 9 + 5i} 
#   for i = 0, 1, 2, ...
ARROW_FEATURE_NUM = 5

### arrow's color
COLORS = [Background.LIGHT_RED, Background.BLUE, Background.GREEN]

###### env configs ######
## shape : (map H, map W)
## signs : env direction signs (i, j, direction)
## arrows: movable arrows (GOAL, START state) = ((GOAL i, GOAL j), (START i, START j, START direction))
ENV_CONFIG_0 = {
    'shape': (7, 8),
    'signs': [(2, 4, DOWN), (3, 2, RIGHT), (4, 1, RIGHT), (5, 2, UP)],
    'arrows': [
        ((3, 4), (4, 4, LEFT)), 
        ((4, 3), (5, 3, LEFT)),
        ((4, 5), (5, 5, LEFT))
    ]
}

OPT_ACTIONS_0 = [0, 0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 2, 1, 1, 1]

ENV_CONFIG_1 = {
    'shape': (7, 8),
    'signs': [(1, 3, RIGHT), (1, 5, DOWN), (2, 4, DOWN), (4, 4, UP), (4, 5, LEFT)],
    'arrows': [
        ((4, 3), (1, 6, LEFT)), 
        ((3, 4), (2, 3, DOWN)),
        ((5, 4), (4, 2, RIGHT))
    ]
}

OPT_ACTIONS_1 = [0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1]