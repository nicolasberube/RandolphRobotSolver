"""Randolph's robot game homemade solver.

This is the first version of an algorithm to solve
Randolph's robot game.
It is not optimised in any way, and was just to test my own
first try to solve this problem without any external help.

More info could be found in the literature, which I deliberetaly have not read
before coding this
- On the Complexity of Randolph’s Robot Game,
  Birgit Engels Tom Kamphans, 2005
- The Parameterized Complexity of Ricochet Robots,
  Adam Hesterberg Justin Kopinsky, 2017
(Could Dijkstra's algorithm have helped?)

This algorithm does not handle re-collisions.
For efficiency's sake, the algorithm can't process that
a robot can collide twice on the same robot.

The Gray robot and the BlackHole goal is not coded yet.

Bumpers are not implemented either.

All tiles data are not entered in tiles.json.
Only one group of tiles was done so far.
"""

import numpy as np
from matplotlib import pyplot as plt
import json
from copy import deepcopy

# Importing tiles data
with open('tiles.json', 'r') as f:
    tiles_data = json.load(f)


def get_idx(idx, idx_names):
    """Sanitise index of an element in a list.

    Will match the first character of the list,
    and will accept integer as input parameters if
    it was already an integer.

    Parameters
    ----------
    idx: str (or int)
        Element to match to the list.
        If integer, will return it as is.

    idx_names: list of str
        List of all elements to match to.

    Returns
    -------
    int
        Index of the element in the list.
    """
    if isinstance(idx, int) or isinstance(idx, np.int64):
        if idx > len(idx_names):
            raise Exception(f'ID "{idx}" higher than number '
                            f'in the map ({len(idx_names)}).')
        return idx
    elif isinstance(idx, str):
        idx_names_firsts = [n[0].lower() for n in idx_names]
        if idx[0].lower() in idx_names_firsts:
            return idx_names_firsts.index(idx[0].lower())
    else:
        raise Exception(f'ID {idx} not recognised.')


class Moves():
    """Class for list of moves on a board.

    Parameters
    ----------
    robots_colors: list of str
        List of all robots colors.
        This could probably be a global variable of some sort.

    Attributes
    ----------
    data: N*2 numpy array
        For each of the N moves, contains the robot index and the direction
        index of the move.

    direction_names: list of str
        List of direction. Used to translate a direction index into direction
        and vice-versa.
    """

    def __init__(self, robots_colors):
        """Init function."""
        self.data = np.zeros((0, 2), dtype=int)
        self.direction_names = ['up', 'down', 'left', 'right']
        self.robots_colors = robots_colors

    def __len__(self):
        """Length function, which is how many moves in this object."""
        return self.data.shape[0]

    def __repr__(self):
        """Representation (and string) function."""
        message = []
        for robot_id, direction in self.data:
            message.append(f'{self.robots_colors[robot_id]}_'
                           f'{self.direction_names[direction]}')
        if len(message) == 0:
            return 'NO MOVE'
        return '-'.join(message)

    def add_move(self, robot_id, direction):
        """Append a move to the move list.

        Parameters
        ----------
        robot_id: int
            Index of the robot to move. Corresponds to self.robots_colors

        direction: int
            Index of the direction to move. Corresponds to self.direction_names
        """
        robot_id = get_idx(robot_id, self.robots_colors)
        direction = get_idx(direction, self.direction_names)
        add_vector = np.array([robot_id, direction], dtype=int).reshape(1, -1)
        self.data = np.concatenate([self.data, add_vector])

    def add_moves(self, moves):
        """Append a Moves() class to the current one.

        Parameters
        ----------
        moves: Moves()
            Moves class containing the moves to append to the current class.
        """
        self.data = np.vstack([self.data, moves.data])

    def is_equal(self, moves):
        """Check if the current class is equal to another Moves() class.

        Parameters
        ----------
        moves: Moves()
            Moves class to compare the current class to.

        Returns
        -------
        bool
            If True, the two classes instanciation are equal.
        """
        if moves.data.shape != self.data.shape:
            return False
        return (moves.data == self.data).all()

    def copy(self):
        """Return deep copy of the current class object.

        Returns
        -------
        Moves()
            A copy of the current object.
        """
        return deepcopy(self)


class TMap():
    """Tracking Maps of a single robot.

    A TMap contains all reachable positions of a robot
    Each position keeps track of the shortest path(s) to it, and the
    conditional positions of other robots to reach it.

    Parameters
    ----------
    robot_id: int
        Robot index of the current TMap.
        Corresponds to the list self.robots_colors.

    initial_position: list of (int, int)
        Initial position on the board for each robots.
        The order corresponds to self.robots_colors.

    self.robots_colors: list of str
        List of color of all robots on the board.
        This could probably be a global variable.

    Attributes
    ----------
    directions: list of str
        List of all directions of movement.
        Useful for curation status when exploring the TMap's position.
        Note: This could probably be a global variable, since the Moves()
        class also uses its own directions list.

    data: 2D int numpy array
        All data relevant to a position except the move list.
        [0: 2]: coordinate (ROW, COLUMN) from (0, 0) being the top left,
                of the position
        [2]   : Length of moves required to reach the position.
        [3: 7]: Curation statuses for the position. Used during computation.
                Each of the 4 curation statuses corresponds to a
                specific direction.
        [7]   : Level of the position. A level corresponds roughly to
                the number of collisions required. Used during
                computation
        [8: X]: Conditions of the current position.
                A condition is a position where a robot needs to be,
                for a collision to reach the position.
                The condition list is a concatenation of all (X, Y)
                position of all robots on the map, corresponding
                to self.robot_colors.
                Default is (-1, -1), indicating no collision is
                necessary.

    all_moves: list of Moves()
        For each position, indicates the moves list to attain it.
        It is indexed in the same way as self.data.
    """

    def __init__(self, robot_id, initial_position, robots_colors):
        """Init function."""
        self.robot_id = robot_id
        self.robots_colors = robots_colors
        self.directions = ['U', 'D', 'L', 'R']
        add_vector = (list(initial_position) + [0]*6 +
                      [-1]*len(self.robots_colors)*2)
        self.data = np.array(add_vector, dtype=int).reshape(1, -1)
        self.all_moves = [Moves(robots_colors)]

    def __len__(self):
        """Length function, which is how many positions are in this TMap."""
        return self.data.shape[0]

    def add_position(self, position, moves, level, conditions):
        """Append a new position to the current TMap object.

        Will automatically compare it to a position currently
        in this object if that's the case, and will only keep the position
        with the shortest path. In case of multiple paths of same length,
        it will keep all of them.

        Parameters
        ----------
        position: (int, int)
            Position on the board that is attainable by robot self.robot_id

        moves: Moves()
            The movement -from the starting positions on the board-
            necessary for the robot to reach its position.

        level: int
            The level depth corresponding to the current position data.
            Each level depth corresponds roughly to a collision with another
            robot.
            It is more for tracking/debugging purposes when computing paths.

        conditions: 2N int list
            For each of the N robots (order corresponding to
            self.robots_colors),
            indicates which position another robot must be in order for this
            position to be reachable, i.e. where another robot must be
            for a collision to occur.
            Like level, this is only for the efficiency of the computing paths
            algorithms, since this information is technically in [moves].
            Default is [-1, -1] for each robot, meaning that no collision
            is needed with that particular robot.

        Returns
        -------
        1D np.array
            indexes of the Tmap that were removed during the function call
        """
        add_vector = np.array((list(position) + [len(moves)] +
                               [0]*4 + [level] + list(conditions)),
                              dtype=int)
        add_vector = add_vector.reshape(1, -1)
        pos_idx = (self.get_positions() == position).all(axis=1)
        if pos_idx.sum() == 0:
            self.data = np.concatenate([self.data, add_vector])
            self.all_moves.append(moves)
            return
        n_moves = self.get_moves_length()[pos_idx]
        if len(np.unique(n_moves)) != 1:
            raise Exception(f'Multiple moves length for single position '
                            f'{position} of robot '
                            f'{self.robots_colors[self.robot_id]}')
        n_moves = n_moves[0]
        removed_ids = []
        if len(moves) < n_moves:
            self.data = self.data[~pos_idx]
            self.all_moves = [self.all_moves[i]
                              for i in np.where(~pos_idx)[0]]
            removed_ids = np.where(pos_idx)[0]
        if len(moves) == n_moves:
            for i in np.where(pos_idx)[0]:
                if self.all_moves[i].is_equal(moves):
                    return removed_ids
        if len(moves) <= n_moves:
            self.data = np.concatenate([self.data, add_vector])
            self.all_moves.append(moves)
            return removed_ids

    def remove_position(self, tmap_id):
        """Remove a position from the TMap.

        Parameters
        ----------
        tmap_id: int
            The index of the position in the position list (self.data)
            to remove. Note that this will reindex all other position
            of higher indexes.
        """
        pos_idx = np.ones(len(self.data), dtype=bool)
        pos_idx[tmap_id] = False
        self.data = self.data[pos_idx].copy()
        self.all_moves = [self.all_moves[i] for i in np.where(pos_idx)[0]]

    def get_positions(self):
        """Return positions that are reachable by the robot.

        Returns
        -------
        N*2 2D numpy array
        """
        return self.data[:, :2]

    def get_moves_length(self):
        """Return number of moves to reach each position.

        Returns
        -------
        N-length 1D numpy array
        """
        return self.data[:, 2]

    def get_curation(self):
        """Return curation status of each position.

        A curation status can be set to 0 or 1.
        This is useful when we need to check every position in the
        list, but each check can affect the list's index and order.

        Each position has 4 curation status, corresponding to
        the 4 directions.

        Returns
        -------
        N*4 2D numpy array
        """
        return self.data[:, 3:7]

    def reset_curation(self):
        """Reset curation status to 0 for all position."""
        self.data[:, 3:7] = 0

    def set_curation(self, tmap_id, direction):
        """Set curation status to 1 for a specific position/direction combo.

        Parameters
        ----------
        tmap_id: int
            The index of the position in the position list (self.data)
            to curate.

        direction: int (or str)
            The direction to curate.
        """
        if isinstance(direction, str):
            direction = direction[0].upper()
        direction_id = get_idx(direction, self.directions)
        self.data[tmap_id, 3+direction_id] = 1

    def get_idx_curation(self):
        """Return all position's indexes that were not curated yet (status 0).

        Returns
        -------
        list of int
            List of tmap_id
        """
        return list(map(list, zip(*np.where(self.get_curation() == 0))))

    def get_levels(self):
        """Return level of each position.

        Returns
        -------
        N-length 1D numpy array
        """
        return self.data[:, 7]

    def get_max_level(self):
        """Return the maximum level of all positions in the object.

        This helps set up the next step of path searches.

        Returns
        -------
        int
        """
        return max(self.get_levels())

    def get_conditions(self):
        """Return the conditions of the position.

        Returns
        -------
        N*(2R) 2D numpy array
            Conditions for all N positions. A condition list is
            the concatenation of (X, Y) positions for all R robots,
            in case the position is conditional to a collision to another
            robot.
            Default value is -1 if no collision is needed.
        """
        return self.data[:, 8:]

    def get_condition(self, robot_id):
        """Return a single robot condition of the position.

        Parameters
        ----------
        robot_id: int
            Index of the robot on whom to check conditions.

        Returns
        -------
        N*2 2D numpy array
            Condition for all N positions for the specified other robot.
            The condition is a single (X, Y) coordinate for the
            specified robot in case the position is conditional to a collision
            to this specified robot.
            Default value is -1 if no collision is needed.
        """
        robot_id = self.get_robot_id(robot_id)
        return self.get_conditions()[2*robot_id: 2*(robot_id+1)]

    def get_robot_id(self, robot_id):
        """Sanitise robot index.

        Will sanitise the robot index to a proper integer
        if that was not the case.

        Parameters
        ----------
        robot_id: int or str

        Returns
        -------
        int
        """
        return get_idx(robot_id, self.robots_colors)

    def get_idx_intersect(self, start_position, end_position, level=None):
        """Get all intersecting indexes of the TMap's position list.

        Those are positions that intersect with a movement from start_position
        to end_position.

        Parameters
        ----------
        start_position: (int, int)
            (ROW, COLUMN) starting position of a single move to analyse.
            (0, 0) is the top left of the board.

        end_position: (int, int)
            (ROW, COLUMN) ending position of a single move to analyse.
            (0, 0) is the top left of the board.

        level: int, optional
            If specified, will only check for positions
            under or equal to a certain level.
            Default is None, which checks for all levels.

        Returns
        -------
        1D int numpy array
            List of all position in the TMap that intersects with the
            specified movement.
        """
        start_row, start_col = start_position
        end_row, end_col = end_position

        if level is None:
            level_filter = np.ones(self.get_levels().shape, dtype=bool)
        else:
            level_filter = self.get_levels() <= level

        if start_col == end_col:
            if start_row > end_row:
                # Move up
                tmap_ids = np.where(
                    (self.get_positions()[:, 1] == start_col) &
                    (self.get_positions()[:, 0] < start_row) &
                    (self.get_positions()[:, 0] >= end_row) &
                    level_filter
                )[0]
            elif start_row < end_row:
                # Move down
                tmap_ids = np.where(
                    (self.get_positions()[:, 1] == start_col) &
                    (self.get_positions()[:, 0] > start_row) &
                    (self.get_positions()[:, 0] <= end_row) &
                    level_filter
                )[0]
            else:
                raise Exception(f'Can\'t infer movement from {start_position} '
                                f'to {end_position}')
        elif start_row == end_row:
            if start_col > end_col:
                # Move left
                tmap_ids = np.where(
                    (self.get_positions()[:, 0] == start_row) &
                    (self.get_positions()[:, 1] < start_col) &
                    (self.get_positions()[:, 1] >= end_col) &
                    level_filter
                )[0]
            elif start_col < end_col:
                # Move right
                tmap_ids = np.where(
                    (self.get_positions()[:, 0] == start_row) &
                    (self.get_positions()[:, 1] > start_col) &
                    (self.get_positions()[:, 1] <= end_col) &
                    level_filter
                )[0]
        else:
            raise Exception(f'Can\'t infer movement from {start_position} '
                            f'to {end_position}')
        return tmap_ids


class Board():
    """Board class of the game.

    Parameters
    ----------
    tiles_names: list of str
        Tiles that define the board. Tile names correspond to data
        in tiles.json.
        Order of tiles is clockwise, starting from top left.

    robots_positions: N*2 2D numpy array
        Positions of all N robots on the board.
        A position is (ROW, COLUMN) where (0, 0) is the top left corner.
        Order corresponds to self.robots_colors.

    Attributes
    ----------
    robots_colors: list of str
        List of all robot colors on the board. Currently hardcoded.

    robots_tmaps: dict of {int: Tmap()}
        Dictionnary of TMaps() for all robots.
        key: robot index or the corresponding TMap()
        value: TMap() object corresponding to the robot

    height: int
        Number of rows of the board. Hardcoded to 16.

    width: int
        Number of columns of the board. Harcocded to 16.

    robot_size: int
        Size of the robot on the matplotlib display.

    vertical_walls: H*(W-1) 2D bool numpy array
        Data for vertical walls of the board.
        If True at a certain coordinate,
        indicates a veritcal wall to the right of the specified position.

    horizontal_walls: (H-1)*W 2D bool numpy array
        Data for horizontal walls of the board.
        If True at a certain coordinate,
        indicates a horizontal wall to the bottom of the specified position.

    goals: dict of {(str, str): (int, int)}
        Data for all goal tiles.
        key[0]: Color of the goal
        key[1]: Shape of the goal
        value: position of the goal on the board.

    fig: Matplotlib Figure()
        Figure data to print on screen

    ax: Matplotlib subplot
        Subplot/Graph data to print on screen
    """

    def __init__(self, tiles_names, robots_positions):
        """Init function."""
        if len(tiles_names) != 4:
            raise Exception('Please specidy 4 tiles names in clockwise manner '
                            'starting from top left.')
        self.robots_colors = ['RED', 'BLUE', 'GREEN', 'YELLOW']
        self.robots_positions = robots_positions
        self.robots_tmaps = {i_p: TMap(i_p,
                                       self.robots_positions[i_p],
                                       self.robots_colors)
                             for i_p in range(len(self.robots_colors))}

        self.height = 16
        self.width = 16
        self.robot_size = 5000/max(self.height, self.width)
        self.load_board_walls(tiles_names)
        self.load_goals(tiles_names)

    def load_board_walls(self, tiles_names):
        """Load board wall data in the class object based on tile data.

        Will load self.vertical_walls and self.horizontal_walls.

        Parameters
        ----------
        tiles_names: list of str
            Tiles that define the board. Tile names correspond to data
            in tiles.json.
            Order of tiles is clockwise, starting from top left.
        """
        self.vertical_walls = np.zeros((self.height, self.width-1),
                                       dtype=bool)
        self.horizontal_walls = np.zeros((self.height-1, self.width),
                                         dtype=bool)
        for k, tile_name in enumerate(tiles_names):
            tile_vertical_walls = np.zeros((8, 8), dtype=bool)
            tile_horizontal_walls = np.zeros((8, 8), dtype=bool)
            for (i, j) in tiles_data[tile_name]['vertical']:
                tile_vertical_walls[i, j] = True
            for (i, j) in tiles_data[tile_name]['horizontal']:
                tile_horizontal_walls[i, j] = True
            if k == 0:
                self.vertical_walls[:8, :8] = (
                    self.vertical_walls[:8, :8] | tile_vertical_walls
                )
                self.horizontal_walls[:8, :8] = (
                    self.horizontal_walls[:8, :8] | tile_horizontal_walls
                )
            elif k == 1:
                self.vertical_walls[:8, 7:] = (
                    self.vertical_walls[:8, 7:] |
                    np.rot90(tile_horizontal_walls, k=3)
                )
                self.horizontal_walls[:8, 8:] = (
                    self.horizontal_walls[:8, 8:] |
                    np.rot90(tile_vertical_walls, k=3)
                )
            elif k == 2:
                self.vertical_walls[8:, 7:] = (
                    self.vertical_walls[8:, 7:] |
                    np.rot90(tile_vertical_walls, k=2)
                )
                self.horizontal_walls[7:, 8:] = (
                    self.horizontal_walls[7:, 8:] |
                    np.rot90(tile_horizontal_walls, k=2)
                )
            elif k == 3:
                self.vertical_walls[8:, :8] = (
                    self.vertical_walls[8:, :8] |
                    np.rot90(tile_horizontal_walls, k=1)
                )
                self.horizontal_walls[7:, :8] = (
                    self.horizontal_walls[7:, :8] |
                    np.rot90(tile_vertical_walls, k=1)
                )

    def load_goals(self, tiles_names):
        """Load goal data in the class object based on tile data.

        Will load self.goals.

        Parameters
        ----------
        tiles_names: list of str
            Tiles that define the board. Tile names correspond to data
            in tiles.json.
            Order of tiles is clockwise, starting from top left.
        """
        shape_list = ["GEAR", "STAR", "MOON", "PLANET"]
        self.goals = {}
        for k, tile_name in enumerate(tiles_names):
            tile_color = tile_name[:max(i for i in range(len(tile_name))
                                        if tile_name[:i].isalpha())]
            if tile_color not in self.robots_colors:
                raise Exception(f'Tile {tile_name} incompatible with '
                                'load_goals()')
            shift_index = self.robots_colors.index(tile_color)
            goal_shapes = shape_list[-shift_index:] + shape_list[:-shift_index]
            goal_list = [(self.robots_colors[i], goal_shapes[i])
                         for i in range(len(self.robots_colors))]
            if tile_color == 'YELLOW':
                goal_list.append(('BLACK', 'HOLE'))
            for i, (row, col) in enumerate(tiles_data[tile_name]['goals']):
                goal_name = goal_list[i]
                if k == 0:
                    goal_position = [row, col]
                elif k == 1:
                    goal_position = [col, self.width-1-row]
                elif k == 2:
                    goal_position = [self.height-1-row, self.width-1-col]
                elif k == 3:
                    goal_position = [self.height-1-col, row]
                self.goals[goal_name] = goal_position

    def get_robot_id(self, robot_id):
        """Sanitise robot index.

        Will sanitise the robot index to a proper integer
        if that was not the case.

        Parameters
        ----------
        robot_id: int or str

        Returns
        -------
        int
        """
        return get_idx(robot_id, self.robots_colors)

    def print(self, goals=True):
        """Print the board on the console.

        Parameters
        ----------
        goals: bool, optional
            If True, will print goals as well.
            Default is True.
        """
        self.fig, self.ax = plt.subplots()
        self.ax.axis([0, self.width, 0, self.height])
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.print_walls()
        self.print_robots()
        if goals:
            self.print_goals()
        self.fig.set_size_inches(6, 6)
        # self.fig.show()

    def print_walls(self, ax=None):
        """Add wall data to a matplotlib suplot.

        Parameters
        ----------
        ax: Matplotlib suplot, optional
            If specified, will add the data to this subplot object.
            Default is self.ax
        """
        if ax is None:
            ax = self.ax
        for i, j in np.array(np.where(self.vertical_walls)).T:
            ax.plot([1+j, 1+j], [self.height-i, self.height-i-1], c='k')
        for i, j in np.array(np.where(self.horizontal_walls)).T:
            ax.plot([j, 1+j], [self.height-1-i, self.height-1-i], c='k')

    def print_robots(self, ax=None):
        """Add robot position's data to a matplotlib suplot.

        Parameters
        ----------
        ax: Matplotlib suplot, optional
            If specified, will add the data to this subplot object.
            Default is self.ax
        """
        if ax is None:
            ax = self.ax
        for i, position in enumerate(self.robots_positions):
            ax.scatter([position[1]+0.5],
                       [self.height - 0.5 - position[0]],
                       c=self.robots_colors[i][0:1].lower(),
                       marker='s',
                       s=self.robot_size)

    def print_goals(self, ax=None):
        """Add goals data to a matplotlib suplot.

        Parameters
        ----------
        ax: Matplotlib suplot, optional
            If specified, will add the data to this subplot object.
            Default is self.ax
        """
        if ax is None:
            ax = self.ax
        for name, position in self.goals.items():
            color, shape = name
            ax.text(position[1]+0.5,
                    self.height - 0.5 - position[0],
                    shape,
                    c=color,
                    ha='center',
                    size=80/self.width)

    def print_tmap(self, robot_id):
        """Print all TMaps position on the console.

        Parameters
        ----------
        robot_id: int
            Robot index of which to print all attainable positions.
        """
        self.fig, self.ax = plt.subplots()
        self.ax.axis([0, self.width, 0, self.height])
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.print_walls()
        self.print_tmap_positions(robot_id)
        self.fig.set_size_inches(6, 6)

    def print_tmap_positions(self, robot_id, ax=None):
        """Add TMap position data to a matplotlib suplot.

        Parameters
        ----------
        robot_id: int
            Robot index of which to print all attainable positions.

        ax: Matplotlib suplot, optional
            If specified, will add the data to this subplot object.
            Default is self.ax
        """
        robot_id = self.get_robot_id(robot_id)
        robot_tmap = self.robots_tmaps[robot_id]
        if ax is None:
            ax = self.ax
        for position in robot_tmap.get_positions():
            ax.scatter([position[1]+0.5],
                       [self.height - 0.5 - position[0]],
                       c=self.robots_colors[robot_id][0:1].lower(),
                       marker='s',
                       s=self.robot_size)

    def move_robot(self, robot_id, direction):
        """Move a robot on the board.

        Will handle all collisions.  Will update self.robots_positions

        Parameters
        ----------
        robot_id: int
            Robot index of which to print all attainable positions.

        direction: str
            Direction where to move the robot.
            Must be within {'U', 'D', 'L', 'R'}
        """
        robot_id = self.get_robot_id(robot_id)
        direction = direction[0].upper()
        if direction == 'U':
            self.move_robot_up(robot_id)
        elif direction == 'D':
            self.move_robot_down(robot_id)
        elif direction == 'L':
            self.move_robot_left(robot_id)
        elif direction == 'R':
            self.move_robot_right(robot_id)
        else:
            raise Exception(f'Direction {direction} not recognised.')

    def move_robot_up(self, robot_id):
        """Move a robot up on the board.

        Will handle all collisions. Will update self.robots_positions

        Parameters
        ----------
        robot_id: int
            Robot index of which to print all attainable positions.
        """
        start_row, start_col = self.robots_positions[robot_id]
        end_col = start_col
        walls = self.horizontal_walls[:start_row, start_col]
        walls = np.where(walls)[0]
        if len(walls) > 0:
            wall_end_row = walls.max()+1
        else:
            wall_end_row = 0
        robots = self.robots_positions[
            (self.robots_positions[:, 1] == start_col) &
            (self.robots_positions[:, 0] < start_row)
        ]
        if len(robots) > 0:
            robots_end_row = robots[:, 0].max()+1
        else:
            robots_end_row = 0
        end_row = max(wall_end_row, robots_end_row)
        self.robots_positions[robot_id] = end_row, end_col

    def move_robot_down(self, robot_id):
        """Move a robot down on the board.

        Will handle all collisions. Will update self.robots_positions

        Parameters
        ----------
        robot_id: int
            Robot index of which to print all attainable positions.
        """
        start_row, start_col = self.robots_positions[robot_id]
        end_col = start_col
        walls = self.horizontal_walls[start_row:, start_col]
        walls = np.where(walls)[0]
        if len(walls) > 0:
            wall_end_row = walls.min()+start_row
        else:
            wall_end_row = self.height-1
        robots = self.robots_positions[
            (self.robots_positions[:, 1] == start_col) &
            (self.robots_positions[:, 0] > start_row)
        ]
        if len(robots) > 0:
            robots_end_row = robots[:, 0].min()-1
        else:
            robots_end_row = self.height-1
        end_row = min(wall_end_row, robots_end_row)
        self.robots_positions[robot_id] = end_row, end_col

    def move_robot_left(self, robot_id):
        """Move a robot left on the board.

        Will handle all collisions. Will update self.robots_positions

        Parameters
        ----------
        robot_id: int
            Robot index of which to print all attainable positions.
        """
        start_row, start_col = self.robots_positions[robot_id]
        end_row = start_row
        walls = self.vertical_walls[start_row, :start_col]
        walls = np.where(walls)[0]
        if len(walls) > 0:
            wall_end_col = walls.max()+1
        else:
            wall_end_col = 0
        robots = self.robots_positions[
            (self.robots_positions[:, 0] == start_row) &
            (self.robots_positions[:, 1] < start_col)
        ]
        if len(robots) > 0:
            robots_end_col = robots[:, 1].max()+1
        else:
            robots_end_col = 0
        end_col = max(wall_end_col, robots_end_col)
        self.robots_positions[robot_id] = end_row, end_col

    def move_robot_right(self, robot_id):
        """Move a robot right on the board.

        Will handle all collisions. Will update self.robots_positions

        Parameters
        ----------
        robot_id: int
            Robot index of which to print all attainable positions.
        """
        start_row, start_col = self.robots_positions[robot_id]
        end_row = start_row
        walls = self.vertical_walls[start_row, start_col:]
        walls = np.where(walls)[0]
        if len(walls) > 0:
            wall_end_col = walls.min()+start_col
        else:
            wall_end_col = self.width-1
        robots = self.robots_positions[
            (self.robots_positions[:, 0] == start_row) &
            (self.robots_positions[:, 1] > start_col)
        ]
        if len(robots) > 0:
            robots_end_col = robots[:, 1].min()-1
        else:
            robots_end_col = self.width-1
        end_col = min(wall_end_col, robots_end_col)
        self.robots_positions[robot_id] = end_row, end_col

    def add_to_tmap(self,
                    robot_id,
                    new_position,
                    tmap_id,
                    direction,
                    level,
                    robot2_id=None,
                    tmap2_id=None):
        """Add a new position to a TMap.

        Will adequately prune the TMap and/or check if the position
        is worth adding.

        Parameters
        ----------
        robot_id: int
            Robot index corresponding to the TMap() on which to add
            a new position.

        new_position: (int, int)
            Coordinates of the position to add.

        tmap_id: int
            Index of the current TMap() corresponding to the last
            position of the robot before reaching the specified position
            to add.

        direction: int
            Direction index where to move the robot from its last position
            to reach the specified position.
            Index corresponds to direction_names attributes of the Moves()
            class. (yeah, this could be better designed)

        level: int
            Depth level of the current searching step that led to this
            new position.

        robot2_id: int, optional
            If specified, corresponds to the index of another robot
            on which a collision is necessary to reach this position.
            In other words, the new position ended on a collision
            to robot2_id.
            Default is None, which means the collision is on a wall
            and not a robot.

        tmap2_id: int, optional
            Index of robot2_id's TMap() corresponding to the
            position where it needs to be for the collision to occur,
            if a collision is necessary to reach the position.
            Default is None, which means the collision is on a wall
            and not a robot.

        Returns
        -------
        1D np.array
            indexes of the Tmap that were removed during the function call
        """
        robot_tmap = self.robots_tmaps[robot_id]
        final_moves = robot_tmap.all_moves[tmap_id].copy()
        final_conditions = robot_tmap.get_conditions()[tmap_id].copy()
        if robot2_id is not None:
            # Check if conditions are compatible
            current_conditions = final_conditions.copy()
            current_conditions[2*robot_id: 2*robot_id+2] = \
                robot_tmap.get_positions()[tmap_id]
            tmap2 = self.robots_tmaps[robot2_id]
            robot2_conditions = tmap2.get_conditions()[tmap2_id].copy()
            robot2_conditions[2*robot2_id: 2*robot2_id+2] = \
                tmap2.get_positions()[tmap2_id]

            equal_conditions_flag = True
            for i in range(len(current_conditions)):
                if ((current_conditions[i] != robot2_conditions[i]) and
                        (current_conditions[i] != -1) and
                        (robot2_conditions[i] != -1)):
                    equal_conditions_flag = False
                    break

            if not equal_conditions_flag:
                return
            final_conditions[2*robot2_id: 2*robot2_id+2] = \
                tmap2.get_positions()[tmap2_id]
            add_moves = tmap2.all_moves[tmap2_id].copy()
            add_moves.add_moves(final_moves)
            final_moves = add_moves
        final_moves.add_move(robot_id, direction)
        removed_ids = self.robots_tmaps[robot_id].add_position(
            new_position,
            final_moves,
            level,
            final_conditions
            )
        return removed_ids

    def up_collision(self, position):
        """Return noclip end position of moving up, ignoring robot collisions.

        Parameters
        ----------
        position: (int, int)
            Starting position from where to move up.

        Returns
        -------
        (int, int)
            End position of the movement, if we only consider wall collision
            and not robot collisions.
        """
        start_row, start_col = position
        end_col = start_col
        walls = self.horizontal_walls[:start_row, start_col]
        walls = np.where(walls)[0]
        if len(walls) > 0:
            wall_end_row = walls.max()+1
        else:
            wall_end_row = 0
        return [wall_end_row, end_col]

    def down_collision(self, position):
        """Return end position of moving down, ignoring robot collisions.

        Parameters
        ----------
        position: (int, int)
            Starting position from where to move down.

        Returns
        -------
        (int, int)
            End position of the movement, if we only consider wall collision
            and not robot collisions.
        """
        start_row, start_col = position
        end_col = start_col
        walls = self.horizontal_walls[start_row:, start_col]
        walls = np.where(walls)[0]
        if len(walls) > 0:
            wall_end_row = walls.min()+start_row
        else:
            wall_end_row = self.height-1
        return [wall_end_row, end_col]

    def left_collision(self, position):
        """Return end position of moving left, ignoring robot collisions.

        Parameters
        ----------
        position: (int, int)
            Starting position from where to move left.

        Returns
        -------
        (int, int)
            End position of the movement, if we only consider wall collision
            and not robot collisions.
        """
        start_row, start_col = position
        end_row = start_row
        walls = self.vertical_walls[start_row, :start_col]
        walls = np.where(walls)[0]
        if len(walls) > 0:
            wall_end_col = walls.max()+1
        else:
            wall_end_col = 0
        return [end_row, wall_end_col]

    def right_collision(self, position):
        """Return end position of moving right, ignoring robot collisions.

        Parameters
        ----------
        position: (int, int)
            Starting position from where to move right.

        Returns
        -------
        (int, int)
            End position of the movement, if we only consider wall collision
            and not robot collisions.
        """
        start_row, start_col = position
        end_row = start_row
        walls = self.vertical_walls[start_row, start_col:]
        walls = np.where(walls)[0]
        if len(walls) > 0:
            wall_end_col = walls.min()+start_col
        else:
            wall_end_col = self.width-1
        return [end_row, wall_end_col]

    def merge_tmap(self, robot_id, tmap_id, direction, level):
        """Compute all possible destinations of a robot movement.

        Will take a robot's starting position and movement,
        and will compare it to all other TMap() containing all attainable
        positions of all other robots, to figure out all possible collisions
        and therefore attainable positions of the robot.

        Will also add the resulting new positions to the current robot's
        TMap(). This function is magical (and probably therefore overscoped).

        Parameters
        ----------
        robot_id: int
            Robot index corresponding to the TMap() on which to add
            a new position.

        tmap_id: int
            Index of the current TMap() corresponding to the last
            position of the robot before reaching the specified position
            to add.

        direction: int
            Direction index where to move the robot from its last position
            to reach the specified position.
            Index corresponds to direction_names attributes of the Moves()
            class. (yeah, this could be better designed)

        level: int
            Depth level of the current searching step that led to this
            new position.
            The new position will be added as part of level+1.
        """
        robot_id = self.get_robot_id(robot_id)
        robot_tmap = self.robots_tmaps[robot_id]
        if isinstance(direction, int) or isinstance(direction, np.int64):
            direction = robot_tmap.directions[direction]
        direction = direction.upper()[0]
        if direction not in {'U', 'D', 'L', 'R'}:
            raise Exception(f'Move {direction} not recognised.')
        start_position = robot_tmap.get_positions()[tmap_id]

        if direction == 'U':
            wall_position = self.up_collision(start_position)
        elif direction == 'D':
            wall_position = self.down_collision(start_position)
        elif direction == 'L':
            wall_position = self.left_collision(start_position)
        elif direction == 'R':
            wall_position = self.right_collision(start_position)

        # Can't move in that direction
        if all(start_position[i] == wall_position[i] for i in range(2)):
            return

        starting_collision_flag = False
        for robot2_id, c in enumerate(self.robots_colors):
            if c == self.robots_colors[robot_id]:
                continue
            tmap2 = self.robots_tmaps[robot2_id]
            tmap2_ids = tmap2.get_idx_intersect(start_position,
                                                wall_position,
                                                level)
            for tmap2_id in tmap2_ids:
                # If robot collides with robot2 on its starting position
                if tmap2.get_moves_length()[tmap2_id] == 0:
                    starting_collision_flag = True
                    # Computing positions to move robot2 so it does not collide 
                    max_length = 0
                    tmap2_ids_subset = np.array([], dtype=int)
                    tmap2_ids_subset_forbidden = np.hstack([
                        tmap2.get_idx_intersect(start_position, wall_position),
                        np.where((tmap2.get_positions() ==
                                  start_position).all(axis=1))[0]
                    ])
                    while len(tmap2_ids_subset) == 0:
                        max_length += 1
                        if max_length > tmap2.get_moves_length().max():
                            break
                        tmap2_ids_subset = np.where(
                            tmap2.get_moves_length() == max_length
                        )[0]
                        tmap2_ids_subset = [
                            idx for idx in tmap2_ids_subset
                            if idx not in tmap2_ids_subset_forbidden
                            ]
                    for tmap2_id_subset in tmap2_ids_subset:
                        removed_ids = self.add_to_tmap(robot_id,
                                                       wall_position,
                                                       tmap_id,
                                                       direction,
                                                       level+1,
                                                       robot2_id,
                                                       tmap2_id_subset)
                        if removed_ids is not None:
                            if tmap_id in removed_ids:
                                raise Exception('An error has occurred')
                            tmap_id -= (removed_ids < tmap_id).sum()
                        
                if direction == 'U':
                    end_row = tmap2.get_positions()[tmap2_id, 0]+1
                    end_col = wall_position[1]
                elif direction == 'D':
                    end_row = tmap2.get_positions()[tmap2_id, 0]-1
                    end_col = wall_position[1]
                elif direction == 'L':
                    end_row = wall_position[0]
                    end_col = tmap2.get_positions()[tmap2_id, 1]+1
                elif direction == 'R':
                    end_row = wall_position[0]
                    end_col = tmap2.get_positions()[tmap2_id, 1]-1
                final_position = [end_row, end_col]
                removed_ids = self.add_to_tmap(robot_id,
                                               final_position,
                                               tmap_id,
                                               direction,
                                               level+1,
                                               robot2_id,
                                               tmap2_id)
                if removed_ids is not None:
                    if tmap_id in removed_ids:
                        raise Exception('An error has occurred')
                    tmap_id -= (removed_ids < tmap_id).sum()

        if not starting_collision_flag:
            self.add_to_tmap(robot_id,
                             wall_position,
                             tmap_id,
                             direction,
                             level+1)

    def explore_tmap(self, robot_id):
        """Generate a new depth level of a robot's TMap().

        Will explore all position of a robot's TMap(), check
        all directions from this position, and generate a new
        depth level of new positions on the TMap().

        Parameters
        ----------
        robot_id: int
            Robot index corresponding to the TMap() to explore.
        """
        robot_id = self.get_robot_id(robot_id)
        self.robots_tmaps[robot_id].reset_curation()
        level = self.robots_tmaps[robot_id].get_max_level()
        explore_ids = self.robots_tmaps[robot_id].get_idx_curation()
        while len(explore_ids) > 0:
            tmap_id, direction_id = explore_ids[0]
            self.robots_tmaps[robot_id].set_curation(tmap_id, direction_id)
            self.merge_tmap(robot_id, tmap_id, direction_id, level)
            explore_ids = self.robots_tmaps[robot_id].get_idx_curation()

    def explore_all_tmaps(self):
        """Generate a new depth level for all robots' TMap()."""
        for robot_id in range(len(self.robots_colors)):
            self.explore_tmap(robot_id)

    def check_destination(self, robot_id, position):
        """Check is a position is reachable by a robot.

        This check is based on current TMap()'s data.

        It will also perform a sanity check to make sure that the
        position is truly reachable. Some paths can be spurious,
        since for efficiency's sake, any robot not at the origin is
        not tracked, and unpredicted collision and re-collision could
        happend through the algorithm search.

        If the sanity check fails, this function will also remove the wrong
        path from the TMap().

        Parameters
        ----------
        robot_id: int
            Robot index corresponding to the TMap() to explore.

        position: (int, int)
            Destination to check

        Returns
        -------
        list of Moves()
            List of all moves that can reach this position.
            The list will be of length > 1 if multiple Moves() list can
            reach the position in the same number of moves.
        """
        robot_id = self.get_robot_id(robot_id)
        tmap = self.robots_tmaps[robot_id]
        tmap_ids = np.where((tmap.get_positions() == position).all(axis=1))[0]
        # Cleans out wrong paths
        # (sometimes a robot can collide with another one's path
        #  with this algorithm)
        bad_tmap_ids = []
        for tmap_id in tmap_ids:
            if not self.sanity_tmap(robot_id, tmap_id):
                bad_tmap_ids.append(tmap_id)
        bad_tmap_ids = sorted(bad_tmap_ids, reverse=True)
        for tmap_id in bad_tmap_ids:
            self.robots_tmaps[robot_id].remove_position(tmap_id)

        tmap = self.robots_tmaps[robot_id]
        tmap_ids = np.where((tmap.get_positions() == position).all(axis=1))[0]
        return [tmap.all_moves[tmap_id] for tmap_id in tmap_ids]

    def sanity_tmap(self, robot_id, tmap_id):
        """Perform sanity check of a specified path to a destination.

        Some paths can be spurious,
        since for efficiency's sake, any robot not at the origin is
        not tracked, and unpredicted collision and re-collision could
        happend through the algorithm search.

        Parameters
        ----------
        robot_id: int
            Robot index corresponding to the TMap() to explore.

        tmap_id: in
            Index of the TMap() corresponding to the position
            and path to sanity check.

        Returns
        -------
        bool
            True if the path is a valid one.
        """
        start_positions = self.robots_positions.copy()
        tmap = self.robots_tmaps[robot_id]
        moves = tmap.all_moves[tmap_id]
        for r_id, d in moves.data:
            direction = moves.direction_names[d]
            self.move_robot(r_id, direction)
        position = tmap.get_positions()[tmap_id]
        if (self.robots_positions[robot_id] == position).all():
            self.robots_positions = start_positions.copy()
            return True
        else:
            self.robots_positions = start_positions.copy()
            return False

    def solve(self, color, shape):
        """Solve for a destination.

        Will return all paths found for a color of the specified
        color to reach its goal specified by shape.

        Parameters
        ----------
        color: str
            Color of the robot and destination.
            Must be within {'RED', 'BLUE', 'GREEN', 'YELLOW'}

        shape: str
            Shape identifying the goal to reach
            Must be within {'GEAR', 'STAR', 'MOON', 'PLANET'}

        Returns
        -------
        list of Moves()
            List of all moves that can reach the goal destination.
            The list will be of length > 1 if multiple Moves() list can
            reach the position in the same number of moves.
        """
        robot_id = self.get_robot_id(color)
        destination = self.goals[(color, shape)]
        n_steps = len(self.robots_colors)
        n_steps = 4
        for i_step in range(n_steps):
            print(f'Step {i_step+1}/{n_steps}')
            if i_step == n_steps-1:
                self.explore_tmap(robot_id)
            else:
                self.explore_all_tmaps()
            paths = self.check_destination(robot_id, destination)
            if len(paths):
                print(f'{len(paths)} solution found with '
                      f'{len(paths[0])} moves.')
        return paths


if __name__ == '__main__':
    # Integration test
    board = Board(['RED1', 'BLUE1', 'YELLOW1', 'GREEN1'],
                  np.array([[0, 1], [4, 6], [2, 3], [0, 4]]))

    board.print()
    paths = board.solve('BLUE', 'PLANET')
