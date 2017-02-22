"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def base_heuristic(game, player):
    return game.utility(player)


def max_player_path(game, player, max_path=0):
    moves = game.get_legal_moves()
    if len(moves) == 0:
        return max_path
    for move in moves:
        new_game = game.forecast_move(move)
        mp = max_player_path(new_game, player, max_path + 1)
        max_path = max(max_path, mp)
    return max_path


def heuristic5(game, player):
    """

    """
    return max_player_path(game, player) - max_player_path(game, game.get_opponent(player))


def heuristic4(game, player):
    """
    The number of open spaces around the location of the player
    """
    location_player = game.get_player_location(player)
    blank_spaces = game.get_blank_spaces()
    open_space_player = 0.0
    for (m, n) in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
        if (location_player[0] + m, location_player[1] + n) in blank_spaces:
            open_space_player += 1
    return open_space_player


def heuristic3(game, player):
    """
    The number of open spaces around the location of the player's future move
    """
    best_num_open_spaces = 0.0
    for legal_move in game.get_legal_moves():
        new_game = game.forecast_move(legal_move)
        location_player = new_game.get_player_location(player)
        blank_spaces = new_game.get_blank_spaces()
        open_space_player = 0.0
        for (m, n) in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            if (location_player[0] + m, location_player[1] + n) in blank_spaces:
                open_space_player += 1
        best_num_open_spaces = max(best_num_open_spaces, open_space_player)
    return best_num_open_spaces


def heuristic2(game, player):
    """
    Difference between the number of open spaces around the location of the players
    """
    location_player = game.get_player_location(player)
    location_opp = game.get_player_location(game.get_opponent(player))
    blank_spaces = game.get_blank_spaces()
    open_space_player = 0.0
    open_space_opp = 0.0
    for (m, n) in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
        if (location_player[0] + m, location_player[1] + n) in blank_spaces:
            open_space_player += 1
        if (location_opp[0] + m, location_opp[1] + n) in blank_spaces:
            open_space_opp += 1
    return open_space_player - open_space_opp


def heuristic1(game, player):
    """
    Number my moves vs opposition moves
    """
    num_moves_player = len(game.get_legal_moves(player))
    num_moves_opposition = len(game.get_legal_moves(game.get_opponent(player)))
    return float(num_moves_player - num_moves_opposition)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # if len(game.get_blank_spaces())>=total_board_spaces/2:
    #     return heuristic1(game, player)
    # else:
    #     return heuristic2(game, player)
    # weight number of moves more highly at the beginning of the game, then amount of open space
    # and finally, maximum path length (winner will have max length)
    # each measure for player is compared to opponent's measure
    prop = len(game.get_blank_spaces()) / total_board_spaces
    if prop<0.2:
        return heuristic5(game,player)
    return prop * heuristic1(game, player) + heuristic2(game, player)


total_board_spaces = 0.1


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate successors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        total_board_spaces = len(game.get_blank_spaces())
        move = (-1, -1)
        if self.method == 'minimax':
            method = self.minimax
        elif self.method == 'alphabeta':
            method = self.alphabeta
        else:
            raise Exception("method not implemented")

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative is True:
                depth = 1
                while True:
                    _, move = method(game, depth)
                    depth += 1
                    if time_left() <= 10:
                        return move
            else:
                score, move = method(game, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True, player=None):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        player : object
            Player perspective for score calculation

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if player is None:
            player = game.active_player

        if maximizing_player:
            best_score = float("-inf")
        else:
            best_score = float("inf")

        best_move = (-1, -1)
        for legal_move in game.get_legal_moves():
            new_game = game.forecast_move(legal_move)
            if depth > 1:
                score, next_move = self.minimax(new_game, depth - 1, not maximizing_player, player)
            else:
                score = self.score(new_game, player)

            if maximizing_player:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)
            if best_score == score:
                # best_score updated, so update best_move accordingly
                best_move = legal_move

        return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True, player=None):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        player : object
            Player perspective for score calculation

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if player is None:
            player = game.active_player

        if maximizing_player:
            best_score = float("-inf")
        else:
            best_score = float("inf")

        best_move = (-1, -1)
        for legal_move in game.get_legal_moves():
            new_game = game.forecast_move(legal_move)
            if depth > 1:
                score, next_move = self.alphabeta(new_game, depth - 1, alpha, beta, not maximizing_player, player)
            else:
                # terminating case
                score = self.score(new_game, player)

            if maximizing_player:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)

            if best_score == score:
                # best_score updated, so update best_move accordingly
                best_move = legal_move
            if (not maximizing_player and best_score <= alpha) or (maximizing_player and best_score >= beta):
                break

        return best_score, best_move
