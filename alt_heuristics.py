from unionfind import UnionFind


def connected_components_with_backoff(game, player):
    """
    Combination of two heuristics with back-off from one to another
    """
    score = connected_components_score(game, player)
    return weighted_number_of_moves(game, player) if score == 0 else score


def weighted_number_of_moves(game, player):
    return float(len(game.get_legal_moves(player))) \
           - float(len(game.get_legal_moves(game.get_opponent(player)))) * 1.5


def connected_components_score(game, player):
    return float(connected_components_diff(game, player))


def baseline(game, player):
    return float(len(game.get_legal_moves(player))) \
           - float(len(game.get_legal_moves(game.get_opponent(player))))


def neighbors(game, position):
    """
    Getting neighbors positions from given one
    """
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    return [(position[0] + x, position[1] + y)
            for (x, y) in directions
            if game.move_is_legal((position[0] + x, position[1] + y))]


def connected_components_diff(game, player):
    """
    Difference between number of connected components
    of one player and its opponent
    """
    size = game.width * game.height
    uf = UnionFind(size)
    blank = game.get_blank_spaces()
    for bs in blank:
        for n in neighbors(game, bs):
            uf.union(bs, n)
    player_location = game.get_player_location(player)
    opp_location = game.get_player_location(game.get_opponent(player))
    for n in neighbors(game, player_location):
        uf.union(n, player_location)
    for n in neighbors(game, opp_location):
        uf.union(n, opp_location)

    pl_score = float(uf.components(player_location))
    op_score = float(uf.components(opp_location))
    return pl_score - op_score
