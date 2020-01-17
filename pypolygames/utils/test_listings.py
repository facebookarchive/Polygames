# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from . import listings


def test_lists() -> None:
    includes = {"Connect4", "TicTacToe", "Othello8", "Othello16", "GameOfTheAmazons",
                "Hex5", "Hex11", "Hex13", "Connect6",
                "Havannah5", "Havannah8", "Breakthrough", "Tristannogo",
                "Minishogi", "Surakarta", "DiceShogi"}
    listed_items = listings.games()
    duplicated = {x: y for x, y in Counter(listed_items).items() if y > 1}
    assert duplicated == {}
    missing = includes - set(listed_items)
    assert not missing, f"Could not find {missing} (screening through core/game.h or model_zoo/init or main), was it renamed?"
