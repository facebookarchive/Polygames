# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import random
from unittest import SkipTest
from pathlib import Path
import pytest
from .. import params
from .. import evaluation
from .. import utils


@pytest.mark.parametrize(
    "game_name", [game_name for game_name in utils.listings.games()]
)
def test_mcts(game_name) -> None:
    #
    # Important informations in the following block about which games are skipped because they are:
    # - too slow (more than 1min)
    # - crashing (segfault or crash with no more information)
    # - too_bad: not better with larger rollouts (these games are played but not evaluated)
    # also, for some games, we must add some tolerance (they dont win at 100%)
    #
    crashing = []
    is_one_player_game = any(x in game_name for x in ["asterm", "ineswee", "WeakSchur"])
    too_slow = [
        "GameOfTheAmazons",
        "Connect6",
        "KyotoShogi",
        "Hex19",
        "Hex19pie",
        "Minishogi",
        "DiceShogi",
        "ChineseCheckers",
    ]
    too_bad = ["Havannah5", "Havannah5pie", "Surakarta", "DiceShogi", "Connect6"]
    if game_name in crashing + too_slow:  # + one_player_games:
        raise SkipTest(f"Skipping {game_name}")
    if "inesweeper" in game_name and "4_4_4" not in game_name:
        raise SkipTest(f"Skipping {game_name}")
    if "astermind" in game_name and "4_4_6" not in game_name:
        raise SkipTest(f"Skipping {game_name}")
    if "WeakSchur" in game_name:
        raise SkipTest(f"Skipping {game_name} (currently aborts when finished, which kills the CI)")
    # for allowing some tolerance to winning all games with larger rollouts, add here:
    tolerance = {
        "TicTacToe": 4,
        "FreeStyleGomoku": 4,
        "OuterOpenGomoku": 3,
        "Havannah5pieExt": 2,
        "Havannah8": 5,
        "Havannah8pie": 5,
        "Hex13": 2,
        "Hex13pie": 2,
        "Einstein": 3,
        "Othello10": 2,
        "OthelloOpt10": 2,
        "YINSH": 3,
        "Minishogi": 1,
        "GomokuSwap2": 3,
        "BlockGo": 2,
    }.get(game_name, 1)
    #
    game_params = params.GameParams(game_name=game_name)
    case = random.randint(0, 2)
    rollouts = (2, 40)
    if (
        not case
    ):  # In case 0, 0 wins, else 1  (this makes sure results dependent on rollouts)
        rollouts = tuple(reversed(rollouts))
    eval_params = params.EvalParams(
        num_game_eval=10,
        device_eval="cpu",
        checkpoint_dir=Path("mock/path"),  # this should not be *required* here! no network
        num_rollouts_eval=rollouts[0],
        num_rollouts_opponent=rollouts[1],
    )  # device eval is actually not used

    def seed_generator():
        i = 0
        while True:
            yield i
            i += 1

    context, _, _, get_eval_reward = evaluation.create_evaluation_environment(
        seed_generator=seed_generator(),
        game_params=game_params,
        eval_params=eval_params,
        pure_mcts_eval=True,
    )

    context.start()
    while not context.terminated():
        time.sleep(0.01)
    # check that the one with most rollouts wins!
    score = sum(v > 0 for v in get_eval_reward())
    expected = 0 if case else eval_params.num_game_eval
    msg = f"Wrong score for random case {case}, expected {expected} with tol {tolerance} but got {score}."
    if is_one_player_game or game_name in too_bad:
        raise SkipTest(f"Skipping evaluation of {game_name} (not very good, or one player)")
    assert abs(score - expected) <= tolerance, msg
