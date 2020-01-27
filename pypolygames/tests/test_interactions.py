# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import difflib
import tempfile
import subprocess
from pathlib import Path
from pprint import pprint
from unittest import SkipTest
import pytest
from ..utils import listings


class FileStream:
    """Simplifies stdout reading
    """

    def __init__(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        path = Path(self.tempdir.name) / "std_in_out.txt"
        self.writer = path.open("w")
        self.reader = path.open("r")

    def __del__(self) -> None:
        self.writer.close()
        self.reader.close()


# Specify any specific set of actions for your game
GAME_ACTIONS = {"Breakthrough": ["1", "blublu"],
                "GameOfTheAmazons": ["A7", "B6", "C6", "blublu"],
                "Othello10": ["G6", "blublu"],
                "Othello16": ["J9", "blublu"],
                "Havannah5": ["0,4", "blublu"],
                "Havannah8": ["0,7", "blublu"],
                "Hex11": ["a1", "blublu"],
                "Hex13": ["a1", "blublu"],
                "Surakarta": ["A5-B4", "blublu"],
                "DiceShogi": ["1", "1", "blublu"],
                "ChineseCheckers": ["C4", "G35", "A10", "blublu"],
                }


@pytest.mark.parametrize(
    "game_name", [game_name for game_name in listings.games(olympiads=True)]
)

def test_game_interactions(game_name: str):
    raise SkipTest
    if game_name in ["Einstein", "DiceShogi"]:
        # Feel free to add name here in order to deactivate a test
        raise SkipTest(f"Skipping {game_name} for lack of reproducibility")
    actions = GAME_ACTIONS.get(game_name, ["0", "blublu"])
    # let's play
    fsout = FileStream()
    command = ['timeout', '--signal=SIGTERM', '20', 'python', '-um', 'pypolygames', 'human',
               "--pure_mcts", f'--game_name={game_name}', '--num_rollouts=2', '--seed=12']
    input_requests = ["Input", "Random outcome", "Chess you choose is:", "Where you wanna go:"]  # this may need to be made more robust
    text = ""
    popen = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=fsout.writer)
    try:
        # wait for the process to initialize
        for _ in range(120):  # wait for input request
            if any(x in text for x in input_requests):
                break
            text = fsout.reader.read()
            if text:
                print(text)  # for debugging
            time.sleep(.1)
        for action in actions:
            print(f"*** PLAYING: {action} ***")
            popen.stdin.write((action + "\n").encode())
            popen.stdin.flush()
            text = ""
            for _ in range(20):  # wait for input request
                if any(x in text for x in input_requests):
                    break
                text = fsout.reader.read()
                if text:
                    print(text)
                time.sleep(.1)
    except Exception as e:
        popen.terminate()  # make sure the process is killed, whatever happens
        raise e
    popen.terminate()
    fsout.reader.seek(0)
    all_text = f"actions: {actions}\n" + fsout.reader.read()
    #
    # compare the outputs to records
    filepath = Path(__file__).parent / "data" / f"{game_name}.txt"
    filepath.parent.mkdir(exist_ok=True)
    if not filepath.exists():
        filepath.write_text(all_text)
        raise AssertionError("Logs were written, rerun once again to test reproducibility")
    expected = filepath.read_text()
    if all_text != expected:
        print("\n\n\nHERE IS THE DIFF:\n\n\n")
        pprint(list(difflib.Differ().compare(all_text.splitlines(), expected.splitlines())))
        raise ValueError(f"String differ. If the new string is better, delete {filepath}\n"
                         "and rerun twice (pytest pypolygames/tests/test_interactions)\n"
                         "Alternatively, feel free to add failing games to the list of skipped\n"
                         "tests at the top of this function, and notify jrapin or teytaud\n"
                         "(we'll reactivate it for you later on, since it can be cumbersome).")
