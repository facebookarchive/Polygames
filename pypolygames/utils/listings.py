import re
import typing as tp
from pathlib import Path


def games(olympiads: bool = False) -> tp.List[str]:
    """List games using pattern in core/game.h

    Parameters
    ----------
    olympiads: bool
        only list olympiad games
    """
    if olympiads:
        pies = {"Hex11pie", "Hex13pie", "Hex19pie", "Havannah5pie", "Havannah8pie"} & set(
            games()
        )  # to ready yet
        return [
            "BlockGo",
            "Einstein",
            "Othello8",
            "Othello10",
            "Othello16",
            "Minishogi",
            "DiceShogi",
            "Surakarta",
            "Breakthrough",
            "Tristannogo",
            "GameOfTheAmazons",
        ] + list(pies)
    filepath = Path(__file__).parents[2] / "core" / "game.h"
    assert filepath.exists()
    pattern = r".*if\s*?\(\s*?isGameNameMatched\s*?\(\s*?\{\s*?\"(?P<name>\w+)\"[^}]*\}\s*?\)\s*?\)\s*?\{.*"
    iterator = re.finditer(pattern, filepath.read_text())
    return list(
        x.group("name") for x in iterator if not x.group().strip().startswith("//")
    )
