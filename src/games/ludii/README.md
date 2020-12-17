# Polygames + Ludii integration

We have implemented a bridge between Polygames' tree search and learning algorithms, and the large library of games implemented in the
[Ludii general game system](https://ludii.games/). The game logic is run in Ludii, and training logic / action selection are performed
by Polygames. In theory, this can work for **any game** that can be run in Ludii. In practice, there may be some games that fail
(such as extremely large games that run out of memory, or games with a complex state representation for which appropriate support 
for building tensors has not yet been built into Ludii), but many hundreds of games should work.

## Requirements

Since Ludii uses Java, the Ludii integration of Polygames requires the optional step of installing `openjdk`
from [Polygames' main installation instructions](https://github.com/facebookincubator/Polygames) to be followed.

When building Polygames, make sure **not** to use the `-DWITH_LUDII=OFF` argument for `cmake`, because that will
disable support for Ludii.

## Installation

After installing Polygames as per usual, Ludii itself must also be installed in the correct place such that Polygames
can find and run it. More specifically:

1. Download any desired version of the Ludii player from https://ludii.games/download.php (at least versions 1.1.6 and higher
should run correctly, some older versions may also still work well).
2. Rename the downloaded file from `Ludii-X.Y.Z.jar` to `Ludii.jar`, and place it in `<Polygames install directory>/ludii/Ludii.jar`
(create a new `ludii` directory under `Polygames` if it does not already exist).

## Using Ludii Games

Any command-line option in Polygames that accepts `--game_name` arguments (such as `train`, `eval`, etc.) can also run any game
through Ludii by specifying it in the following format:

```
--game_name="Ludii<NAME>.lud"
```

The `<NAME>` part of such an argument must match the name of the game as it is inside Ludii exactly, including whitespaces. This
works in the same way as [programmatic loading of games in Java when using Ludii as a library for Java code](https://ludiitutorials.readthedocs.io/en/latest/loading_games.html).
The exact game names are also displayed inside the game loader of the GUI of Ludii, which is visible when the Ludii jar is run as
an executable.

For example, a training run with otherwise default arguments for Ludii's implementation of Tic-Tac-Toe (as opposed to the built-in C++
implementation of the game in Polygames) can be launched using:

```
python -m pypolygames train --game_name="LudiiTic-Tac-Toe.lud"
```

## Using Game Options

For many of its games, Ludii also provides additional *options* that can be used to load different variants of a game, with
different board sizes, board shapes, different rulesets, etc. Non-default variants of any Ludii game can also be loaded
in Polygames, through an additional `--game_options` argument followed up by any arbitrary number of Strings, which are
subsequently all passed into Ludii as options. These should again be provided in the same format as when options are provided
[programmatically to Ludii from Java](https://ludiitutorials.readthedocs.io/en/latest/loading_games.html), and the exact
strings to be entered can also be found in Ludii's GUI from the options menu after loading a particular game. For example,
we can launch a training run in Ludii's implementation of Hex, with a board size of 13x13 and an inverted win condition 
("Misere") as follows:

```
python -m pypolygames train --game_name="LudiiHex.lud" --game_options "Board Size/13x13" "End Rules/Misere"
```

## Trained Models

Checkpoints of training runs for some Ludii games have been made [publicly available here](http://dl.fbaipublicfiles.com/polygames/ludii_checkpoints/list.txt).
Each of these checkpoints was trained on the default variant of its game (no custom options specified), for 20 hours on 8 GPUs and 80 CPU cores.