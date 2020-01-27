# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys


root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
game = os.path.join(root, "build")
if game not in sys.path:
    sys.path.append(game)

tube = os.path.join(root, "build", "torchRL", "tube")
if tube not in sys.path:
    sys.path.append(tube)
pytube = os.path.join(root, "torchRL", "tube")
if pytube not in sys.path:
    sys.path.append(pytube)

mcts = os.path.join(root, "build", "torchRL", "mcts")
if mcts not in sys.path:
    sys.path.append(mcts)
