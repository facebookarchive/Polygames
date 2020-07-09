# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict
from .. import params
from .. import env_creation_helpers

import torch


MODELS: Dict[str, torch.jit.ScriptModule] = {}


def register_model(cls):
    MODELS[cls.__name__] = cls
    return cls


def get_game_info(game_params: params) -> Dict[str, list]:
    game = env_creation_helpers.create_game(
        game_params=game_params,
        num_episode=-1,
        seed=0,
        eval_mode=False,
        per_thread_batchsize=0,
    )
    info = {"feature_size": game.get_feat_size(), "action_size": game.get_action_size()}
    info["raw_feature_size"] = game.get_raw_feat_size()
    return info


def get_consistent_padding_from_nnks(nnks: int, dilation: int = 1) -> int:
    # the params are such than the output layer of a Conv2d is the same
    # size as the input layer assuming the stride is one
    padding = dilation * (nnks - 1) / 2
    if padding != int(padding):
        raise ValueError(
            "The values of nnks, padding and dilation must be integers "
            "such as 2 * padding == dilation * (nnks - 1) - "
            f"nnks={nnks} dilation={dilation} padding={padding} - "
            "for default values for dilation and padding, nnks should be even"
        )
    return int(padding)
