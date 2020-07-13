# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Iterator, Tuple, Callable, Optional, List, Dict
import copy

import torch

import polygames

from . import utils
from .params import GameParams, ModelParams, SimulationParams, ExecutionParams
from .env_creation_helpers import (
    sanitize_game_params,
    create_model,
    create_game,
    create_player,
)


def convert_checkpoint(
    command_history: utils.CommandHistory,
    game_params: GameParams,
    model_params: ModelParams,
    out: str,
    skip: List[str]
):
    checkpoint = utils.load_checkpoint(
        checkpoint_path=model_params.init_checkpoint)
    old_model_params = checkpoint["model_params"]
    old_game_params = checkpoint["game_params"]
    model_state_dict = checkpoint["model_state_dict"]

    print(old_model_params.model_name)
    print(getattr(old_model_params, "model_name"))

    new_model_params = copy.deepcopy(old_model_params)
    new_game_params = copy.deepcopy(old_game_params)
    for k, v in vars(model_params).items():
        if not command_history.last_command_contains(k) or k == "init_checkpoint":
            continue
        ov = getattr(new_model_params, k)
        if v != ov:
            print("Changing %s from %s to %s" % (k, ov, v))
            setattr(new_model_params, k, v)
    for k, v in vars(game_params).items():
        if not command_history.last_command_contains(k):
            continue
        ov = getattr(new_game_params, k)
        if v != ov:
            print("Changing %s from %s to %s" % (k, ov, v))
            setattr(new_game_params, k, v)

    m = create_model(game_params=new_game_params,
                     model_params=new_model_params)
    s = m.state_dict()
    params_added = 0
    params_removed = 0
    params_reinitialized = 0
    for k, src in model_state_dict.items():
        if not k in s:
            print("%s shape %s removed" % (k, src.shape))
            params_removed += src.numel()
    for k, dst in s.items():
        if skip is not None and k in skip:
            print("%s shape %s skipped" % (k, src.shape))
            params_reinitialized += dst.numel()
            continue
        if not k in model_state_dict:
            params_added += dst.numel()
            continue
        src = model_state_dict[k]
        delta = dst.numel() - src.numel()
        if delta > 0:
            params_added += delta
        else:
            params_removed -= delta
        if dst.shape != src.shape:
            print("%s shape %s -> %s" % (k, src.shape, dst.shape))
        while dst.dim() < src.dim():
            dst = dst.unsqueeze(0)
        while src.dim() < dst.dim():
            src = src.unsqueeze(0)
        for i in range(dst.dim()):
            if src.shape[i] > dst.shape[i]:
                src = src.narrow(i, 0, dst.shape[i])
            if dst.shape[i] > src.shape[i]:
                dst = dst.narrow(i, 0, src.shape[i])
        dst.copy_(src)
    print("Parameters added: %d" % params_added)
    print("Parameters removed: %d" % params_removed)
    print("Parameters reinitialized: %d" % params_reinitialized)
    checkpoint["model_state_dict"] = s
    checkpoint["model_params"] = new_model_params
    checkpoint["game_params"] = new_game_params
    import gzip
    with gzip.open(out, "wb") as f:
        torch.save(checkpoint, f)
