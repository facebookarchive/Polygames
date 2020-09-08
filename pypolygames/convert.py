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

    fix_global_pooling = old_model_params.model_name == "ResConvConvLogitPoolModelV2" and new_model_params.model_name == "ResConvConvLogitPoolModelV2" and new_model_params.global_pooling > 0
    if fix_global_pooling:
        print("Note: attempting to patch global pooling weights to match the new model")

    m = create_model(game_params=new_game_params,
                     model_params=new_model_params)
    s = m.state_dict()
    params_added = 0
    params_removed = 0
    params_reinitialized = 0
    taken = []
    for k, src in model_state_dict.items():
        if not k in s:
            moved = False
            for k2, dst in s.items():
                if not k2 in model_state_dict and src.shape == dst.shape and not k2 in taken:
                  print("%s shape %s moved to %s" % (k, src.shape, k2))
                  taken.append(k2)
                  dst.copy_(src)
                  moved = True
                  break
            if not moved:
              print("%s shape %s removed" % (k, src.shape))
              params_removed += src.numel()
    for k, dst in s.items():
        if k in taken:
            continue
        if skip is not None and k in skip:
            print("%s shape %s skipped" % (k, dst.shape))
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
        if fix_global_pooling and "resnets" in k and "0.0" in k and "weight" in k:
          if src.dim() == 4 and dst.dim() == 4:

            src_c = src.shape[0]
            dst_c = dst.shape[0]

            src_s = int(src_c * old_model_params.global_pooling)
            dst_s = int(dst_c * new_model_params.global_pooling)

            src_d = src_c + src_s
            dst_d = dst_c + src_s

            print("Moving global pooling weights from %d:%d to %d:%d" % (src_c, src_d, dst_c, dst_d))

            min_c = min(src_c, dst_c)
            dst[:min_c, dst_c:dst_d, :, :] = src[:min_c, src_c:src_d, :, :]
            #dst.narrow(0, 0, src_c).narrow(1, dst_c, src_s).copy_(src.narrow(1, src_c, src_s))

            print("Moving global pooling weights from %d:%d to %d:%d" % (src_c+src_s, src_d+src_s, dst_c+dst_s, dst_d+dst_s))

            #dst.narrow(0, 0, src_c).narrow(1, dst_c+dst_s, src_s).copy_(src.narrow(1, src_c+src_s, src_s))
            dst[:min_c, dst_c+dst_s:dst_d+dst_s, :, :] = src[:min_c, src_c+src_s:src_d+src_s, :, :]

            src = src[:, :src_c, :, :]
            dst = dst[:, :dst_c, :, :]
            #src = src.narrow(1, 0, src_c)
            #dst = dst.narrow(1, 0, dst_c)

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
