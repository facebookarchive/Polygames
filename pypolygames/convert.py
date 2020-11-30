# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Iterator, Tuple, Callable, Optional, List, Dict
from pathlib import Path
import copy

import torch

import polygames

from . import utils
from .model_zoo import utils as zutils
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
    skip: List[str],
    auto_tune_nnsize: bool,
    zero_shot: bool,
    move_source_channels: List[int],
    state_source_channels: List[int],
):
    checkpoint = utils.load_checkpoint(
        checkpoint_path=model_params.init_checkpoint)
    old_model_params = checkpoint["model_params"]
    old_game_params = checkpoint["game_params"]
    sanitize_game_params(old_game_params) # backwards compatibility for models without game_options
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
        
    if zero_shot:
        print("Note: converting model for zero-shot evaluation only! Added/reinitialized params will be all 0 and untrainable!")
        
    if auto_tune_nnsize or move_source_channels is not None or state_source_channels is not None:
        # We'll need to load the game info
        old_game_info = zutils.get_game_info(old_game_params)
        new_game_info = zutils.get_game_info(new_game_params)
        
    if auto_tune_nnsize:
        # We want to automatically tune nnsize, such that the number
        # of filters in hidden layers does not change from source to
        # target model
        c_old, _, _ = old_game_info["feature_size"][:3]
        c_new, _, _ = new_game_info["feature_size"][:3]
        new_nnsize = float((getattr(old_model_params, 'nnsize') * c_old) / c_new)
        print("Auto-tuning nnsize to:", new_nnsize)
        setattr(new_model_params, 'nnsize', new_nnsize)
        
    if move_source_channels is not None:
        c_action_new, _, _ = new_game_info["action_size"][:3]
        if c_action_new != len(move_source_channels):
            print("ERROR: if --move_source_channels is specified, it must have exactly c_action_new entries!")
            print("c_action_new = ", c_action_new)
            print("len(move_source_channels) = ", len(move_source_channels))
            
    if state_source_channels is not None:
        c_state_new, _, _ = new_game_info["feature_size"][:3]
        if c_state_new != len(state_source_channels):
            print("ERROR: if --state_source_channels is specified, it must have exactly c_state_new entries!")
            print("c_state_new = ", c_state_new)
            print("len(state_source_channels) = ", len(state_source_channels))

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
            
        if zero_shot:
            dst = dst.fill_(0) 
        
        if skip is not None and k in skip:
            print("%s shape %s skipped" % (k, dst.shape))
            params_reinitialized += dst.numel()
            continue
        if not k in model_state_dict:
            params_added += dst.numel()
            continue
        src = model_state_dict[k]
        
        if move_source_channels is not None and "pi_logit." in k:
            # Use manually specified channels to transfer from for
            # last Conv2D operation that produces pi logits
            if "weight" in k:
                for i in range(len(move_source_channels)):
                    if move_source_channels[i] >= 0:
                        dst_view = dst
                        src_view = src
                        for j in range(dst_view.dim()):
                            if j == 0:  # Don't narrow this dim, need original indexing
                                continue
                            if src_view.shape[j] > dst_view.shape[j]:
                                src_view = src_view.narrow(j, 0, dst_view.shape[j])
                            if dst_view.shape[j] > src_view.shape[j]:
                                dst_view = dst_view.narrow(j, 0, src_view.shape[j])
                    
                        dst_view[i] = src_view[move_source_channels[i]]
            elif "bias" in k:
                for i in range(len(move_source_channels)):
                    if move_source_channels[i] >= 0:
                        dst[i] = src[move_source_channels[i]]
            continue
            
        if state_source_channels is not None and "mono.0." in k:
            # Use manually specified channels to transfer from for
            # first Conv2D operation on state tensor
            if "weight" in k:
                for i in range(len(state_source_channels)):
                    if state_source_channels[i] >= 0:
                        dst_view = dst
                        src_view = src
                        for j in range(dst_view.dim()):
                            if j == 1:  # Don't narrow this dim, need original indexing
                                continue
                            if src_view.shape[j] > dst_view.shape[j]:
                                src_view = src_view.narrow(j, 0, dst_view.shape[j])
                            if dst_view.shape[j] > src_view.shape[j]:
                                dst_view = dst_view.narrow(j, 0, src_view.shape[j])
                    
                        dst_view[:, i] = src_view[:, state_source_channels[i]]
            elif "bias" in k:
                for i in range(len(state_source_channels)):
                    if state_source_channels[i] >= 0:
                        dst[i] = src[state_source_channels[i]]
            continue
        
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
    
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    import gzip
    with gzip.open(out, "wb") as f:
        torch.save(checkpoint, f)
