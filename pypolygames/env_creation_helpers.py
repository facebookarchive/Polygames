# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Iterator

import torch  # must be loaded before tube
import tube
import mcts
import polygames

from . import model_zoo
from .params import GameParams, ModelParams
from .weight_init import WEIGHT_INIT


def sanitize_game_params(game_params: GameParams) -> None:
    # eval and human modes do not support `per_thread_batchsize` != 0
    # while in training, the option could have been set to > 0
    # ideally this should be a simulation parameter, but it would change
    # the C++ 'polygames.Game' signature
    # EDIT: now it a simulation parameter, but for retro-compatibility
    #  we keep that function
    game_params.per_thread_batchsize = 0


def create_game(
    game_params: GameParams,
    num_episode: int,
    seed: int,
    eval_mode: bool,
    per_thread_batchsize: int = 0,
) -> polygames.Game:
    # Many old models don't have the game_options attribute
    if hasattr(game_params, 'game_options'):
        game_options = game_params.game_options
    else:
        game_options = list()

    return polygames.Game(
        game_params.game_name,
        game_options,
        num_episode,
        seed,
        eval_mode,
        game_params.out_features,
        game_params.turn_features,
        game_params.geometric_features,
        game_params.history,
        game_params.random_features,
        game_params.one_feature,
        per_thread_batchsize,
    )  # cannot use named parameters :(


def create_model(
    game_params: GameParams,
    model_params: ModelParams,
    resume_training: bool = False,
    model_state_dict: Optional[dict] = None,
) -> torch.jit.ScriptModule:
    if model_params.model_name is not None:
        if model_params.model_name in model_zoo.MODELS:
            model = model_zoo.MODELS[model_params.model_name](
                game_params=game_params, model_params=model_params
            )
        else:
            raise RuntimeError(
                f'The model "{model_params.model_name}" has not been implemented '
                f'in the "model_zoo" package'
            )
    else:
        print("creating a generic model")
        model = model_zoo.GenericModel(
            game_params=game_params, model_params=model_params
        )
    if resume_training:
        if model_state_dict is not None:
            print("load state dict!")
            model.load_state_dict(model_state_dict)
    else:
        model.apply(WEIGHT_INIT[model_params.init_method])

    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total #trainable params = {nb_params}")
    return model


def _set_mcts_option(
    num_rollouts: int,
    seed: int,
    human_mode: bool = False,
    time_ratio: float = 0.7,
    total_time: float = 0,
) -> mcts.MctsOption:
    # TODO: put hardcoded value in conf file
    mcts_option = mcts.MctsOption()
    mcts_option.use_mcts = True
    mcts_option.puct = 1
    mcts_option.sample_before_step_idx = 2 if human_mode else 32
    mcts_option.num_rollout_per_thread = num_rollouts
    mcts_option.seed = seed
    mcts_option.virtual_loss = 1
    mcts_option.total_time = total_time
    mcts_option.time_ratio = time_ratio
    return mcts_option


def _create_pure_mcts_player(
    game: polygames.Game, mcts_option: mcts.MctsOption, num_actor: int
) -> mcts.MctsPlayer:
    """a player that uses only mcts + random rollout, no neural net"""
    player = mcts.MctsPlayer(mcts_option)
    for _ in range(num_actor):
        actor = polygames.Actor(
            None, game.get_feat_size(), game.get_action_size(), False, False, None
        )
        player.add_actor(actor)
    return player


def _create_neural_mcts_player(
    game: polygames.Game,
    mcts_option: mcts.MctsOption,
    num_actor: int,
    actor_channel: tube.DataChannel,
    assembler: Optional[tube.ChannelAssembler] = None,
) -> mcts.MctsPlayer:
    player = mcts.MctsPlayer(mcts_option)
    for _ in range(num_actor):
        num_actor += 1
        actor = polygames.Actor(
            actor_channel,
            game.get_feat_size(),
            game.get_action_size(),
            True,
            True,
            assembler,
        )
        player.add_actor(actor)
    return player


def create_player(
    seed_generator: Iterator[int],
    game: polygames.Game,
    num_actor: int,
    num_rollouts: int,
    pure_mcts: bool,
    actor_channel: Optional[tube.DataChannel],
    assembler: Optional[tube.ChannelAssembler] = None,
    human_mode: bool = False,
    time_ratio: float = 0.07,
    total_time: float = 0,
):
    mcts_option = _set_mcts_option(
        num_rollouts=num_rollouts,
        seed=next(seed_generator),
        human_mode=human_mode,
        time_ratio=time_ratio,
        total_time=total_time,
    )
    if pure_mcts:
        return _create_pure_mcts_player(
            game=game, mcts_option=mcts_option, num_actor=num_actor
        )
    else:
        return _create_neural_mcts_player(
            game=game,
            mcts_option=mcts_option,
            num_actor=num_actor,
            actor_channel=actor_channel,
            assembler=assembler,
        )
