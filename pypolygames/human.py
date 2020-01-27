# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Tuple, Callable, Optional, List, Dict

import torch

import tube
import polygames
from pytube.data_channel_manager import DataChannelManager

from . import utils
from .params import GameParams, ModelParams, SimulationParams, ExecutionParams
from .env_creation_helpers import (
    sanitize_game_params,
    create_model,
    create_game,
    create_player,
)


#######################################################################################
# HUMAN-PLAYED ENVIRONMENT CREATION
#######################################################################################


def create_human_environment(
    seed_generator: Iterator[int],
    game_params: GameParams,
    simulation_params: SimulationParams,
    execution_params: ExecutionParams,
    pure_mcts: bool,
) -> Tuple[tube.Context, Optional[tube.DataChannel], Callable[[], int]]:
    human_first = execution_params.human_first
    time_ratio = execution_params.time_ratio
    total_time = execution_params.total_time
    context = tube.Context()
    actor_channel = (
        None if pure_mcts else tube.DataChannel("act", simulation_params.num_actor, 1)
    )
    game = create_game(
        game_params,
        num_episode=1,
        seed=next(seed_generator),
        eval_mode=True,
        per_thread_batchsize=0,
    )
    player = create_player(
        seed_generator=seed_generator,
        game=game,
        num_actor=simulation_params.num_actor,
        num_rollouts=simulation_params.num_rollouts,
        pure_mcts=pure_mcts,
        actor_channel=actor_channel,
        assembler=None,
        human_mode=True,
        total_time=total_time,
        time_ratio=time_ratio,
    )
    human_player = polygames.HumanPlayer()
    if game.is_one_player_game():
        game.add_human_player(human_player)
    else:
        if human_first:
            game.add_human_player(human_player)
            game.add_eval_player(player)
        else:
            game.add_eval_player(player)
            game.add_human_player(human_player)

    context.push_env_thread(game)

    def get_result_for_human_player():
        nonlocal game, human_first
        return game.get_result()[not human_first]

    return context, actor_channel, get_result_for_human_player


def create_tp_environment(
    seed_generator: Iterator[int],
    game_params: GameParams,
    simulation_params: SimulationParams,
    execution_params: ExecutionParams,
    pure_mcts: bool,
) -> Tuple[tube.Context, Optional[tube.DataChannel], Callable[[], int]]:
    human_first = execution_params.human_first
    time_ratio = execution_params.time_ratio
    total_time = execution_params.total_time
    context = tube.Context()
    actor_channel = (
        None if pure_mcts else tube.DataChannel("act", simulation_params.num_actor, 1)
    )
    game = create_game(
        game_params,
        num_episode=1,
        seed=next(seed_generator),
        eval_mode=True,
        per_thread_batchsize=0,
    )
    player = create_player(
        seed_generator=seed_generator,
        game=game,
        num_actor=simulation_params.num_actor,
        num_rollouts=simulation_params.num_rollouts,
        pure_mcts=pure_mcts,
        actor_channel=actor_channel,
        assembler=None,
        human_mode=True,
        total_time=total_time,
        time_ratio=time_ratio,
    )
    tp_player = polygames.TPPlayer()
    if game.is_one_player_game():
        game.add_tp_player(tp_player)
    else:
        if human_first:
            game.add_tp_player(tp_player)
            game.add_eval_player(player)
        else:
            game.add_eval_player(player)
            game.add_tp_player(tp_player)

    context.push_env_thread(game)

    def get_result_for_tp_player():
        nonlocal game, human_first
        return game.get_result()[not human_first]

    return context, actor_channel, get_result_for_tp_player


#######################################################################################
# HUMAN-PLAYED GAME
#######################################################################################


def _forward_pass_on_device(
    device: torch.device, model: torch.jit.ScriptModule, batch_s: torch.Tensor
) -> Dict[str, torch.Tensor]:
    batch_s = utils.to_device(batch_s, device)
    with torch.no_grad():
        reply = model(batch_s)
    return reply


def _play_game_against_mcts(context: tube.Context) -> None:
    context.start()
    while not context.terminated():
        time.sleep(1)


def _play_game_against_neural_mcts(
    devices: List[torch.device],
    models: List[torch.jit.ScriptModule],
    context: tube.Context,
    actor_channel: tube.DataChannel,
) -> None:
    nb_devices = len(devices)
    context.start()
    dcm = DataChannelManager([actor_channel])
    while not context.terminated():
        batch = dcm.get_input(max_timeout_s=1)
        if len(batch) == 0:
            continue

        assert len(batch) == 1

        # split in as many part as there are devices
        batches_s = torch.chunk(
            batch[actor_channel.name]["s"], nb_devices, dim=0
        )
        futures = []
        reply_eval = {"v": None, "pi": None}
        # multithread
        with ThreadPoolExecutor(max_workers=nb_devices) as executor:
            for device, model, batch_s in zip(
                devices, models, batches_s
            ):
                futures.append(
                    executor.submit(_forward_pass_on_device, device, model, batch_s)
                )
            results = [future.result() for future in futures]
            reply_eval["v"] = torch.cat([result["v"] for result in results], dim=0)
            reply_eval["pi"] = torch.cat([result["pi"] for result in results], dim=0)
        dcm.set_reply(actor_channel.name, reply_eval)
    dcm.terminate()


def play_game(
    pure_mcts: bool,
    devices: Optional[List[torch.device]],
    models: Optional[List[torch.jit.ScriptModule]],
    context: tube.Context,
    actor_channel: Optional[tube.DataChannel],
    get_result_for_human_player: Callable[[], int],
) -> int:
    if pure_mcts:
        _play_game_against_mcts(context)
    else:
        _play_game_against_neural_mcts(
            devices=devices, models=models, context=context, actor_channel=actor_channel
        )
    print("game over")
    return get_result_for_human_player()


def play_tp_game(   #FIXME TODO not sure this helps
    pure_mcts: bool,
    devices: Optional[List[torch.device]],
    models: Optional[List[torch.jit.ScriptModule]],
    context: tube.Context,
    actor_channel: Optional[tube.DataChannel],
    get_result_for_human_player: Callable[[], int],
) -> int:
    if pure_mcts:
        _play_game_against_mcts(context)
    else:
        _play_game_against_neural_mcts(
            devices=devices, models=models, context=context, actor_channel=actor_channel
        )
    print("#game over")
    return get_result_for_human_player()


#######################################################################################
# OVERALL HUMAN-PLAYED GAME WORKFLOW
#######################################################################################


def run_human_played_game(
    game_params: GameParams,
    model_params: ModelParams,
    simulation_params: SimulationParams,
    execution_params: ExecutionParams,
):
    print("#" * 70)
    print("#" + "HUMAN-PLAYED GAME".center(68) + "#")
    print("#" * 70)

    print("setting-up pseudo-random generator...")
    seed_generator = utils.generate_random_seeds(seed=execution_params.seed)

    devices, models = None, None
    if not model_params.pure_mcts:
        print("loading pretrained model from checkpoint...")
        checkpoint = utils.load_checkpoint(checkpoint_path=model_params.init_checkpoint)
        game_params = checkpoint["game_params"]
        sanitize_game_params(game_params)
        model_params = checkpoint["model_params"]
        model_state_dict = checkpoint["model_state_dict"]
        del checkpoint
        print("creating model(s) and device(s)...")
        models = []
        devices = [torch.device(device) for device in execution_params.device]
        for device in devices:
            model = create_model(game_params=game_params, model_params=model_params).to(
                device
            )
            print("updating model...")
            model.load_state_dict(model_state_dict)
            model.eval()
            models.append(model)

    print("creating human-played environment")
    context, actor_channel, get_result_for_human_player = create_human_environment(
        seed_generator=seed_generator,
        game_params=game_params,
        simulation_params=simulation_params,
        execution_params=execution_params,
        pure_mcts=model_params.pure_mcts,
    )

    print("playing against a human player...")
    human_score = play_game(
        pure_mcts=model_params.pure_mcts,
        devices=devices,
        models=models,
        context=context,
        actor_channel=actor_channel,
        get_result_for_human_player=get_result_for_human_player,
    )

    print(f"result for the human human_player: {human_score}")


def run_tp_played_game(
    game_params: GameParams,
    model_params: ModelParams,
    simulation_params: SimulationParams,
    execution_params: ExecutionParams,
):
    print("#" * 70)
    print("#" + "HUMAN-PLAYED GAME".center(68) + "#")
    print("#" * 70)

    print("#setting-up pseudo-random generator...")
    seed_generator = utils.generate_random_seeds(seed=execution_params.seed)

    devices, models = None, None
    if not model_params.pure_mcts:
        print("#loading pretrained model from checkpoint...")
        checkpoint = utils.load_checkpoint(checkpoint_path=model_params.init_checkpoint)
        game_params = checkpoint["game_params"]
        sanitize_game_params(game_params)
        model_params = checkpoint["model_params"]
        model_state_dict = checkpoint["model_state_dict"]
        del checkpoint
        print("#creating model(s) and device(s)...")
        models = []
        devices = [torch.device(device) for device in execution_params.device]
        for device in devices:
            model = create_model(game_params=game_params, model_params=model_params).to(
                device
            )
            print("#updating model...")
            model.load_state_dict(model_state_dict)
            model.eval()
            models.append(model)

    print("#creating human-played environment")
    context, actor_channel, get_result_for_human_player = create_tp_environment(
        seed_generator=seed_generator,
        game_params=game_params,
        simulation_params=simulation_params,
        execution_params=execution_params,
        pure_mcts=model_params.pure_mcts,
    )

    print("#playing against a tp player...")
    human_score = play_tp_game(
        pure_mcts=model_params.pure_mcts,
        devices=devices,
        models=models,
        context=context,
        actor_channel=actor_channel,
        get_result_for_human_player=get_result_for_human_player,
    )

    print(f"#result for the TP_player: {human_score}")
