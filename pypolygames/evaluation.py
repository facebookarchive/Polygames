# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Tuple, List, Callable, Optional, Dict

import torch

import tube
from pytube.data_channel_manager import DataChannelManager

from .params import GameParams, EvalParams
from . import utils
from .env_creation_helpers import (
    sanitize_game_params,
    create_model,
    create_game,
    create_player,
)


#######################################################################################
# PLOTTER CREATION
#######################################################################################


def create_plotter(eval_params: EvalParams) -> utils.Plotter:
    checkpoint_dir = eval_params.checkpoint_dir
    if checkpoint_dir[-1] == "/":
        checkpoint_dir = checkpoint_dir[:-1]
    plot_env = os.path.basename(checkpoint_dir)
    return utils.Plotter(
        plot_enabled=eval_params.plot_enabled,
        env=plot_env,
        server=eval_params.plot_server,
        port=eval_params.plot_port,
    )


#######################################################################################
# CHECKPOINT ITERATOR CREATION
#######################################################################################


def create_checkpoint_iter(eval_params: EvalParams, only_last: bool = False):
    if eval_params.checkpoint_dir is not None:
        return utils.gen_checkpoints(
            checkpoint_dir=eval_params.checkpoint_dir,
            real_time=eval_params.real_time and not only_last,
            only_last=only_last,
        )
    else:
        return [utils.load_checkpoint(eval_params.checkpoint)]


#######################################################################################
# OPPONENT MODEL AND DEVICE CREATION
#######################################################################################


def create_models_and_devices_opponent(
    eval_params: EvalParams
) -> Tuple[List[torch.jit.ScriptModule], List[torch.device], GameParams]:
    devices_opponent = [
        torch.device(device_opponent) for device_opponent in eval_params.device_opponent
    ]
    checkpoint_opponent = utils.load_checkpoint(eval_params.checkpoint_opponent)
    model_state_dict_opponent = checkpoint_opponent["model_state_dict"]
    game_params_opponent = checkpoint_opponent["game_params"]
    sanitize_game_params(game_params_opponent)
    model_params_opponent = checkpoint_opponent["model_params"]
    models_opponent = []
    for device_opponent in devices_opponent:
        model_opponent = create_model(
            game_params=game_params_opponent,
            model_params=model_params_opponent,
            resume_training=False,
        ).to(device_opponent)
        model_opponent.load_state_dict(model_state_dict_opponent)
        model_opponent.eval()
        models_opponent.append(model_opponent)
    return models_opponent, devices_opponent, game_params_opponent


#######################################################################################
# EVALUATION ENVIRONMENT CREATION
#######################################################################################


def create_evaluation_environment(
    seed_generator: Iterator[int],
    game_params: GameParams,
    eval_params: EvalParams,
    current_batch_size: int = None,
    pure_mcts_eval: bool = False,
    pure_mcts_opponent: bool = True,
    num_evaluated_games: int = 0
) -> Tuple[
    tube.Context,
    Optional[tube.DataChannel],
    Optional[tube.DataChannel],
    Callable[[], List[int]],
]:
    num_game = eval_params.num_game_eval
    num_actor_eval = eval_params.num_actor_eval
    num_rollouts_eval = eval_params.num_rollouts_eval
    num_actor_opponent = eval_params.num_actor_opponent
    num_rollouts_opponent = eval_params.num_rollouts_opponent
    first_hand = []
    second_hand = []
    games = []

    context = tube.Context()
    actor_channel_eval = (
        None
        if pure_mcts_eval
        else tube.DataChannel("act_eval", num_game * num_actor_eval, 1)
    )
    actor_channel_opponent = (
        None
        if pure_mcts_opponent
        else tube.DataChannel("act_opponent", num_game * num_actor_opponent, 1)
    )
    for game_no in range(current_batch_size if current_batch_size else num_game):
        game = create_game(
            game_params, num_episode=1, seed=next(seed_generator), eval_mode=True
        )
        player = create_player(
            seed_generator=seed_generator,
            game=game,
            num_actor=num_actor_eval,
            num_rollouts=num_rollouts_eval,
            pure_mcts=pure_mcts_eval,
            actor_channel=actor_channel_eval,
            assembler=None,
            human_mode=False,
        )
        if game.is_one_player_game():
            game.add_eval_player(player)
            first_hand.append(game)
        else:
            opponent = create_player(
                seed_generator=seed_generator,
                game=game,
                num_actor=num_actor_opponent,
                num_rollouts=num_rollouts_opponent,
                pure_mcts=pure_mcts_opponent,
                actor_channel=actor_channel_opponent,
                assembler=None,
                human_mode=False,
            )
            game_id = num_evaluated_games + game_no
            if player_moves_first(game_id, num_game):
                game.add_eval_player(player)
                game.add_eval_player(opponent)
                first_hand.append(game)
            else:
                game.add_eval_player(opponent)
                game.add_eval_player(player)
                second_hand.append(game)

        context.push_env_thread(game)
        games.append(game)

    def get_eval_reward():
        nonlocal first_hand, second_hand
        reward = []
        for hand in first_hand:
            reward.append(hand.get_result()[0])
        for hand in second_hand:
            reward.append(hand.get_result()[1])
        return reward

    return context, actor_channel_eval, actor_channel_opponent, get_eval_reward


def player_moves_first(game_id, num_games_eval):
    return game_id < num_games_eval // 2

#######################################################################################
# EVALUATION
#######################################################################################


def _forward_pass_on_device(
    device: torch.device, model: torch.jit.ScriptModule, batch_s: torch.Tensor
) -> Dict[str, torch.Tensor]:
    batch_s = utils.to_device(batch_s, device)
    with torch.no_grad():
        reply = model(batch_s)
    return reply


def _play_game_neural_mcts_against_pure_mcts_opponent(
    context: tube.Context,
    actor_channel_eval: tube.DataChannel,
    devices_eval: List[torch.device],
    models_eval: List[torch.jit.ScriptModule],
) -> None:
    nb_devices_eval = len(devices_eval)
    context.start()
    dcm = DataChannelManager([actor_channel_eval])
    while not context.terminated():
        batch = dcm.get_input(max_timeout_s=1)
        if len(batch) == 0:
            continue

        assert len(batch) == 1  # only one channel

        # split in as many part as there are devices
        batches_eval_s = torch.chunk(
            batch[actor_channel_eval.name]["s"], nb_devices_eval, dim=0
        )
        futures = []
        reply_eval = {"v": None, "pi": None}
        # multithread
        with ThreadPoolExecutor(max_workers=nb_devices_eval) as executor:
            for device, model, batch_s in zip(
                devices_eval, models_eval, batches_eval_s
            ):
                futures.append(
                    executor.submit(_forward_pass_on_device, device, model, batch_s)
                )
            results = [future.result() for future in futures]
            reply_eval["v"] = torch.cat([result["v"] for result in results], dim=0)
            reply_eval["pi"] = torch.cat([result["pi"] for result in results], dim=0)
        dcm.set_reply(actor_channel_eval.name, reply_eval)
    dcm.terminate()


def _play_game_neural_mcts_against_neural_mcts_opponent(
    context: tube.Context,
    actor_channel_eval: tube.DataChannel,
    actor_channel_opponent: tube.DataChannel,
    devices_eval: List[torch.device],
    models_eval: List[torch.jit.ScriptModule],
    devices_opponent: List[torch.device],
    models_opponent: List[torch.jit.ScriptModule],
) -> None:
    nb_devices_eval = len(devices_eval)
    nb_devices_opponent = len(devices_opponent)
    context.start()
    dcm = DataChannelManager([actor_channel_eval, actor_channel_opponent])
    while not context.terminated():
        batch = dcm.get_input(max_timeout_s=1)
        if len(batch) == 0:
            continue

        assert len(batch) <= 2  # up to two channels

        if actor_channel_eval.name in batch:
            # split in as many part as there are devices
            batches_eval_s = torch.chunk(
                batch[actor_channel_eval.name]["s"], nb_devices_eval, dim=0
            )
            futures = []
            reply_eval = {"v": None, "pi": None}
            # multithread
            with ThreadPoolExecutor(max_workers=nb_devices_eval) as executor:
                for device, model, batch_s in zip(
                    devices_eval, models_eval, batches_eval_s
                ):
                    futures.append(
                        executor.submit(_forward_pass_on_device, device, model, batch_s)
                    )
                results = [future.result() for future in futures]
                reply_eval["v"] = torch.cat([result["v"] for result in results], dim=0)
                reply_eval["pi"] = torch.cat(
                    [result["pi"] for result in results], dim=0
                )
            dcm.set_reply(actor_channel_eval.name, reply_eval)

        if actor_channel_opponent.name in batch:
            # split in as many part as there are devices
            batches_opponent_s = torch.chunk(
                batch[actor_channel_opponent.name]["s"], nb_devices_opponent, dim=0
            )
            futures = []
            reply_opponent = {"v": None, "pi": None}
            # multithread
            with ThreadPoolExecutor(max_workers=nb_devices_opponent) as executor:
                for device, model, batch_s in zip(
                    devices_opponent, models_opponent, batches_opponent_s
                ):
                    futures.append(
                        executor.submit(_forward_pass_on_device, device, model, batch_s)
                    )
                results = [future.result() for future in futures]
                reply_opponent["v"] = torch.cat(
                    [result["v"] for result in results], dim=0
                )
                reply_opponent["pi"] = torch.cat(
                    [result["pi"] for result in results], dim=0
                )
            dcm.set_reply(actor_channel_opponent.name, reply_opponent)
    dcm.terminate()


def evaluate_on_checkpoint(
    game_params: GameParams,
    eval_params: EvalParams,
    context: tube.Context,
    actor_channel_eval: Optional[tube.DataChannel],
    actor_channel_opponent: Optional[tube.DataChannel],
    get_eval_reward: Callable[[], List[int]],
    devices_eval: Optional[List[torch.device]],
    models_eval: Optional[List[torch.jit.ScriptModule]],
    pure_mcts_eval: bool,
    devices_opponent: Optional[List[torch.device]],
    models_opponent: Optional[List[torch.jit.ScriptModule]],
    pure_mcts_opponent: bool,
) -> utils.Result:
    if eval_params.eval_verbosity:
        print(f"Playing {eval_params.num_game_eval} games of {game_params.game_name}:")
        print(
            f"- {'pure MCTS' if pure_mcts_eval else type(models_eval[0]).__name__} "
            f"player uses "
            f"{eval_params.num_rollouts_eval} rollouts per actor "
            f"with {eval_params.num_actor_eval} "
            f"actor{'s' if eval_params.num_actor_eval > 1 else ''}"
        )
        print(
            f"- {'pure MCTS' if pure_mcts_opponent else type(models_opponent[0]).__name__} "
            f"opponent uses "
            f"{eval_params.num_rollouts_opponent} rollouts per actor "
            f"with {eval_params.num_actor_opponent} "
            f"actor{'s' if eval_params.num_actor_opponent > 1 else ''}"
        )
    if pure_mcts_eval:
        pass  # not implemented
    else:
        if pure_mcts_opponent:
            _play_game_neural_mcts_against_pure_mcts_opponent(
                context=context,
                actor_channel_eval=actor_channel_eval,
                devices_eval=devices_eval,
                models_eval=models_eval,
            )
        else:
            _play_game_neural_mcts_against_neural_mcts_opponent(
                context=context,
                actor_channel_eval=actor_channel_eval,
                actor_channel_opponent=actor_channel_opponent,
                devices_eval=devices_eval,
                models_eval=models_eval,
                devices_opponent=devices_opponent,
                models_opponent=models_opponent,
            )
    result = utils.Result(get_eval_reward())
    if eval_params.eval_verbosity >= 2:
        print("@@@eval: %s" % result.log())
    return result


#######################################################################################
# OVERALL EVALUATION WORKFLOW
#######################################################################################


def run_evaluation(eval_params: EvalParams, only_last: bool = False) -> None:
    start_time = time.time()
    logger_dir = eval_params.checkpoint_dir
    if eval_params.checkpoint_dir is None:
        logger_dir = os.path.dirname(eval_params.checkpoint)
    logger_path = os.path.join(logger_dir, "eval.log")
    sys.stdout = utils.Logger(logger_path)

    print("#" * 70)
    print("#" + "EVALUATION".center(68) + "#")
    print("#" * 70)

    # evaluation is done on a NN-powered MCTS
    pure_mcts_eval = False

    print("setting-up pseudo-random generator...")
    seed_generator = utils.generate_random_seeds(seed=eval_params.seed_eval)

    if eval_params.plot_enabled:
        print("creating plotter...")
        plotter = create_plotter(eval_params=eval_params)

    print("finding checkpoints...")
    checkpoint_iter = create_checkpoint_iter(
        eval_params=eval_params, only_last=only_last
    )

    models_opponent = []
    pure_mcts_opponent = True
    devices_opponent = None
    game_params_opponent = None
    if eval_params.checkpoint_opponent is not None:
        print("creating opponent model(s) and device(s)...")
        pure_mcts_opponent = False
        (
            models_opponent,
            devices_opponent,
            game_params_opponent,
        ) = create_models_and_devices_opponent(eval_params=eval_params)

    results = []
    first_checkpoint = False
    game_params = None
    for checkpoint in checkpoint_iter:
        epoch = checkpoint.get("epoch", 0)  # 0 when checkpoint_dir is None
        model_state_dict_eval = checkpoint["model_state_dict"]
        model_params_eval = checkpoint["model_params"]
        if game_params is None:
            game_params = checkpoint["game_params"]
            sanitize_game_params(game_params)
        # check that game_params are consistent between the model_eval and
        # the model_opponent
        if game_params_opponent is not None and game_params != game_params_opponent:
            raise ValueError(
                "The game parameters between the model to be tested"
                "and the opponent model are different"
            )
        # check that game_params are consistent from one epoch to the other
        checkpoint_game_params = checkpoint["game_params"]
        sanitize_game_params(checkpoint_game_params)
        if game_params != checkpoint_game_params:
            raise ValueError(f"The game parameters have changed at checkpoint #{epoch}")

        if not first_checkpoint:
            print("creating model(s) and device(s)...")
            devices_eval = [
                torch.device(device_eval) for device_eval in eval_params.device_eval
            ]
            models_eval = []
            for device_eval in devices_eval:
                models_eval.append(
                    create_model(
                        game_params=game_params,
                        model_params=model_params_eval,
                        resume_training=False,
                    ).to(device_eval)
                )
            first_checkpoint = True

        print("updating model(s)...")
        for model_eval in models_eval:
            model_eval.load_state_dict(model_state_dict_eval)
            model_eval.eval()

        num_evaluated_games = 0
        rewards = []

        eval_batch_size = eval_params.num_parallel_games_eval if eval_params.num_parallel_games_eval else eval_params.num_game_eval
        print("evaluating {} games with batches of size {}".format(eval_params.num_game_eval, eval_batch_size))
        while num_evaluated_games < eval_params.num_game_eval:
            if eval_params.eval_verbosity:
                print("creating evaluation environment...")
            current_batch_size = min(eval_batch_size, eval_params.num_game_eval - num_evaluated_games)
            (
                context,
                actor_channel_eval,
                actor_channel_opponent,
                get_eval_reward,
            ) = create_evaluation_environment(
                seed_generator=seed_generator,
                game_params=game_params,
                eval_params=eval_params,
                current_batch_size=current_batch_size,
                pure_mcts_eval=pure_mcts_eval,
                pure_mcts_opponent=pure_mcts_opponent,
                num_evaluated_games=num_evaluated_games,
            )
            if eval_params.eval_verbosity:
                print("evaluating...")
            partial_result = evaluate_on_checkpoint(
                game_params=game_params,
                eval_params=eval_params,
                context=context,
                actor_channel_eval=actor_channel_eval,
                actor_channel_opponent=actor_channel_opponent,
                get_eval_reward=get_eval_reward,
                devices_eval=devices_eval,
                models_eval=models_eval,
                pure_mcts_eval=pure_mcts_eval,
                devices_opponent=devices_opponent,
                models_opponent=models_opponent,
                pure_mcts_opponent=pure_mcts_opponent,
            )
            num_evaluated_games += current_batch_size
            rewards += partial_result.reward
            elapsed_time = time.time() - start_time
            print(f"Evaluated on {num_evaluated_games} games in : {elapsed_time} s")

        result = utils.Result(rewards)
        print("@@@eval: %s" % result.log())
        results.append((epoch, result))

        if eval_params.plot_enabled:
            print("plotting...")
            plotter.plot_results(results)
            plotter.save()

    elapsed_time = time.time() - start_time
    print(f"total time: {elapsed_time} s")
