# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import sys
import warnings
import time
from dataclasses import astuple
import argparse
from multiprocessing import Process
from typing import Union, List

from .params import (
    GameParams,
    ModelParams,
    OptimParams,
    SimulationParams,
    ExecutionParams,
    EvalParams,
)
from .utils import CommandHistory
from .training import run_training
from .evaluation import run_evaluation
from .human import run_human_played_game
from .human import run_tp_played_game
from .convert import convert_checkpoint

DOC = """The python package `pypolygames` can be used in either of the following modes:

- `pypolygames train` (training mode): a game and a model (as well as several other options, see below) are chosen and the model is iteratively trained with MCTS
- `pypolygames eval` (evaluation mode): the model confronts either a pure MCTS or another neural network powered MCTS. The evaluation of a training can be done either offline (from checkpoints periodically saved) or in real time; in that case, the evaluation considers only the most recent checkpoint in order to follow closely the training, skipping some checkpoints in case the eval computation takes longer than the time becween consecutive checkpoints. It is displayed through visdom.
- `pypolygames traineval` (training + evaluation mode): it mixes the two previous modes and allow to launch one command instead of two. With the `real_time` option the modes can be launched in parallel instead of sequentially.
- `pypolygames human` (human mode): a human player plays against the machine

Trainings log the following relevant files in the `checkpoint_dir`:
- `model.pt`
- `train.log`
- `stat.tb`
- `checkpoints_<epoch>.pt` for for checkpoints saved each `saving_period` epoch (e.g., if `saving_period == 10`, `checkpoints_0.pt`, `checkpoints_9.pt`, `checkpoints_19.pt`, `checkpoints_29.pt`)

By default, the checkpoint_dir is exps/dev/game_<game_name>_model_<model_name>_feat_<featurization>_GMT_<YYYYMMDDHHMMSS>

This directory will be the `checkpoint_dir` directory used by evaluation to retrieve the checkpoints to perform eval computation."""


def _check_arg_consistency(args: argparse.Namespace) -> None:
    # Most of the consistency is done in the `__post_init__` methods in the params class
    if (
        args.command_history.last_command_contains("pure_mcts")
        and getattr(args, "game_name", None) is None
    ):
        raise ValueError(
            "In '--pure_mcts' the game must be specified with '--game_name'"
        )
    if args.command_history.last_command_contains("human"):
        if (
            getattr(args, "pure_mcts", None) is False
            and getattr(args, "init_checkpoint", None) is None
        ):
            raise ValueError(
                "The human player need to play either a '--pure_mcts' "
                "or a '--init_checkpoint' neural network powered MCTS"
            )
    if args.command_history.last_command_contains("device_opponent"):
        if getattr(args, "checkpoint_opponent", None) is None:
            raise ValueError(
                "If the opponent is a pure MCTS player "
                "('--checkpoint_opponent' not set), "
                "all its computation will happen on CPU, "
                "'--device_opponent' should not be set"
            )
    if args.command_history.last_command_contains(
        "per_thread_batchsize"
    ) and args.command_history.last_command_contains("act_batchsize"):
        raise ValueError(
            "When '--per_thread_batchsize' is set, '--act_batchsize' is not used"
        )

    if getattr(args, "total_time", 0) is not None and getattr(args, "total_time", 0) > 0:
        if args.command_history.last_command_contains("num_rollouts"):
            raise ValueError(
                "When a '--total_time' is set, "
                "the '--num_rollouts' will adapt automatically and should not be set"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=DOC, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False,
    )
    parser.set_defaults(func=run_training_and_evaluation_from_args_warning)

    subparsers = parser.add_subparsers(
        help="Modes to be chosen from: `python -m pypolygames MODE`"
    )

    # TRAINING
    parser_train = subparsers.add_parser("train")
    parser_train.set_defaults(func=run_training_from_args)

    # EVALUATION
    parser_eval = subparsers.add_parser("eval")
    parser_eval.set_defaults(func=run_evaluation_from_args)

    # TRAINING + EVALUATION
    parser_traineval = subparsers.add_parser("traineval")
    parser_traineval.set_defaults(func=run_training_and_evaluation_from_args)

    # HUMAN-PLAYED GAME
    parser_human = subparsers.add_parser("human")
    parser_human.set_defaults(func=run_human_played_game_from_args)

    # TEXT-PROTOCOLE GAME
    parser_tp = subparsers.add_parser("tp")
    parser_tp.set_defaults(func=run_tp_played_game_from_args)

    # CONVERT CHECKPOINT COMMAND
    parser_convert = subparsers.add_parser("convert")
    parser_convert.set_defaults(func=convert_checkpoint_from_args)

    parser_convert.add_argument('--out', type=str, required=True, help='File name to save the converted checkpoint to')
    parser_convert.add_argument('--skip', type=str, nargs="*", help='List of attributes to not copy, leaving them initialized')

    # Game params
    train_game_params_group = parser_train.add_argument_group(
        "Game parameters",
        "Not to be specified in case of loading a checkpoint or a pretrained model",
    )
    traineval_game_params_group = parser_traineval.add_argument_group(
        "Game parameters",
        "Not to be specified in case of loading a checkpoint or a pretrained model",
    )
    game_params_group = parser.add_argument_group(
        "Game parameters",
        "Not to be specified in case of loading a checkpoint or a pretrained model",
    )
    human_game_params_group = parser_human.add_argument_group(
        "Game parameters",
        "Mandatory for pure MTCS, "
        "but not to be specified in case of loading a pretrained model",
    )
    for arg_name, arg_field in GameParams.arg_fields():
        train_game_params_group.add_argument(arg_field.name, **arg_field.opts)
        traineval_game_params_group.add_argument(arg_field.name, **arg_field.opts)
        game_params_group.add_argument(
            arg_field.name, **{**arg_field.opts, **dict(help=argparse.SUPPRESS)}
        )
        human_game_params_group.add_argument(arg_field.name, **arg_field.opts)
        parser_convert.add_argument(arg_field.name, **arg_field.opts)

    # Model params
    train_model_params_group = parser_train.add_argument_group(
        "Model parameters",
        "Not to be specified in case of loading a checkpoint or a pretrained model",
    )
    traineval_model_params_group = parser_traineval.add_argument_group(
        "Model parameters",
        "Not to be specified in case of loading a checkpoint or a pretrained model",
    )
    model_params_group = parser.add_argument_group("Model parameters")
    human_model_params_group = parser_human.add_argument_group(
        "Model parameters",
        "The machine model can be either a '--pure_mcts' or "
        "a '--init_checkpoint' neural network powered MCTS",
    )
    for arg_name, arg_field in ModelParams.arg_fields():
        if arg_name != "pure_mcts":
            train_model_params_group.add_argument(arg_field.name, **arg_field.opts)
            traineval_model_params_group.add_argument(arg_field.name, **arg_field.opts)
            model_params_group.add_argument(
                arg_field.name, **{**arg_field.opts, **dict(help=argparse.SUPPRESS)}
            )
        if arg_name in {"pure_mcts", "init_checkpoint"}:
            human_model_params_group.add_argument(arg_field.name, **arg_field.opts)
        if arg_name != "pure_mcts":
            parser_convert.add_argument(arg_field.name, **arg_field.opts)

    # Optimizer params
    train_optim_params_group = parser_train.add_argument_group("Optimizer parameters")
    traineval_optim_params_group = parser_traineval.add_argument_group(
        "Optimizer parameters"
    )
    optim_params_group = parser.add_argument_group("Optimizer parameters")
    for _, arg_field in OptimParams.arg_fields():
        train_optim_params_group.add_argument(arg_field.name, **arg_field.opts)
        traineval_optim_params_group.add_argument(arg_field.name, **arg_field.opts)
        optim_params_group.add_argument(
            arg_field.name, **{**arg_field.opts, **dict(help=argparse.SUPPRESS)}
        )

    # Simulation params
    train_simulation_params_group = parser_train.add_argument_group(
        "Simulation parameters"
    )
    traineval_simulation_params_group = parser_traineval.add_argument_group(
        "Simulation parameters"
    )
    simulation_params_group = parser.add_argument_group("Simulation parameters")
    human_simulation_params_group = parser_human.add_argument_group(
        "Simulation parameters"
    )
    for arg_name, arg_field in SimulationParams.arg_fields():
        if arg_name not in {
            "human_first",
            "time_ratio",
            "total_time",
        }:  # , "num_actor"}:
            train_simulation_params_group.add_argument(arg_field.name, **arg_field.opts)
            traineval_simulation_params_group.add_argument(
                arg_field.name, **arg_field.opts
            )
            simulation_params_group.add_argument(
                arg_field.name, **{**arg_field.opts, **dict(help=argparse.SUPPRESS)}
            )
        if arg_name in {"num_actor", "num_rollouts"}:
            human_simulation_params_group.add_argument(arg_field.name, **arg_field.opts)

    # Execution params
    train_execution_params_group = parser_train.add_argument_group(
        "Execution parameters"
    )
    traineval_execution_params_group = parser_traineval.add_argument_group(
        "Execution parameters"
    )
    human_execution_params_group = parser_human.add_argument_group(
        "Execution parameters"
    )
    execution_params_group = parser.add_argument_group("Execution parameters")
    for arg_name, arg_field in ExecutionParams.arg_fields():
        if arg_name not in {"human_first", "time_ratio", "total_time"}:
            train_execution_params_group.add_argument(arg_field.name, **arg_field.opts)
            traineval_execution_params_group.add_argument(
                arg_field.name, **arg_field.opts
            )
            execution_params_group.add_argument(
                arg_field.name, **{**arg_field.opts, **dict(help=argparse.SUPPRESS)}
            )
        if arg_name in {"human_first", "time_ratio", "total_time", "device", "seed"}:
            human_execution_params_group.add_argument(arg_field.name, **arg_field.opts)

    # Evaluation params
    eval_eval_params_group = parser_eval.add_argument_group("Evaluation parameters")
    traineval_eval_params_group = parser_traineval.add_argument_group(
        "Evaluation parameters"
    )
    eval_params_group = parser.add_argument_group("Evaluation parameters")
    for arg_name, arg_field in EvalParams.arg_fields():
        eval_eval_params_group.add_argument(arg_field.name, **arg_field.opts)
        if arg_name not in {"checkpoint_dir", "checkpoint"}:
            traineval_eval_params_group.add_argument(arg_field.name, **arg_field.opts)
            eval_params_group.add_argument(
                arg_field.name, **{**arg_field.opts, **dict(help=argparse.SUPPRESS)}
            )

    args = parser.parse_args()
    args.command_history = CommandHistory()

    # check arg consistency
    _check_arg_consistency(args)
    return args


def _get_game_features(game_params: GameParams) -> str:
    return "_".join(str(x) for x in astuple(game_params))


def _get_timestamp() -> str:
    return time.strftime("%Y%m%d%H%M%S", time.gmtime())


def update_and_create_checkpoint_dir(
    game_params: GameParams,
    model_params: ModelParams,
    execution_params: ExecutionParams,
) -> None:
    # create a dedicated folder if none is provided
    if execution_params.checkpoint_dir is None:
        game_name = game_params.game_name
        model_name = model_params.model_name
        game_features = _get_game_features(game_params)
        timestamp = _get_timestamp()
        subfolder = f"game_{game_name}_model_{model_name}_feat_{game_features}_GMT_{timestamp}"
        execution_params.checkpoint_dir = Path("exps").absolute() / "dev" / subfolder
    execution_params.checkpoint_dir.mkdir(exist_ok=True, parents=True)


def instanciate_params_from_args(
    Dataclass, args: argparse.Namespace
) -> Union[
    GameParams, ModelParams, OptimParams, SimulationParams, ExecutionParams, EvalParams
]:
    return Dataclass(
        **{param: getattr(args, param, None) for param, _ in Dataclass.arg_fields()}
    )


def run_training_from_args(args: argparse.Namespace):
    command_history = args.command_history
    game_params = instanciate_params_from_args(GameParams, args)
    model_params = instanciate_params_from_args(ModelParams, args)
    optim_params = instanciate_params_from_args(OptimParams, args)
    simulation_params = instanciate_params_from_args(SimulationParams, args)
    execution_params = instanciate_params_from_args(ExecutionParams, args)

    update_and_create_checkpoint_dir(
        game_params=game_params,
        model_params=model_params,
        execution_params=execution_params,
    )
    run_training(
        command_history=command_history,
        game_params=game_params,
        model_params=model_params,
        optim_params=optim_params,
        simulation_params=simulation_params,
        execution_params=execution_params,
    )


def run_evaluation_from_args(args: argparse.Namespace):
    eval_params = instanciate_params_from_args(EvalParams, args)
    run_evaluation(eval_params=eval_params)


def run_training_and_evaluation_from_args(args: argparse.Namespace):
    command_history = args.command_history
    game_params = instanciate_params_from_args(GameParams, args)
    model_params = instanciate_params_from_args(ModelParams, args)
    optim_params = instanciate_params_from_args(OptimParams, args)
    simulation_params = instanciate_params_from_args(SimulationParams, args)
    execution_params = instanciate_params_from_args(ExecutionParams, args)
    # create the save dir
    update_and_create_checkpoint_dir(
        game_params=game_params,
        model_params=model_params,
        execution_params=execution_params,
    )
    args.checkpoint_dir = execution_params.checkpoint_dir
    eval_params = instanciate_params_from_args(EvalParams, args)
    if args.real_time:
        eval_process = Process(target=run_evaluation, args=(eval_params,))
        eval_process.start()
        run_training(
            command_history=command_history,
            game_params=game_params,
            model_params=model_params,
            optim_params=optim_params,
            simulation_params=simulation_params,
            execution_params=execution_params,
        )
        eval_process.join()
    else:
        run_training(
            command_history=command_history,
            game_params=game_params,
            model_params=model_params,
            optim_params=optim_params,
            simulation_params=simulation_params,
            execution_params=execution_params,
        )
        run_evaluation(eval_params=eval_params, only_last=True)


def run_human_played_game_from_args(args: argparse.Namespace):
    game_params = instanciate_params_from_args(GameParams, args)
    model_params = instanciate_params_from_args(ModelParams, args)
    simulation_params = instanciate_params_from_args(SimulationParams, args)
    simulation_params.num_game = 1
    execution_params = instanciate_params_from_args(ExecutionParams, args)
    run_human_played_game(
        game_params=game_params,
        model_params=model_params,
        simulation_params=simulation_params,
        execution_params=execution_params,
    )


def run_tp_played_game_from_args(args: argparse.Namespace):
    game_params = instanciate_params_from_args(GameParams, args)
    model_params = instanciate_params_from_args(ModelParams, args)
    simulation_params = instanciate_params_from_args(SimulationParams, args)
    simulation_params.num_game = 1
    execution_params = instanciate_params_from_args(ExecutionParams, args)
    run_tp_played_game(
        game_params=game_params,
        model_params=model_params,
        simulation_params=simulation_params,
        execution_params=execution_params,
    )

def convert_checkpoint_from_args(args: argparse.Namespace):
    command_history = args.command_history
    game_params = instanciate_params_from_args(GameParams, args)
    model_params = instanciate_params_from_args(ModelParams, args)
    convert_checkpoint(
        command_history=command_history,
        game_params=game_params,
        model_params=model_params,
        out=args.out,
        skip=args.skip
    )


def run_training_and_evaluation_from_args_warning(args: argparse.Namespace):
    # pypolygames called directly
    if len(sys.argv) == 1:
        print(DOC)
    # otherwise default to traineval
    else:
        warnings.warn(
            "'pypolygames' called with arguments runs as 'pypolygames traineval'",
            DeprecationWarning,
        )
        run_training_and_evaluation_from_args(args)


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
