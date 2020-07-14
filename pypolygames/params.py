# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Iterator, Tuple, Union, List, Optional, Dict, Any

from .weight_init import WEIGHT_INIT


@dataclass
class ArgFields:
    name: Optional[str] = None
    opts: Optional[Dict[str, Any]] = None


@dataclass
class GameParams:
    game_name: Optional[str] = None
    out_features: bool = False
    turn_features: bool = False
    geometric_features: bool = False
    random_features: int = 0
    one_feature: bool = False
    history: int = 0
    predict_end_state: bool = False
    predict_n_states: int = 0

    def __setattr__(self, attr, value):
        if value is None:
            value = getattr(self, attr)
        super().__setattr__(attr, value)

    def __eq__(self, other_game_params):
        return all(
            getattr(self, field) == getattr(other_game_params, field)
            for field in {
                "game_name",
                "out_features",
                "turn_features",
                "geometric_features",
                "one_feature",
                "history",
            }
        )

    @classmethod
    def arg_fields(cls) -> Iterator[Tuple[str, ArgFields]]:
        params = OrderedDict(
            game_name=ArgFields(
                opts=dict(
                    type=str,
                    help="Game name - if left unspecified it will default to the game "
                    "that the model selected with '--model_name' refers to as default",
                )
            ),
            out_features=ArgFields(
                opts=dict(
                    action="store_false" if cls.out_features else "store_true",
                    help="If set, the input to the NN includes a channel "
                    "with 1 on the frontier",
                )
            ),
            turn_features=ArgFields(
                opts=dict(
                    action="store_false" if cls.turn_features else "store_true",
                    help="If set, the input to the NN includes a channel "
                    "with the player index broadcasted",
                )
            ),
            geometric_features=ArgFields(
                opts=dict(
                    action="store_false" if cls.geometric_features else "store_true",
                    help="If set, the input to the NN includes "
                    "4 geometric channels representing the position on the board",
                )
            ),
            random_features=ArgFields(
                opts=dict(type=int, help="Number of random features the input includes")
            ),
            one_feature=ArgFields(
                opts=dict(
                    action="store_false" if cls.one_feature else "store_true",
                    help="If set, the input to the NN includes "
                    "a channel with 1 everywhere",
                )
            ),
            history=ArgFields(
                opts=dict(
                    type=int,
                    help="Number of last steps whose representation is "
                    "added in the featurization",
                )
            ),
            predict_end_state=ArgFields(
                opts=dict(
                    type=bool,
                    help="Side learning: predict end state (not functional for training yet)",
                )
            ),
            predict_n_states=ArgFields(
                opts=dict(
                    type=int,
                    help="Side learning: predict N next game states (not functional for training yet)",
                )
            ),
        )
        for param, arg_field in params.items():
            if arg_field.name is None:
                arg_field.name = f"--{param}"
            if arg_field.opts is None:
                arg_field.opts = {}
            if "help" not in arg_field.opts:
                arg_field.opts["help"] = ""
            arg_field.opts["help"] += f" (DEFAULT: {getattr(cls(), param)})"
            yield param, arg_field


@dataclass
class ModelParams:
    """Model parameters - all set to 'None' as they have sensible default
    specified in their definitions"""

    init_checkpoint: Path = None
    pure_mcts: bool = False
    model_name: str = None
    nb_nets: int = None
    nb_layers_per_net: int = None
    nnsize: float = None
    fcsize: int = None
    nnks: int = None
    pooling: bool = False
    bn: bool = False
    # bn_affine: bool = False
    init_method: str = next(iter(WEIGHT_INIT))
    activation_function: str = "relu"
    global_pooling: float = 0
    batchnorm_momentum: float = 0.01

    def __setattr__(self, attr, value):
        if value is None:
            value = getattr(self, attr)
        super().__setattr__(attr, value)

    def __post_init__(self):
        if self.init_checkpoint is not None:
            if self.pure_mcts:
                raise ValueError(
                    "The MCTS can be either assisted with a "
                    "'--init_checkpoint' neural network or be a '--pure_mcts'"
                )
            self.init_checkpoint = self.init_checkpoint.absolute()
        # if self.bn and self.bn_affine:
        #     raise ValueError(
        #         "At most one of the options '--bn' and '--bn_affine' can be selected"
        #     )

    @classmethod
    def arg_fields(cls) -> Iterator[Tuple[str, ArgFields]]:
        params = OrderedDict(
            init_checkpoint=ArgFields(
                opts=dict(
                    type=Path,
                    help="Path to pretrained model (a checkpoint), in case of a "
                    "simulation or for fine-tuning - if specified the game parameters "
                    "and model parameters should be left unspecified as the pretrained "
                    "model contains them",
                )
            ),
            pure_mcts=ArgFields(
                opts=dict(
                    action="store_false" if cls.pure_mcts else "store_true",
                    help="If set, the inference will be done with MCTS only "
                    "- no Neural Network",
                )
            ),
            model_name=ArgFields(
                opts=dict(
                    type=str,
                    help="Model name - if left unspecified "
                    "it will default to a generic model",
                )
            ),
            nb_nets=ArgFields(
                opts=dict(type=int, help="Number of subnets, when applicable")
            ),
            nb_layers_per_net=ArgFields(
                opts=dict(type=int, help="Number of layer per subnets, when applicable")
            ),
            nnsize=ArgFields(
                opts=dict(
                    type=float, help="Number of units per hidden layer, when applicable"
                )
            ),
            fcsize=ArgFields(
                opts=dict(
                    type=int,
                    help="Size of final full-connected layers, when applicable",
                )
            ),
            nnks=ArgFields(
                opts=dict(
                    type=int,
                    help="Kernel size for convolutional layers, when applicable "
                    "- dilation and stride are set to one, "
                    "so it must be an even number",
                )
            ),
            pooling=ArgFields(
                opts=dict(
                    action="store_false" if cls.pooling else "store_true",
                    help="If set, adds pooling layers following convolutional layers, "
                    "when applicable",
                )
            ),
            bn=ArgFields(
                opts=dict(
                    action="store_false" if cls.bn else "store_true",
                    # help="If set, adds batch normalisation with "
                    # "no learnable affine parameters",
                    help="If set, adds batch normalisation with "
                    "learnable affine parameters",
                )
            ),
            # bn_affine=ArgFields(
            #     opts=dict(
            #         action="store_false" if cls.bn_affine else "store_true",
            #         help="If set, adds batch normalisation with "
            #         "learnable affine parameters",
            #     )
            # ),
            init_method=ArgFields(
                opts=dict(
                    type=str,
                    help="Weight initialisation method",
                    choices=list(WEIGHT_INIT),
                )
            ),
            activation_function=ArgFields(
                opts=dict(
                    type=str,
                    help="Activation function to use",
                )
            ),
            global_pooling=ArgFields(
                opts=dict(
                    type=float,
                    help="Global pooling - this will, for the models that support it, "
                    "add global pooling over some channels after convolutional layers. "
                    "The parameter is the proportion of the channels that should be pooled. "
                    "Eg. 0.1 will specify that we should pool 10% of the channels"
                )
            ),
            batchnorm_momentum=ArgFields(
                opts=dict(
                    type=float,
                    help="Batch normalization momentum",
                )
            ),
        )
        for param, arg_field in params.items():
            if arg_field.name is None:
                arg_field.name = f"--{param}"
            if arg_field.opts is None:
                arg_field.opts = {}
            if "help" not in arg_field.opts:
                arg_field.opts["help"] = ""
            arg_field.opts["help"] += f" (DEFAULT: {getattr(cls(), param)})"
            yield param, arg_field


@dataclass
class OptimParams:
    num_epoch: int = 10_000_000  # basically infinity
    epoch_len: int = 1000
    batchsize: int = 128
    lr: float = 1e-3
    eps: float = 1.5e-4
    grad_clip: float = 0.25

    def __setattr__(self, attr, value):
        if value is None:
            value = getattr(self, attr)
        super().__setattr__(attr, value)

    @classmethod
    def arg_fields(cls) -> Iterator[Tuple[str, ArgFields]]:
        params = OrderedDict(
            num_epoch=ArgFields(opts=dict(type=int, help=f"Number of epochs")),
            epoch_len=ArgFields(
                opts=dict(type=int, help=f"Number of train batches per epoch")
            ),
            batchsize=ArgFields(
                opts=dict(
                    type=int,
                    help="Number of training examples in a mini-batch (train batch)"
                    "- also the batchsize in GPU (when enabled) for training",
                )
            ),
            lr=ArgFields(opts=dict(type=float, default=cls.lr, help=f"Learning rate")),
            eps=ArgFields(
                opts=dict(
                    type=float,
                    help="Term added to the denominator to improve "
                    "numerical stability ",
                )
            ),
            grad_clip=ArgFields(
                opts=dict(type=float, help=f"Max norm of the gradients")
            ),
        )
        for param, arg_field in params.items():
            if arg_field.name is None:
                arg_field.name = f"--{param}"
            if arg_field.opts is None:
                arg_field.opts = {}
            if "help" not in arg_field.opts:
                arg_field.opts["help"] = ""
            arg_field.opts["help"] += f" (DEFAULT: {getattr(cls(), param)})"
            yield param, arg_field


@dataclass
class SimulationParams:
    num_game: int = 10  # number of cores
    num_actor: int = 1  # should be 1 at training time
    num_rollouts: int = 1600
    replay_capacity: int = 1_000_000
    replay_warmup: int = 10_000
    sync_period: int = 100
    act_batchsize: int = 1
    per_thread_batchsize: int = 0
    train_channel_timeout_ms: int = 1
    train_channel_num_slots: int = 1000

    def __setattr__(self, attr, value):
        if value is None:
            value = getattr(self, attr)
        super().__setattr__(attr, value)

    def __post_init__(self) -> None:
        if self.per_thread_batchsize == 0 and self.act_batchsize > self.num_game:
            raise ValueError("'act_batchsize' cannot be larger than 'num_games'")

    @classmethod
    def arg_fields(cls) -> Iterator[Tuple[str, ArgFields]]:
        params = OrderedDict(
            num_game=ArgFields(
                opts=dict(type=int, help=f"Number of game-running threads")
            ),
            num_actor=ArgFields(
                opts=dict(
                    type=int,
                    help=f"Number of actors per non-human player, "
                    "one actor being one thread doing MCTS "
                    "- the more num_actor, the larger the MCTS",
                )
            ),
            num_rollouts=ArgFields(
                opts=dict(type=int, help="Number of rollouts per actor/thread")
            ),
            replay_capacity=ArgFields(
                opts=dict(
                    type=int, help="Nb of act_batches the replay buffer can contain"
                )
            ),
            replay_warmup=ArgFields(
                opts=dict(
                    type=int,
                    help="Nb of act_batches the replay buffer needs to buffer "
                    "before the training can start",
                )
            ),
            sync_period=ArgFields(
                opts=dict(
                    type=int,
                    help="Number of epochs between two consecutive sync "
                    "between the model and the assembler",
                )
            ),
            act_batchsize=ArgFields(
                opts=dict(
                    type=int,
                    help="When '--per_thread_batchsize' is not set, "
                    "number or requests batched together for inference",
                )
            ),
            per_thread_batchsize=ArgFields(
                opts=dict(
                    type=int,
                    help="When non-zero, "
                    "number of games in each game-running threads run sequentially "
                    "and batched together for inference (see '--act_batchsize')",
                )
            ),
            train_channel_timeout_ms=ArgFields(
                opts=dict(
                    type=int,
                    help="Timeout (in milliseconds) to wait for actors to produce "
                    "trajectories",
                )
            ),
            train_channel_num_slots=ArgFields(
                opts=dict(
                    type=int,
                    help="Number of slots in train channel used to send trajectories",
                )
            ),
        )
        for param, arg_field in params.items():
            if arg_field.name is None:
                arg_field.name = f"--{param}"
            if arg_field.opts is None:
                arg_field.opts = {}
            if "help" not in arg_field.opts:
                arg_field.opts["help"] = ""
            arg_field.opts["help"] += f" (DEFAULT: {getattr(cls(), param)})"
            yield param, arg_field


@dataclass
class ExecutionParams:
    checkpoint_dir: Path = None
    save_dir: str = None  # keep for deprecation warning
    save_uncompressed: bool = False
    do_not_save_replay_buffer: bool = False
    saving_period: int = 100
    max_time: Optional[int] = None
    human_first: bool = False
    time_ratio: float = 0.07
    total_time: float = 0
    device: List[str] = field(default_factory=lambda: ["cuda:0"])
    seed: int = 1
    ddp: bool = False
    server_listen_endpoint: str = ""
    server_connect_hostname: str = ""
    opponent_model_path: Path = None

    def __setattr__(self, attr, value):
        if value is None:
            try:
                value = getattr(self, attr)
            except AttributeError:
                value = getattr(type(self)(), attr)
        super().__setattr__(attr, value)

    def __post_init__(self) -> None:
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = self.checkpoint_dir.resolve().absolute()
        if self.save_dir is not None:
            raise RuntimeError("""--save_dir is deprecated, use --checkpoint_dir instead, with slightly different behavior:
    - no subfolder creation.
    - resumes from latest checkpoint if available in the directory.""")

    @classmethod
    def arg_fields(cls) -> Iterator[Tuple[str, ArgFields]]:
        params = OrderedDict(
            checkpoint_dir=ArgFields(
                opts=dict(
                    type=Path,
                    help="Directory for saving checkpoints. "
                         "If the directory is not empty, the latest checkpoint will be resumed",
                )
            ),
            save_dir=ArgFields(
                opts=dict(
                    type=Path,
                    help="Deprecated, use checkpoint_dir with slightly different behavior instead"
                )
            ),
            save_uncompressed=ArgFields(
                opts=dict(
                    action="store_false" if cls.save_uncompressed else "store_true",
                    help="If set, saved checkpoints will be saved uncompressed",
                )
            ),
            do_not_save_replay_buffer=ArgFields(
                opts=dict(
                    action="store_false"
                    if cls.do_not_save_replay_buffer
                    else "store_true",
                    help="If set, the replay buffer will be not saved "
                    "in the checkpoint",
                )
            ),
            saving_period=ArgFields(
                opts=dict(
                    type=int,
                    help="Number of epochs between two consecutive checkpoints",
                )
            ),
            max_time=ArgFields(
                opts=dict(type=int, help="Maximum time allowed for a run (in seconds)")
            ),
            human_first=ArgFields(
                opts=dict(
                    action="store_false" if cls.human_first else "store_true",
                    help="If set in a two-player game, " "the human player plays first",
                )
            ),
            time_ratio=ArgFields(
                opts=dict(
                    type=float, help="Part of the remaining time for the next move"
                )
            ),
            total_time=ArgFields(
                opts=dict(
                    type=float,
                    help="Total time in seconds for the entire game for one player",
                )
            ),
            device=ArgFields(
                opts=dict(
                    type=str,
                    nargs="*",
                    help="List of torch devices where the computation for the model"
                    "will happen "
                    '(e.g., "cpu", "cuda:0") '
                    "- in training mode, if there are several devices in the list, "
                    " the first device will be the one used for training "
                    "while the others will be used generating games",
                )
            ),
            seed=ArgFields(
                opts=dict(type=int, help="Seed for pseudo-random number generator")
            ),
            ddp=ArgFields(
                opts=dict(
                    type=bool,
                    help="Use DistributedDataParallel for training (multi GPU)",
                )
            ),
            server_listen_endpoint=ArgFields(
                opts=dict(
                    type=str,
                    help="Listen for distributed training, eg. tcp://0.0.0.0:5611",
                )
            ),
            server_connect_hostname=ArgFields(
                opts=dict(
                    type=str,
                    help="Connect to hostname for distributed training, eg. tcp://127.0.0.1:5611",
                )
            ),
            opponent_model_path=ArgFields(
                opts=dict(
                    type=Path,
                    help="Load this model as the opponent - will not request model updates",
                )
            ),
        )
        for param, arg_field in params.items():
            if arg_field.name is None:
                arg_field.name = f"--{param}"
            if arg_field.opts is None:
                arg_field.opts = {}
            if "help" not in arg_field.opts:
                arg_field.opts["help"] = ""
            arg_field.opts["help"] += f" (DEFAULT: {getattr(cls(), param)})"
            yield param, arg_field


@dataclass
class EvalParams:
    real_time: bool = False
    checkpoint_dir: Path = None
    checkpoint: Path = None
    device_eval: List[str] = field(default_factory=lambda: ["cuda:1"])
    num_game_eval: int = 100
    num_parallel_games_eval: int = None
    num_actor_eval: int = 1
    num_rollouts_eval: int = 400
    checkpoint_opponent: Path = None
    device_opponent: List[str] = field(default_factory=lambda: ["cuda:0"])
    num_actor_opponent: int = 1
    num_rollouts_opponent: int = 2000
    seed_eval: int = 2
    plot_enabled: bool = False
    plot_server: str = "http://localhost"
    plot_port: int = 8097
    eval_verbosity: int = 1

    def __setattr__(self, attr, value):
        if value is None:
            try:
                value = getattr(self, attr)
            except AttributeError:
                # TODO: this part may be buggy (infinite recursion), is it needed?
                # cannot create with both None for checkpoint_dir and checkpoint
                defaults = self.__class__(checkpoint_dir=Path("blublu"))
                if attr != "checkpoint_dir":
                    value = getattr(defaults, attr)
        super().__setattr__(attr, value)

    def __post_init__(self) -> None:
        if self.real_time and self.checkpoint is not None:
            raise ValueError(
                "In '--real_time' the evaluation follow the training "
                "so '--checkpoint' should not be set"
            )
        if self.checkpoint_dir is None and self.checkpoint is None:
            raise ValueError(
                "Either a '--checkpoint_dir' or a path to a '--checkpoint' "
                "must be specified"
            )
        if self.checkpoint_dir is not None and self.checkpoint is not None:
            raise ValueError(
                "Either a '--checkpoint_dir' or a path to a '--checkpoint' "
                "must be specified, but not both"
            )
        if self.checkpoint is not None and self.plot_enabled:
            raise ValueError(
                "Plotting is not available if the evaluation is performed "
                "only on one checkpoint"
            )
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = self.checkpoint_dir.absolute()
        if self.checkpoint is not None:
            self.checkpoint = self.checkpoint.absolute()

    @classmethod
    def arg_fields(cls) -> Iterator[Tuple[str, ArgFields]]:
        params = OrderedDict(
            real_time=ArgFields(
                opts=dict(
                    action="store_false" if cls.real_time else "store_true",
                    help="In 'real_time' the evaluation follows the training "
                    "as it goes, "
                    "taking the last available checkpoints and "
                    "skipping some previous checkpoints "
                    "if they are taking too much time to compute",
                )
            ),
            checkpoint_dir=ArgFields(
                opts=dict(
                    type=Path,
                    help="Directory storing the checkpoints "
                    "- if set, '--checkpoint' should not be set",
                )
            ),
            checkpoint=ArgFields(
                opts=dict(
                    type=Path,
                    help="Path to the individual checkpoint to be evaluated "
                    "- if set, '--checkpoint_dir' should not be set",
                )
            ),
            device_eval=ArgFields(
                opts=dict(
                    type=str,
                    nargs="*",
                    help="List of torch devices where the computation for the model"
                    "to be tested will happen "
                    '(e.g., "cpu", "cuda:0")',
                )
            ),
            num_game_eval=ArgFields(
                opts=dict(
                    type=int, help="Number of games played against a pure MCTS opponent"
                )
            ),
            num_parallel_games_eval=ArgFields(
                opts=dict(
                    type=int, help="Number of evaluation games to be played in parallel. "
                                   "If set to None, all games are played in parallel"
                )
            ),
            num_actor_eval=ArgFields(
                opts=dict(
                    type=int,
                    help="Number of actors per player for the model to be tested, "
                    "one actor being one thread doing MCTS "
                    "- the more num_actor_eval, the larger the MCTS "
                    "- when the model plays against another model as opponent, "
                    "it needs to be set to a number > 1",
                )
            ),
            num_rollouts_eval=ArgFields(
                opts=dict(
                    type=int,
                    help="Number of rollouts per actor/thread for "
                    "the model to be tested",
                )
            ),
            checkpoint_opponent=ArgFields(
                opts=dict(
                    type=Path,
                    help="Path to the checkpoint the opponent will use as model"
                    " - if not set, the opponent will be a pure MCTS",
                )
            ),
            device_opponent=ArgFields(
                opts=dict(
                    type=str,
                    nargs="*",
                    help="List of torch devices where the computation for the opponent"
                    "will happen "
                    '(e.g., "cpu", "cuda:0")',
                )
            ),
            num_actor_opponent=ArgFields(
                opts=dict(
                    type=int,
                    help="Number of actors per player for the pure MCTS opponent, "
                    "one actor being one thread doing MCTS "
                    "- the more num_actor_opponent, the larger the MCTS "
                    "- when a NN as been chosen as an opponent "
                    "(see '--checkpoint_opponent'),"
                    "it needs to be set to a number > 1",
                )
            ),
            num_rollouts_opponent=ArgFields(
                opts=dict(
                    type=int,
                    help="Number of rollouts per actor/thread for the opponent",
                )
            ),
            seed_eval=ArgFields(
                opts=dict(type=int, help="Seed for pseudo-random number generator")
            ),
            plot_enabled=ArgFields(
                opts=dict(
                    action="store_false" if cls.plot_enabled else "store_true",
                    help="If set, visdom plots the evaluation as it is computed",
                )
            ),
            plot_server=ArgFields(opts=dict(type=str, help="Visdom server url")),
            plot_port=ArgFields(opts=dict(type=int, help="Visdom server port")),
            eval_verbosity=ArgFields(opts=dict(type=int, help="Verbosity during the evaluation")),
        )
        defaults = cls(checkpoint_dir=Path("blublu"))  # cannot create with both None for checkpoint_dir and checkpoint
        defaults.checkpoint_dir = None  # revert
        for param, arg_field in params.items():
            if arg_field.name is None:
                arg_field.name = f"--{param}"
            if arg_field.opts is None:
                arg_field.opts = {}
            if "help" not in arg_field.opts:
                arg_field.opts["help"] = ""
            arg_field.opts[
                "help"
            ] += f" (DEFAULT: {getattr(defaults, param)})"
            yield param, arg_field


GenericParams = Union[
    GameParams, ModelParams, OptimParams, SimulationParams, ExecutionParams
]
