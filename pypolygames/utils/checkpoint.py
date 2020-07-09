# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import time
import gzip
import zipfile
import re
from pathlib import Path
from typing import Iterator, Dict, Union, Any

import torch

import tube

from .command_history import CommandHistory
from ..params import (
    GameParams,
    ModelParams,
    OptimParams,
    SimulationParams,
    ExecutionParams,
)

Checkpoint = Dict[
    str,
    Union[
        int,
        bytes,
        Dict[str, Any],
        GameParams,
        ModelParams,
        OptimParams,
        SimulationParams,
        ExecutionParams,
    ],
]


def save_checkpoint(
    command_history: CommandHistory,
    epoch: int,
    model: torch.jit.ScriptModule,
    optim: torch.optim.Optimizer,
    assembler: tube.ChannelAssembler,
    game_params: GameParams,
    model_params: ModelParams,
    optim_params: OptimParams,
    simulation_params: SimulationParams,
    execution_params: ExecutionParams,
) -> None:
    checkpoint_dir = execution_params.checkpoint_dir
    save_uncompressed = execution_params.save_uncompressed
    do_not_save_replay_buffer = execution_params.do_not_save_replay_buffer
    checkpoint_name = f"checkpoint_{epoch}"
    checkpoint = {
        "command_history": command_history,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "game_params": game_params,
        "model_params": model_params,
        "optim_params": optim_params,
        "simulation_params": simulation_params,
        "execution_params": execution_params,
    }
    if not do_not_save_replay_buffer:
        checkpoint.update({"replay_buffer": assembler.buffer})

    if save_uncompressed:
        torch.save(checkpoint, checkpoint_dir / f"{checkpoint_name}.pt")
    else:
        # with zipfile.ZipFile(Path(checkpoint_dir) / f"{checkpoint_name}.zip", "w", allowZip64=True) as z:
        #    with z.open(f"{checkpoint_name}.pt", "w", force_zip64=True) as f:
        #        torch.save(checkpoint, f)
        with gzip.open(checkpoint_dir / f"{checkpoint_name}.pt.gz", "wb") as f:
            torch.save(checkpoint, f)


def load_checkpoint(checkpoint_path: Path) -> Checkpoint:
    ext = checkpoint_path.suffix
    if ext == ".pt":
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    elif ext == ".gz":
        with gzip.open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location=torch.device('cpu'))
    elif ext == ".zip":
        with zipfile.ZipFile(checkpoint_path, "r", allowZip64=True) as z:
            checkpoint_unzipped_name = z.namelist()[0]
            with z.open(checkpoint_unzipped_name, "r") as f:
                checkpoint = torch.load(f)
    else:
        raise ValueError(
            "The checkpoint file extension must be either "
            "'.pt', '.gz', '.pt.gz' or '.zip'"
        )
    return checkpoint


EXT_PATTERN = re.compile(r"(\.pt|\.gz|\.pt\.gz|\.zip)$")


def gen_checkpoints(
    checkpoint_dir: Path, real_time: bool, only_last: bool = False
) -> Iterator[Checkpoint]:
    checkpoint_basepath = str(checkpoint_dir / "checkpoint_")
    epoch_list = set()
    checkpoint_ext_detected = False
    first_time = True
    # infinite loop, could be made elegant with inotify
    while first_time or real_time:
        if not first_time:
            time.sleep(2)
        first_time = False
        checkpoint_path_list_no_ext = [
            re.sub(EXT_PATTERN, "", checkpoint_path)
            for checkpoint_path in glob.glob(f"{checkpoint_basepath}*")
        ]
        new_epoch_list = {
            int(checkpoint_path_no_ext[len(checkpoint_basepath):])
            for checkpoint_path_no_ext in checkpoint_path_list_no_ext
        }
        if not checkpoint_ext_detected and new_epoch_list:
            checkpoint_ext = re.search(
                EXT_PATTERN, next(iter(glob.glob(f"{checkpoint_basepath}*")))
            ).group(0)
            checkpoint_ext_detected = True

        added_epoch_list = sorted(new_epoch_list - epoch_list)
        # if the evaluation runs in real time, only consider the latest checkpoint
        if real_time or only_last:
            added_epoch_list = added_epoch_list[-1:]
        epoch_list = new_epoch_list
        for epoch in added_epoch_list:
            print(f"loading checkpoint #{epoch}...")
            checkpoint_path = f"{checkpoint_basepath}{epoch}{checkpoint_ext}"
            yield load_checkpoint(checkpoint_path=Path(checkpoint_path))
