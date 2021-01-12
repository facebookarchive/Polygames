# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Iterator, Tuple, Callable, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn

import tube
import polygames

from .params import (
    GameParams,
    ModelParams,
    OptimParams,
    SimulationParams,
    ExecutionParams,
)
from . import utils
from .env_creation_helpers import create_model, create_game, create_player

from .model_zoo import utils as zutils

from .model_zoo import loss as model_loss

#######################################################################################
# OPTIMIZER CREATION
#######################################################################################


def create_optimizer(
    model: torch.jit.ScriptModule,
    optim_params: OptimParams,
    optim_state_dict: Optional[dict] = None,
) -> torch.optim.Optimizer:
    optim = torch.optim.Adam(
        model.parameters(), lr=optim_params.lr, eps=optim_params.eps
    )
    if optim_state_dict is not None and not optim_params.reset_optimizer_state:
        try:
            optim.load_state_dict(optim_state_dict)
        except ValueError:
            print("Optimizer state not compatible... skipping.")
    return optim


#######################################################################################
# TRAINING ENVIRONMENT CREATION
#######################################################################################


def create_training_environment(
    seed_generator: Iterator[int],
    model_path: Path,
    device: str,
    game_params: GameParams,
    simulation_params: SimulationParams,
    execution_params: ExecutionParams,
    model
) -> Tuple[tube.Context, polygames.ModelManager, Callable[[], List[int]], bool]:
    games = []
    context = tube.Context()
    print("Game generation device: {}".format(device))
    listen_ep = execution_params.listen
    connect_ep = execution_params.connect
    opponent_model_path = execution_params.opponent_model_path
    is_server = listen_ep != ""
    is_client = connect_ep != ""
    print("is_server is ", is_server)
    print("is_client is ", is_client)
    model_manager = polygames.ModelManager(
        simulation_params.act_batchsize,
        str(device),
        simulation_params.replay_capacity,
        next(seed_generator),
        str(model_path),
        simulation_params.train_channel_timeout_ms,
        simulation_params.train_channel_num_slots,
    )
    model_manager.set_find_batch_size_max_bs(simulation_params.bsfinder_max_bs)
    model_manager.set_find_batch_size_max_ms(simulation_params.bsfinder_max_ms)
    if is_server:
        model_manager.start_server(listen_ep)
    if is_client:
        model_manager.start_client(connect_ep)
    if is_client and is_server:
        raise RuntimeError("Client and server parameters have both been specified")

    rnn_state_shape = getattr(model, "rnn_state_shape", [])
    logit_value = getattr(model, "logit_value", False)

    print("rnn_state_shape is ", rnn_state_shape)

    if simulation_params.num_threads != 0:
        polygames.init_threads(simulation_params.num_threads)

    opgame = None
    op_rnn_state_shape = None
    op_rnn_seqlen = None
    op_logit_value = None
    if not is_server:
      if opponent_model_path:
        print("loading opponent model")
        checkpoint = utils.load_checkpoint(checkpoint_path=opponent_model_path)
        opmodel = create_model(
            game_params=checkpoint["game_params"],
            model_params=checkpoint["model_params"],
            resume_training=True,
            model_state_dict=checkpoint["model_state_dict"],
        )
        opponent_model_path = execution_params.checkpoint_dir / "model_opponent.pt"
        opmodel.save(str(opponent_model_path))
        opgame = create_game(
            checkpoint["game_params"],
            num_episode=-1,
            seed=next(seed_generator),
            eval_mode=False,
        )
        op_rnn_state_shape = getattr(opmodel, "rnn_state_shape", [])
        op_rnn_seqlen = 0
        if hasattr(checkpoint["execution_params"], "rnn_seqlen"):
          op_rnn_seqlen = checkpoint["execution_params"].rnn_seqlen
        op_logit_value = getattr(opmodel, "logit_value", False)
      model_manager_opponent = polygames.ModelManager(
          simulation_params.act_batchsize,
          str(device),
          simulation_params.replay_capacity,
          next(seed_generator),
          str(opponent_model_path) if opponent_model_path else str(model_path),
          simulation_params.train_channel_timeout_ms,
          simulation_params.train_channel_num_slots,
      )
      model_manager_opponent.set_find_batch_size_max_bs(simulation_params.bsfinder_max_bs)
      model_manager_opponent.set_find_batch_size_max_ms(simulation_params.bsfinder_max_ms)
      print("tournament_mode is " + str(execution_params.tournament_mode))
      if execution_params.tournament_mode:
        model_manager_opponent.set_is_tournament_opponent(True)
      if opponent_model_path:
        model_manager_opponent.set_dont_request_model_updates(True)
      if is_client:
        model_manager_opponent.start_client(connect_ep)
    if not is_server:
      train_channel = model_manager.get_train_channel()
      actor_channel = model_manager.get_act_channel()

      op_actor_channel = actor_channel
      if model_manager_opponent is not None:
        op_actor_channel = model_manager_opponent.get_act_channel()

      for i in range(simulation_params.num_game):
          game = create_game(
              game_params,
              num_episode=-1,
              seed=next(seed_generator),
              eval_mode=False,
              per_thread_batchsize=simulation_params.per_thread_batchsize,
              rewind=simulation_params.rewind,
              predict_end_state=game_params.predict_end_state,
              predict_n_states=game_params.predict_n_states,
          )
          player_1 = create_player(
              seed_generator=seed_generator,
              game=game,
              player=game_params.player,
              num_actor=simulation_params.num_actor,
              num_rollouts=simulation_params.num_rollouts,
              pure_mcts=False,
              actor_channel=actor_channel,
              model_manager=model_manager,
              human_mode=False,
              sample_before_step_idx=simulation_params.sample_before_step_idx,
              randomized_rollouts=simulation_params.randomized_rollouts,
              sampling_mcts=simulation_params.sampling_mcts,
              rnn_state_shape=rnn_state_shape,
              rnn_seqlen=execution_params.rnn_seqlen,
              logit_value=logit_value
          )
          player_1.set_name("dev")
          if game.is_one_player_game():
            game.add_player(player_1, train_channel)
          else:
            player_2 = create_player(
                seed_generator=seed_generator,
                game=opgame if opgame is not None else game,
                player=game_params.player,
                num_actor=simulation_params.num_actor,
                num_rollouts=simulation_params.num_rollouts,
                pure_mcts=False,
                actor_channel=op_actor_channel,
                model_manager=model_manager_opponent,
                human_mode=False,
                sample_before_step_idx=simulation_params.sample_before_step_idx,
                randomized_rollouts=simulation_params.randomized_rollouts,
                sampling_mcts=simulation_params.sampling_mcts,
                rnn_state_shape=op_rnn_state_shape if op_rnn_state_shape is not None else rnn_state_shape,
                rnn_seqlen=op_rnn_seqlen if op_rnn_seqlen is not None else execution_params.rnn_seqlen,
                logit_value=op_logit_value if op_logit_value is not None else logit_value
            )
            player_2.set_name("opponent")
            if next(seed_generator) % 2 == 0:
              game.add_player(player_1, train_channel, game, player_1)
              game.add_player(player_2, train_channel, opgame if opgame is not None else game, player_1)
            else:
              game.add_player(player_2, train_channel, opgame if opgame is not None else game, player_1)
              game.add_player(player_1, train_channel, game, player_1)

          context.push_env_thread(game)
          games.append(game)

    def get_train_reward() -> Callable[[], List[int]]:
        nonlocal games
        nonlocal opgame
        reward = []
        for game in games:
            reward.append(game.get_result()[0])
        if opgame is not None:
          reward.append(opgame.get_result()[0])

        return reward

    return context, model_manager, get_train_reward, is_client


#######################################################################################
# REPLAY BUFFER WARMING-UP
#######################################################################################


def warm_up_replay_buffer(
    model_manager: polygames.ModelManager, replay_warmup: int
) -> None:
    model_manager.start()
    prev_buffer_size = -1
    t = t_init = time.time()
    t0 = -1
    size0 = 0
    while model_manager.buffer_size() < replay_warmup:
        buffer_size = model_manager.buffer_size()
        if buffer_size != prev_buffer_size:  # avoid flooding stdout
            if buffer_size > 10000 and t0 == -1:
                size0 = buffer_size
                t0 = time.time()
            prev_buffer_size = max(prev_buffer_size, 0)
            frame_rate = (buffer_size - prev_buffer_size) / (time.time() - t)
            frame_rate = int(frame_rate)
            prev_buffer_size = buffer_size
            t = time.time()
            duration = t - t_init
            print(
                f"warming-up replay buffer: {(buffer_size * 100) // replay_warmup}% "
                f"({buffer_size}/{replay_warmup}) in {duration:.2f}s "
                f"- speed: {frame_rate} frames/s",
            )
        time.sleep(2)
    print(
        f"replay buffer warmed up: 100% "
        f"({model_manager.buffer_size()}/{replay_warmup})"
        "                                                                          "
    )
    print(
        "avg speed: %.2f frames/s"
        % ((model_manager.buffer_size() - size0) / (time.time() - t0))
    )


#######################################################################################
# TRAINING
#######################################################################################

class ModelWrapperForDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x: torch.Tensor, rnn_state: torch.Tensor=None, rnn_state_mask: torch.Tensor=None):
        if rnn_state is None:
          return self.module._forward(x, return_logit=True)
        else:
          return self.module._forward(x, rnn_state, rnn_state_mask, return_logit=True)

class DDPWrapperForModel(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def _forward(self, x: torch.Tensor, rnn_state: torch.Tensor=None, rnn_state_mask: torch.Tensor=None, return_logit: bool=False):
        if not return_logit:
            raise RuntimeError("DDPWrapperForModel: return_logit is false")
        if rnn_state is None:
          return self.module.forward(x)
        else:
          return self.module.forward(x, rnn_state, rnn_state_mask)

_pre_num_add = None
_pre_num_sample = None
_running_add_rate = 0
_running_sample_rate = 0
_last_train_time = 0
def _train_epoch(
    model: torch.jit.ScriptModule,
    device: torch.device,
    ddpmodel: ModelWrapperForDDP,
    batchsizes,
    optim: torch.optim.Optimizer,
    model_manager: polygames.ModelManager,
    stat: utils.MultiCounter,
    epoch: int,
    optim_params: OptimParams,
    sync_period: int,
) -> None:
    global _pre_num_add
    global _pre_num_sample
    global _running_add_rate
    global _running_sample_rate
    global _last_train_time
    global _remote_replay_buffer_inited
    if _pre_num_add is None:
        pre_num_add = model_manager.buffer_num_add()
        pre_num_sample = model_manager.buffer_num_sample()
    else:
        pre_num_add = _pre_num_add
        pre_num_sample = _pre_num_sample
    sync_s = 0.
    num_sync = 0

    train_start_time = time.time()

    if pre_num_sample > 0:
        print("sample/add ratio ", float(pre_num_sample) / pre_num_add)

    if _last_train_time == 0:
      _last_train_time = time.time();

    batchsize = optim_params.batchsize

    lossmodel = DDPWrapperForModel(ddpmodel) if ddpmodel is not None else model

    lossmodel.train()

    world_size = 0
    rank = 0
    if ddpmodel is not None:
      print("DDP is active")
      world_size = torch.distributed.get_world_size()
      rank = torch.distributed.get_rank()

      print("World size %d, rank %d. Waiting for all processes" % (world_size, rank))
      torch.distributed.barrier()
      print("Synchronizing model")
      for p in ddpmodel.parameters():
        torch.distributed.broadcast(p.data, 0)
      for p in ddpmodel.buffers():
        torch.distributed.broadcast(p.data, 0)
      print("Synchronized, start training")

    has_predict = False
    cpubatch = {}
    for k, v in batchsizes.items():
      sizes = v.copy()
      sizes.insert(0, batchsize)
      cpubatch[k] = torch.empty(sizes)
      if k == "predict_pi":
        has_predict = True

    for eid in range(optim_params.epoch_len):
        while _running_add_rate * 1.25 < _running_sample_rate:
          print("add rate insufficient, waiting")
          time.sleep(5)
          t = time.time()
          time_elapsed = t - _last_train_time
          _last_train_time = t
          alpha = pow(0.99, time_elapsed)
          post_num_add = model_manager.buffer_num_add()
          post_num_sample = model_manager.buffer_num_sample()
          delta_add = post_num_add - pre_num_add
          delta_sample = post_num_sample - pre_num_sample
          _running_add_rate = _running_add_rate * alpha + (delta_add / time_elapsed) * (1 - alpha)
          _running_sample_rate = _running_sample_rate * alpha + (delta_sample / time_elapsed) * (1 - alpha)
          pre_num_add = post_num_add
          pre_num_sample = post_num_sample
          print("running add rate: %.2f / s" % (_running_add_rate))
          print("running sample rate: %.2f / s" % (_running_sample_rate))
          print("current add rate: %.2f / s" % (delta_add / time_elapsed))
          print("current sample rate: %.2f / s" % (delta_sample / time_elapsed))

        if world_size > 0:
          batchlist = None
          if rank == 0:
            batchlist = {}
            for k in cpubatch.keys():
              batchlist[k] = []
            for i in range(world_size):
              for k,v in model_manager.sample(batchsize).items():
                batchlist[k].append(v)
          for k, v in cpubatch.items():
            torch.distributed.scatter(v, batchlist[k] if rank == 0 else None)
          batch = utils.to_device(cpubatch, device)
        else:
          batch = model_manager.sample(batchsize)
          batch = utils.to_device(batch, device)
        for k, v in batch.items():
          batch[k] = v.detach()
        loss, v_err, pi_err, predict_err = model_loss.mcts_loss(model, lossmodel, batch)
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(lossmodel.parameters(), optim_params.grad_clip)
        optim.step()
        optim.zero_grad()

        stat["v_err"].feed(v_err.item())
        stat["pi_err"].feed(pi_err.item())
        if has_predict:
          stat["predict_err"].feed(predict_err.item())
        stat["loss"].feed(loss.item())
        stat["grad_norm"].feed(grad_norm)

        if (epoch * optim_params.epoch_len + eid + 1) % sync_period == 0:
            sync_t0 = time.time()
            model_manager.update_model(model.state_dict())
            sync_s += time.time() - sync_t0
            num_sync += 1

        t = time.time()
        time_elapsed = t - _last_train_time
        _last_train_time = t
        alpha = pow(0.99, time_elapsed)
        post_num_add = model_manager.buffer_num_add()
        post_num_sample = model_manager.buffer_num_sample()
        delta_add = post_num_add - pre_num_add
        delta_sample = post_num_sample - pre_num_sample
        _running_add_rate = _running_add_rate * alpha + (delta_add / time_elapsed) * (1 - alpha)
        _running_sample_rate = _running_sample_rate * alpha + (delta_sample / time_elapsed) * (1 - alpha)
        pre_num_add = post_num_add
        pre_num_sample = post_num_sample

    total_time_elapsed = time.time() - train_start_time

    print("running add rate: %.2f / s" % (_running_add_rate))
    print("running sample rate: %.2f / s" % (_running_sample_rate))
    print("current add rate: %.2f / s" % (delta_add / time_elapsed))
    print("current sample rate: %.2f / s" % (delta_sample / time_elapsed))
    print(f"syncing duration: {sync_s:2f}s for {num_sync} syncs ({int(100 * sync_s / total_time_elapsed)}% of train time)")

    _pre_num_add = pre_num_add
    _pre_num_sample = pre_num_sample

    stat.summary(epoch)
    stat.reset()

def train_model(
    command_history: utils.CommandHistory,
    start_time: float,
    model: torch.jit.ScriptModule,
    device: torch.device,
    ddpmodel,
    optim: torch.optim.Optimizer,
    context: tube.Context,
    model_manager: polygames.ModelManager,
    get_train_reward: Callable[[], List[int]],
    game_params: GameParams,
    model_params: ModelParams,
    optim_params: OptimParams,
    simulation_params: SimulationParams,
    execution_params: ExecutionParams,
    epoch: int = 0,
) -> None:

    info = zutils.get_game_info(game_params)
    c, h, w = info["feature_size"][:3]
    rc, rh, rw = info["raw_feature_size"][:3]
    c_prime, h_prime, w_prime = info[
        "action_size"
    ][:3]

    predicts = (2 if game_params.predict_end_state else 0) + game_params.predict_n_states

    batchsizes = {
      "s":  [c, h, w],
      "v": [3 if getattr(model, "logit_value", False) else 1],
      "pred_v": [1],
      "pi": [c_prime, h_prime, w_prime],
      "pi_mask": [c_prime, h_prime, w_prime]
    }

    if game_params.player == "forward":
      batchsizes["action_pi"] = [c_prime, h_prime, w_prime]

    if predicts > 0:
      batchsizes["predict_pi"] = [rc * predicts, rh, rw]
      batchsizes["predict_pi_mask"] = [rc * predicts, rh, rw]

    if getattr(model, "rnn_state_shape", None) is not None:
      batchsizes["rnn_state_mask"] = [1]

    if execution_params.rnn_seqlen > 0:
      for k, v in batchsizes.items():
        batchsizes[k] = [execution_params.rnn_seqlen, *v]

    if getattr(model, "rnn_state_shape", None) is not None:
      batchsizes["rnn_initial_state"] = model.rnn_state_shape

    rank = 0
    if ddpmodel:
        rank = torch.distributed.get_rank()

    executor = ThreadPoolExecutor(max_workers=1)
    savefuture = None

    stat = utils.MultiCounter(execution_params.checkpoint_dir)
    max_time = execution_params.max_time
    init_epoch = epoch
    while max_time is None or time.time() < start_time + max_time:
        if epoch - init_epoch >= optim_params.num_epoch:
            break
        epoch += 1
        if rank == 0 and epoch % execution_params.saving_period == 0:
            model_manager.add_tournament_model("e%d" % (epoch), model.state_dict())
            savestart = time.time()
            if savefuture is not None:
               savefuture.result()
            savefuture = utils.save_checkpoint(
                command_history=command_history,
                epoch=epoch,
                model=model,
                optim=optim,
                game_params=game_params,
                model_params=model_params,
                optim_params=optim_params,
                simulation_params=simulation_params,
                execution_params=execution_params,
                executor=executor
            )
            print("checkpoint saved in %gs" % (time.time() - savestart))
        _train_epoch(
            model=model,
            device=device,
            ddpmodel=ddpmodel,
            batchsizes=batchsizes,
            optim=optim,
            model_manager=model_manager,
            stat=stat,
            epoch=epoch,
            optim_params=optim_params,
            sync_period=simulation_params.sync_period,
        )
        # resource usage stats
        print("Resource usage:")
        print(utils.get_res_usage_str())
        print("Context stats:")
        print(context.get_stats_str())
        # train result
        print(
            ">>>train: epoch: %d, %s" % (epoch, utils.Result(get_train_reward()).log()),
            flush=True,
        )
    if savefuture is not None:
        savefuture.result()
    # checkpoint last state
    utils.save_checkpoint(
        command_history=command_history,
        epoch=epoch,
        model=model,
        optim=optim,
        game_params=game_params,
        model_params=model_params,
        optim_params=optim_params,
        simulation_params=simulation_params,
        execution_params=execution_params,
    )

def client_loop(
    model_manager: polygames.ModelManager,
    start_time: float,
    context: tube.Context,
    execution_params: ExecutionParams,
) -> None:
    model_manager.start()
    max_time = execution_params.max_time
    while max_time is None or time.time() < start_time + max_time:
        time.sleep(60)
        print("Resource usage:")
        print(utils.get_res_usage_str())
        print("Context stats:")
        print(context.get_stats_str())

#######################################################################################
# OVERALL TRAINING WORKFLOW
#######################################################################################


def run_training(
    command_history: utils.CommandHistory,
    game_params: GameParams,
    model_params: ModelParams,
    optim_params: OptimParams,
    simulation_params: SimulationParams,
    execution_params: ExecutionParams,
) -> None:
    start_time = time.time()
    logger_path = os.path.join(execution_params.checkpoint_dir, "train.log")
    sys.stdout = utils.Logger(logger_path)

    print("#" * 70)
    print("#" + "TRAINING".center(68) + "#")
    print("#" * 70)

    if execution_params.rnn_seqlen > 0:
      optim_params.batchsize //= execution_params.rnn_seqlen
      simulation_params.replay_capacity //= execution_params.rnn_seqlen
      simulation_params.replay_warmup //= execution_params.rnn_seqlen
      simulation_params.train_channel_num_slots //= execution_params.rnn_seqlen


    print("setting-up pseudo-random generator...")
    seed_generator = utils.generate_random_seeds(seed=execution_params.seed)

    # checkpoint, resume from where it stops
    epoch = 0
    ckpts = list(utils.gen_checkpoints(checkpoint_dir=execution_params.checkpoint_dir, only_last=True, real_time=False))
    checkpoint = {}
    if ckpts:
        checkpoint = ckpts[0]
        former_command_history = checkpoint["command_history"]
        command_history.build_history(former_command_history)
        optim_params = command_history.update_params_from_checkpoint(
            checkpoint_params=checkpoint["optim_params"], resume_params=optim_params
        )
        simulation_params = command_history.update_params_from_checkpoint(
            checkpoint_params=checkpoint["simulation_params"],
            resume_params=simulation_params,
        )
        execution_params = command_history.update_params_from_checkpoint(
            checkpoint_params=checkpoint["execution_params"],
            resume_params=execution_params,
        )
    if command_history.last_command_contains("init_checkpoint"):
        if ckpts:
            raise RuntimeError("Cannot restart from init_checkpoint, already restarting from non-empty checkpoint_dir")
        # pretrained model, consider new training from epoch zero
        print("loading pretrained model from checkpoint...")
        checkpoint = utils.load_checkpoint(checkpoint_path=model_params.init_checkpoint)
    if checkpoint:
        # game_params and model_params cannot change on a checkpoint
        # either write the same, or don't specify them
        ignored = {"init_checkpoint", "game_name"}  # this one can change
        current_params = dict(game_params=game_params, model_params=model_params)
#        for params_name, params in current_params.items():
#            for attr, val in asdict(params).items():
#                if command_history.last_command_contains(attr) and attr not in ignored:
#                    ckpt_val = getattr(checkpoint[params_name], attr)
#                    assert val == ckpt_val, f"When resuming, got '{val}' for {attr} but cannot override from past run with '{ckpt_val}'."
        specified_game_name = game_params.game_name
        game_params = checkpoint["game_params"]
        if specified_game_name is not None:
          game_params.game_name = specified_game_name
        model_params = checkpoint["model_params"]
        for params_name, params in current_params.items():
            for attr, val in asdict(params).items():
                if command_history.last_command_contains(attr) and attr not in ignored:
                    ckpt_val = getattr(checkpoint[params_name], attr)
                    if val != ckpt_val:
                      print(f"Note: overrriding {attr} from {ckpt_val} to {val}")
                      setattr(checkpoint[params_name], attr, val)
        epoch = checkpoint["epoch"]
        print("reconstructing the model...")
    else:
        print("creating and saving the model...")
    if len(execution_params.devices) != 1:
        raise RuntimeError("Only one device is supported for training")
    device = execution_params.devices[0]

    model = create_model(
          game_params=game_params,
          model_params=model_params,
          resume_training=bool(checkpoint),
          model_state_dict=checkpoint["model_state_dict"] if checkpoint else None,
      ).to(device)

    model_path = execution_params.checkpoint_dir / "model.pt"
    model.save(str(model_path))

    ddpmodel = None
    if os.environ.get("RANK") is not None:
        torch.distributed.init_process_group(backend="gloo", timeout=datetime.timedelta(0, 864000))
        ddpmodel = nn.parallel.DistributedDataParallel(ModelWrapperForDDP(model), broadcast_buffers=False, find_unused_parameters=False)

    print("creating optimizer...")
    optim = create_optimizer(
        model=ddpmodel if ddpmodel is not None else model,
        optim_params=optim_params,
        optim_state_dict=checkpoint.get("optim_state_dict", None),
    )

    print("creating training environment...")
    context, model_manager, get_train_reward, is_client = create_training_environment(
        seed_generator=seed_generator,
        model_path=model_path,
        device=device,
        game_params=game_params,
        simulation_params=simulation_params,
        execution_params=execution_params,
        model=model
    )
    if not is_client:
        model_manager.update_model(model.state_dict())
    model_manager.add_tournament_model("init", model.state_dict())
    context.start()

    if is_client:
      client_loop(
          model_manager=model_manager,
          start_time=start_time,
          context=context,
          execution_params=execution_params
      )
    else:
      if ddpmodel is None or torch.distributed.get_rank() == 0:
        print("warming-up replay buffer...")
        warm_up_replay_buffer(
            model_manager=model_manager,
            replay_warmup=simulation_params.replay_warmup
        )

      print("training model...")
      train_model(
          command_history=command_history,
          start_time=start_time,
          model=model,
          device=device,
          ddpmodel=ddpmodel,
          optim=optim,
          context=context,
          model_manager=model_manager,
          get_train_reward=get_train_reward,
          game_params=game_params,
          model_params=model_params,
          optim_params=optim_params,
          simulation_params=simulation_params,
          execution_params=execution_params,
          epoch=epoch
      )

    elapsed_time = time.time() - start_time
    print(f"total time: {elapsed_time} s")
