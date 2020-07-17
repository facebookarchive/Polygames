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
#    if optim_state_dict is not None:
#        try:
#            optim.load_state_dict(optim_state_dict)
#        except ValueError:
#            print("Optimizer state not compatible... skipping.")
    return optim


#######################################################################################
# TRAINING ENVIRONMENT CREATION
#######################################################################################


def create_training_environment(
    seed_generator: Iterator[int],
    model_path: Path,
    game_generation_devices: List[str],
    game_params: GameParams,
    simulation_params: SimulationParams,
    execution_params: ExecutionParams,
    rnn_state_shape: List[int]
) -> Tuple[tube.Context, tube.ChannelAssembler, Callable[[], List[int]], bool]:
    games = []
    context = tube.Context()
    print("Game generation devices: {}".format(game_generation_devices))
    server_listen_endpoint = execution_params.server_listen_endpoint
    server_connect_hostname = execution_params.server_connect_hostname
    opponent_model_path = execution_params.opponent_model_path
    is_server = server_listen_endpoint != ""
    is_client = server_connect_hostname != ""
    print("is_server is ", is_server)
    print("is_client is ", is_client)
    assembler = tube.ChannelAssembler(
        simulation_params.act_batchsize,
        len(game_generation_devices) if not is_server else 0,
        [str(i) for i in game_generation_devices],
        simulation_params.replay_capacity,
        next(seed_generator),
        str(model_path),
        simulation_params.train_channel_timeout_ms,
        simulation_params.train_channel_num_slots,
    )
    if is_server:
      assembler.start_server(server_listen_endpoint)
    if is_client:
      assembler.start_client(server_connect_hostname)
    if is_client and is_server:
      raise RuntimeError("Client and server parameters have both been specified")
    opgame = None
    op_rnn_state_shape = None
    op_rnn_seqlen = None
    if not is_server:
      if opponent_model_path:
        print("loading opponent model")
        checkpoint = utils.load_checkpoint(checkpoint_path=opponent_model_path)
        model = create_model(
            game_params=checkpoint["game_params"],
            model_params=checkpoint["model_params"],
            resume_training=True,
            model_state_dict=checkpoint["model_state_dict"],
        )
        opponent_model_path = execution_params.checkpoint_dir / "model_opponent.pt"
        model.save(str(opponent_model_path))
        opgame = create_game(
            checkpoint["game_params"],
            num_episode=-1,
            seed=next(seed_generator),
            eval_mode=False,
        )
        op_rnn_state_shape = []
        if hasattr(model, "rnn_cells") and model.rnn_cells > 0:
          op_rnn_state_shape = [model.rnn_cells, model.rnn_channels]
        op_rnn_seqlen = 0
        if hasattr(checkpoint["execution_params"], "rnn_seqlen"):
          op_rnn_seqlen = checkpoint["execution_params"].rnn_seqlen
      assembler_opponent = tube.ChannelAssembler(
          simulation_params.act_batchsize,
          len(game_generation_devices) if not is_server else 0,
          [str(i) for i in game_generation_devices],
          simulation_params.replay_capacity,
          next(seed_generator),
          str(opponent_model_path) if opponent_model_path else str(model_path),
          simulation_params.train_channel_timeout_ms,
          simulation_params.train_channel_num_slots,
      )
      print("tournament_mode is " + str(execution_params.tournament_mode))
      if execution_params.tournament_mode:
        assembler_opponent.set_is_tournament_opponent(True)
      if opponent_model_path:
        assembler_opponent.set_dont_request_model_updates(True)
      if is_client:
        assembler_opponent.start_client(server_connect_hostname)
    if not is_server:
      train_channel = assembler.get_train_channel()
      actor_channels = assembler.get_act_channels()
      actor_channel = actor_channels[0]

      op_actor_channel = actor_channel
      if assembler_opponent is not None:
        op_actor_channel = assembler_opponent.get_act_channels()[0]

      for i in range(simulation_params.num_game):
          game = create_game(
              game_params,
              num_episode=-1,
              seed=next(seed_generator),
              eval_mode=False,
              per_thread_batchsize=simulation_params.per_thread_batchsize,
              rewind=simulation_params.rewind,
              persistent_tree=simulation_params.persistent_tree,
              predict_end_state=game_params.predict_end_state,
              predict_n_states=game_params.predict_n_states,
          )
          if simulation_params.per_thread_batchsize != 0:
              player_1 = create_player(
                  seed_generator=seed_generator,
                  game=game,
                  num_actor=simulation_params.num_actor,
                  num_rollouts=simulation_params.num_rollouts,
                  pure_mcts=False,
                  actor_channel=actor_channel,
                  assembler=assembler,
                  human_mode=False,
                  sample_before_step_idx=simulation_params.sample_before_step_idx,
                  randomized_rollouts=simulation_params.randomized_rollouts,
                  sampling_mcts=simulation_params.sampling_mcts,
                  move_select_use_mcts_value=simulation_params.move_select_use_mcts_value,
                  rnn_state_shape=rnn_state_shape,
                  rnn_seqlen=execution_params.rnn_seqlen,
              )
              player_1.set_name("dev")
              if game.is_one_player_game():
                game.add_player(player_1, train_channel)
              else:
                player_2 = create_player(
                    seed_generator=seed_generator,
                    game=opgame if opgame is not None else game,
                    num_actor=simulation_params.num_actor,
                    num_rollouts=simulation_params.num_rollouts,
                    pure_mcts=False,
                    actor_channel=op_actor_channel,
                    assembler=assembler_opponent,
                    human_mode=False,
                    sample_before_step_idx=simulation_params.sample_before_step_idx,
                    randomized_rollouts=simulation_params.randomized_rollouts,
                    sampling_mcts=simulation_params.sampling_mcts,
                    move_select_use_mcts_value=simulation_params.move_select_use_mcts_value,
                    rnn_state_shape=op_rnn_state_shape if op_rnn_state_shape is not None else rnn_state_shape,
                    rnn_seqlen=op_rnn_seqlen if op_rnn_seqlen is not None else execution_params.rnn_seqlen,
                )
                player_2.set_name("opponent")
                if next(seed_generator) % 2 == 0:
                  game.add_player(player_1, train_channel, game, player_1)
                  game.add_player(player_2, train_channel, opgame if opgame is not None else game, player_1)
                else:
                  game.add_player(player_2, train_channel, opgame if opgame is not None else game, player_1)
                  game.add_player(player_1, train_channel, game, player_1)
          else:
              player_1 = create_player(
                  seed_generator=seed_generator,
                  game=game,
                  num_actor=simulation_params.num_actor,
                  num_rollouts=simulation_params.num_rollouts,
                  pure_mcts=False,
                  actor_channel=actor_channels[i % len(actor_channels)],
                  assembler=None,
                  human_mode=False,
              )
              game.add_player(player_1, train_channel)
              if not game.is_one_player_game():
                  player_2 = create_player(
                      seed_generator=seed_generator,
                      game=game,
                      num_actor=simulation_params.num_actor,
                      num_rollouts=simulation_params.num_rollouts,
                      pure_mcts=False,
                      actor_channel=actor_channels[i % len(actor_channels)],
                      assembler=None,
                      human_mode=False,
                  )
                  game.add_player(player_2, train_channel)

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

    return context, assembler, get_train_reward, is_client


#######################################################################################
# REPLAY BUFFER WARMING-UP
#######################################################################################


def warm_up_replay_buffer(
    assembler: tube.ChannelAssembler, replay_warmup: int, replay_buffer: Optional[bytes]
) -> None:
    if replay_buffer is not None:
        print("loading replay buffer...")
        assembler.buffer = replay_buffer
    assembler.start()
    prev_buffer_size = -1
    t = t_init = time.time()
    t0 = -1
    size0 = 0
    while assembler.buffer_size() < replay_warmup:
        buffer_size = assembler.buffer_size()
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
        f"({assembler.buffer_size()}/{replay_warmup})"
        "                                                                          "
    )
    print(
        "avg speed: %.2f frames/s"
        % ((assembler.buffer_size() - size0) / (time.time() - t0))
    )


#######################################################################################
# TRAINING
#######################################################################################

class LossOnDevice(torch.nn.Module):
  def __init__(self, model : torch.jit.ScriptModule):
    torch.nn.Module.__init__(self)
    self.model = model

  def forward(self, batch: Dict[str, torch.Tensor]):
    loss, v_err, pi_err = self.model.loss(batch["s"], batch["v"], batch["pi"], batch["pi_mask"])
    loss.backward()
    return loss.detach(), v_err, pi_err

def _loss_on_device(batch: Dict[str, torch.Tensor], model: torch.jit.ScriptModule, device: torch.device):
  loss, v_err, pi_err = model.loss(batch["s"], batch["v"], batch["pi"], batch["pi_mask"])
  loss.backward()
  return loss.detach(), v_err, pi_err

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
          return self.module.forward(x, state, rnn_state_mask)

_pre_num_add = None
_pre_num_sample = None
_running_add_rate = 0
_running_sample_rate = 0
_last_train_time = 0
_remote_replay_buffer_inited = False
def _train_epoch(
    models: List[torch.jit.ScriptModule],
    devices: List[torch.device],
    ddpmodel: ModelWrapperForDDP,
    batchsizes,
    optim: torch.optim.Optimizer,
    assembler: tube.ChannelAssembler,
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
      pre_num_add = assembler.buffer_num_add()
      pre_num_sample = assembler.buffer_num_sample()
    sync_s = 0.
    num_sync = 0

    if pre_num_sample > 0:
        print("sample/add ratio ", float(pre_num_sample) / pre_num_add)

    if _last_train_time == 0:
      _last_train_time = time.time();

    batchsize = optim_params.batchsize
    #executor = ThreadPoolExecutor(max_workers=len(models))

    #lossc = [torch.jit.script(LossOnDevice(m)) for m in models]
    #lossf = None

    lossmodel = DDPWrapperForModel(ddpmodel) if ddpmodel is not None else models[0]

    lossmodel.train()

    next_batch = None

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

      if not _remote_replay_buffer_inited:
        _remote_replay_buffer_inited = True

        if rank == 0:
          assembler.start_replay_buffer_server("0.0.0.0:20806")
        else:
          time.sleep(1)
          assembler.start_replay_buffer_client("127.0.0.1:20806")
      if rank != 0:
        next_batch = assembler.remote_sample(batchsize)

    start = time.time()

    t0 = 0
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    t6 = 0
    t7 = 0

    has_predict = False
    cpubatch = {}
    for k, v in batchsizes.items():
      sizes = v.copy()
      sizes.insert(0, batchsize)
      cpubatch[k] = torch.empty(sizes)
      if k == "predict_pi":
        has_predict = True

    for eid in range(optim_params.epoch_len):
        while _running_add_rate * 1.5 < _running_sample_rate:
          print("add rate insufficient, waiting")
          time.sleep(5)
          t = time.time()
          time_elapsed = t - _last_train_time
          _last_train_time = t
          alpha = pow(0.99, time_elapsed)
          post_num_add = assembler.buffer_num_add()
          post_num_sample = assembler.buffer_num_sample()
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
          start = time.time()

        #if ddpmodel is not None:
        if True:
          t = time.time()
          t0 += t - start
          start = t
          if world_size > 0:
            batchlist = None
            if False:
              if rank == 0:
                batch = assembler.sample(batchsize)
              else:
                batch = next_batch.get()
                next_batch = assembler.remote_sample(batchsize)
              batch = utils.to_device(batch, devices[0])
            else:
              if rank == 0:
                batchlist = {}
                for k in cpubatch.keys():
                  batchlist[k] = []
                for i in range(world_size):
                  for k,v in assembler.sample(batchsize).items():
                    batchlist[k].append(v)
              for k, v in cpubatch.items():
                torch.distributed.scatter(v, batchlist[k] if rank == 0 else None)
              batch = utils.to_device(cpubatch, devices[0])
          else:
            batch = assembler.sample(batchsize)
            batch = utils.to_device(batch, devices[0])
          for k, v in batch.items():
            batch[k] = v.detach()
          t = time.time()
          t1 += t - start
          start = t
          if "rnn_initial_state" in batch:
            if has_predict:
              loss, v_err, pi_err, predict_err = models[0].loss(lossmodel, batch["s"], batch["rnn_initial_state"], batch["rnn_state_mask"], batch["v"], batch["pi"], batch["pi_mask"], batch["predict_pi"], batch["predict_pi_mask"])
            else:
              loss, v_err, pi_err, predict_err = models[0].loss(lossmodel, batch["s"], batch["rnn_initial_state"], batch["rnn_state_mask"], batch["v"], batch["pi"], batch["pi_mask"])
          else:
            if has_predict:
              loss, v_err, pi_err, predict_err = models[0].loss(lossmodel, batch["s"], batch["v"], batch["pi"], batch["pi_mask"], batch["predict_pi"], batch["predict_pi_mask"])
            else:
              loss, v_err, pi_err, predict_err = models[0].loss(lossmodel, batch["s"], batch["v"], batch["pi"], batch["pi_mask"])
          t = time.time()
          t2 += t - start
          start = t
          loss.backward()
          t = time.time()
          t3 += t - start
          start = t
          grad_norm = nn.utils.clip_grad_norm_(lossmodel.parameters(), optim_params.grad_clip)
          t = time.time()
          t4 += t - start
          start = t
          optim.step()
          optim.zero_grad()

          t = time.time()
          t5 += t - start
          start = t

          stat["v_err"].feed(v_err.item())
          stat["pi_err"].feed(pi_err.item())
          if has_predict:
            stat["predict_err"].feed(predict_err.item())
          stat["loss"].feed(loss.item())
        else:

          losses = []
          if False:
            losses = assembler.loss(lossc)
          elif False:
            losses = assembler.loss(batchsize, devices, models)
          elif True:
            futures = []
            for device, m, lc in zip(devices, models, lossc):
              t = time.time()
              t0 += t - start
              start = t
              batch = assembler.sample(batchsize)
              batch = utils.to_device(batch, device)
              t = time.time()
              t1 += t - start
              start = t
              #if lossf is None:
              #  lossf = torch.jit.trace(lc, (batch, device))
              futures.append(torch.jit._fork(lc, batch))
              #futures.append(
              #    executor.submit(torch.jit._fork, lc, batch)
              #)
            t = time.time()
            t0 += t - start
            start = t
            #tmp = [future.result() for future in futures]
            #losses = [torch.jit._wait(future) for future in tmp]
            losses = [torch.jit._wait(future) for future in futures]
            #losses = [future.result() for future in futures]
            t = time.time()
            t2 += t - start
            start = t
          else:
            futures = []
            losses = []
            for device, m in zip(devices, models):
              batch = assembler.sample(batchsize)
              batch = utils.to_device(batch, device)
              loss, v_err, pi_err = _loss_on_device(batch, m, device)
              losses.append((loss, v_err, pi_err))
              #futures.append(
              #    executor.submit(_loss_on_device, assembler, batchsize, m, device)
              #)
            #results = [future.result() for future in futures]

          for loss, v_err, pi_err in losses:
            stat["v_err"].feed(v_err.item())
            stat["pi_err"].feed(pi_err.item())
            stat["loss"].feed(loss.item())

          params = [p for p in models[0].parameters()]
          buffers = [p for p in models[0].buffers()]

          t = time.time()
          t3 += t - start
          start = t

          for m in models:
            if m is not models[0]:
              for i, p in enumerate(m.parameters()):
                if p.grad is not None:
                  params[i].grad.data.add(p.grad.data.to(devices[0]))
                  p.grad.detach_()
                  p.grad.zero_()

          t = time.time()
          t4 += t - start
          start = t

          grad_norm = nn.utils.clip_grad_norm_(params, optim_params.grad_clip)
          t = time.time()
          t5 += t - start
          start = t
          optim.step()
          optim.zero_grad()

          t = time.time()
          t6 += t - start
          start = t

          for m in models:
            if m is not models[0]:
              for i, p in enumerate(m.parameters()):
                p.data.copy_(params[i].data)
              for i, p in enumerate(m.buffers()):
                p.data.copy_(buffers[i].data)

        t = time.time()
        t7 += t - start
        start = t

        if (epoch * optim_params.epoch_len + eid + 1) % sync_period == 0:
            sync_t0 = time.time()
            assembler.update_model(models[0].state_dict())
            sync_s += time.time() - sync_t0
            num_sync += 1

        stat["grad_norm"].feed(grad_norm)

        t = time.time()
        time_elapsed = t - _last_train_time
        _last_train_time = t
        alpha = pow(0.99, time_elapsed)
        post_num_add = assembler.buffer_num_add()
        post_num_sample = assembler.buffer_num_sample()
        delta_add = post_num_add - pre_num_add
        delta_sample = post_num_sample - pre_num_sample
        _running_add_rate = _running_add_rate * alpha + (delta_add / time_elapsed) * (1 - alpha)
        _running_sample_rate = _running_sample_rate * alpha + (delta_sample / time_elapsed) * (1 - alpha)
        pre_num_add = post_num_add
        pre_num_sample = post_num_sample

    print("times: ", t0, t1, t2, t3, t4, t5, t6, t7)

    print("running add rate: %.2f / s" % (_running_add_rate))
    print("running sample rate: %.2f / s" % (_running_sample_rate))
    print("current add rate: %.2f / s" % (delta_add / time_elapsed))
    print("current sample rate: %.2f / s" % (delta_sample / time_elapsed))
    print(f"syncing duration: {sync_s:2f}s for {num_sync} syncs ({int(100 * sync_s / time_elapsed)}% of train time)")

    stat.summary(epoch)
    stat.reset()

def train_model(
    command_history: utils.CommandHistory,
    start_time: float,
    models: List[torch.jit.ScriptModule],
    devices: List[torch.device],
    ddpmodel,
    optim: torch.optim.Optimizer,
    context: tube.Context,
    assembler: tube.ChannelAssembler,
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
      "v": [1],
      "pi": [c_prime, h_prime, w_prime],
      "pi_mask": [c_prime, h_prime, w_prime]
    }

    if predicts > 0:
      batchsizes["predict_pi"] = [rc * predicts, rh, rw]
      batchsizes["predict_pi_mask"] = [rc * predicts, rh, rw]

    if model_params.rnn_interval > 0:
      batchsizes["rnn_state_mask"] = [1]

    if execution_params.rnn_seqlen > 0:
      for k, v in batchsizes.items():
        batchsizes[k] = [execution_params.rnn_seqlen, *v]

    if model_params.rnn_interval > 0:
      batchsizes["rnn_initial_state"] = [models[0].rnn_cells * models[0].rnn_channels * rh * rw]

    rank = 0
    if ddpmodel:
        rank = torch.distributed.get_rank()

    stat = utils.MultiCounter(execution_params.checkpoint_dir)
    max_time = execution_params.max_time
    init_epoch = epoch
    while max_time is None or time.time() < start_time + max_time:
        if epoch - init_epoch >= optim_params.num_epoch:
            break
        epoch += 1
        if rank == 0 and epoch % execution_params.saving_period == 0:
        #if epoch % execution_params.saving_period == 0:
            assembler.add_tournament_model("e%d" % (epoch), models[0].state_dict())
            utils.save_checkpoint(
                command_history=command_history,
                epoch=epoch,
                model=models[0],
                optim=optim,
                assembler=assembler,
                game_params=game_params,
                model_params=model_params,
                optim_params=optim_params,
                simulation_params=simulation_params,
                execution_params=execution_params,
            )
        _train_epoch(
            models=models,
            devices=devices,
            ddpmodel=ddpmodel,
            batchsizes=batchsizes,
            optim=optim,
            assembler=assembler,
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
    # checkpoint last state
    utils.save_checkpoint(
        command_history=command_history,
        epoch=epoch,
        model=models[0],
        optim=optim,
        assembler=assembler,
        game_params=game_params,
        model_params=model_params,
        optim_params=optim_params,
        simulation_params=simulation_params,
        execution_params=execution_params,
    )

def client_loop(
    assembler: tube.ChannelAssembler,
    start_time: float,
    context: tube.Context,
    execution_params: ExecutionParams,
) -> None:
    assembler.start()
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
    train_devices = [torch.device(i) for i in execution_params.device_train]
    game_generation_devices = [torch.device(i) for i in execution_params.device_eval]

    models = []
    for device in train_devices:
      models.append(create_model(
          game_params=game_params,
          model_params=model_params,
          resume_training=bool(checkpoint),
          model_state_dict=checkpoint["model_state_dict"] if checkpoint else None,
      ).to(device))

    model_path = execution_params.checkpoint_dir / "model.pt"
    models[0].save(str(model_path))

    ddpmodel = None
    #if execution_params.ddp:
    if os.environ.get("RANK") is not None:
        torch.distributed.init_process_group(backend="gloo", timeout=datetime.timedelta(0, 86400))
        ddpmodel = nn.parallel.DistributedDataParallel(ModelWrapperForDDP(models[0]), broadcast_buffers=False, find_unused_parameters=False)

    print("creating optimizer...")
    optim = create_optimizer(
        model=ddpmodel if ddpmodel is not None else models[0],
        optim_params=optim_params,
        optim_state_dict=checkpoint.get("optim_state_dict", None),
    )

    rnn_state_shape = []
    if hasattr(models[0], "rnn_cells") and models[0].rnn_cells > 0:
      rnn_state_shape = [models[0].rnn_cells, models[0].rnn_channels]

    print("creating training environment...")
    context, assembler, get_train_reward, is_client = create_training_environment(
        seed_generator=seed_generator,
        model_path=model_path,
        game_generation_devices=game_generation_devices,
        game_params=game_params,
        simulation_params=simulation_params,
        execution_params=execution_params,
        rnn_state_shape=rnn_state_shape
    )
    if not is_client:
        assembler.update_model(models[0].state_dict())
    assembler.add_tournament_model("init", models[0].state_dict())
    context.start()

    if is_client:
      client_loop(
          assembler=assembler,
          start_time=start_time,
          context=context,
          execution_params=execution_params
      )
    else:
      if ddpmodel is None or torch.distributed.get_rank() == 0:
        print("warming-up replay buffer...")
        warm_up_replay_buffer(
            assembler=assembler,
            replay_warmup=simulation_params.replay_warmup,
            replay_buffer=checkpoint.get("replay_buffer", None),
        )

      print("training model...")
      train_model(
          command_history=command_history,
          start_time=start_time,
          models=models,
          devices=train_devices,
          ddpmodel=ddpmodel,
          optim=optim,
          context=context,
          assembler=assembler,
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
