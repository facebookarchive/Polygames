# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple, Optional

from . import utils as zutils
from ..params import GameParams, ModelParams
from .. import utils

from . import vtrace

@zutils.register_model
class ResConvGruConvLogitPoolModel(torch.jit.ScriptModule):
    __constants__ = [
        "c_prime",
        "h_prime",
        "w_prime",
        "nb_layers_per_net",
    ]

    DEFAULT_NB_NETS = 5
    DEFAULT_NB_LAYERS_PER_NET = 3
    DEFAULT_NNSIZE = 2
    DEFAULT_NNKS = 3
    DEFAULT_STRIDE = 1
    DEFAULT_DILATION = 1
    DEFAULT_POOLING = False
    DEFAULT_BN = False
    # DEFAULT_BN_AFFINE = False

    default_game_name = "Hex13"

    def __init__(self, game_params: GameParams, model_params: ModelParams):
        torch.jit.ScriptModule.__init__(self)
        if game_params.game_name is None:
            game_params.game_name = self.__class__.default_game_name
        self.game_name = game_params.game_name
        self.game_params = game_params
        info = zutils.get_game_info(game_params)
        c, h, w = self.c, self.h, self.w = info["feature_size"][:3]
        r_c, r_h, r_w = info["raw_feature_size"]
        c_prime, h_prime, w_prime = self.c_prime, self.h_prime, self.w_prime = info[
            "action_size"
        ][:3]
        if h_prime != h or w_prime != w:
            raise RuntimeError(
                f'The game "{self.game_name}" is not eligible to a conv-computed logit '
                f'model such as "{self.__class__.__name__}" - try with '
                f'"{self.__class__.__name__.replace("ConvLogit", "FCLogit")}" instead'
            )

        # nb resnets
        if model_params.nb_nets is None:
            model_params.nb_nets = self.DEFAULT_NB_NETS
        nb_nets = model_params.nb_nets
        # nb layers per resnet
        if model_params.nb_layers_per_net is None:
            model_params.nb_layers_per_net = self.DEFAULT_NB_LAYERS_PER_NET
        nb_layers_per_net = model_params.nb_layers_per_net
        self.nb_layers_per_net = nb_layers_per_net
        # nn size
        if model_params.nnsize is None:
            model_params.nnsize = self.DEFAULT_NNSIZE
        nnsize = model_params.nnsize
        # kernel size
        if model_params.nnks is None:
            model_params.nnks = self.DEFAULT_NNKS
        nnks = model_params.nnks
        # stride
        stride = self.DEFAULT_STRIDE
        # dilation
        dilation = self.DEFAULT_DILATION
        # padding
        padding = zutils.get_consistent_padding_from_nnks(nnks=nnks, dilation=dilation)
        # pooling
        if model_params.pooling is None:
            model_params.pooling = self.DEFAULT_POOLING
        pooling = model_params.pooling
        # batch norm
        if model_params.bn is None:
            model_params.bn = self.DEFAULT_BN
        bn = model_params.bn
        # # batch norm affine
        # if model_params.bn_affine is None:
        #     model_params.bn_affine = self.DEFAULT_BN_AFFINE
        # bn_affine = model_params.bn_affine
        bn_affine = bn
        self.model_params = model_params

        rnn_interval = model_params.rnn_interval
        self.global_pooling = model_params.global_pooling
        if model_params.activation_function == "relu":
          self.af = F.relu
        elif model_params.activation_function == "gelu":
          self.af = F.gelu
        elif model_params.activation_function == "celu":
          self.af = F.celu
        else:
          raise RuntimeError("Unknown activation function")
        batchnorm_momentum = model_params.batchnorm_momentum

        self.predict_end_state = game_params.predict_end_state
        self.predict_n_states = game_params.predict_n_states

        self.predicts = self.predict_n_states + (2 if self.predict_end_state else 0)

        print("GRU net")
        print("global pooling ", self.global_pooling)
        print("af ", model_params.activation_function)

        mono = [
            nn.Conv2d(
                c,
                int(nnsize * c),
                nnks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not bn_affine,
            )
        ]

        resnet_list = []
        for i in range(nb_nets):
            nets = [
                nn.Conv2d(
                    int(nnsize * c) + int(nnsize * c * (self.global_pooling if _ == 0 else 0)) * 2,
                    int(nnsize * c),
                    nnks,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=not bn_affine,
                )
                for _ in range(nb_layers_per_net)
            ]
            if bn or bn_affine:
                for j in range(nb_layers_per_net):
                    nets[j] = nn.Sequential(
                        nets[j],
                        nn.BatchNorm2d(
                            int(nnsize * c), track_running_stats=True, affine=bn_affine, momentum=batchnorm_momentum
                        ),
                    )
            if pooling:
                for j in range(nb_layers_per_net):
                    nets[j] = nn.Sequential(
                        nets[j],
                        nn.MaxPool2d(
                            kernel_size=nnks,
                            padding=padding,
                            stride=stride,
                            dilation=dilation,
                        ),
                    )
            resnet_list.append(nets)
        if bn or bn_affine:
            mono.append(
                nn.BatchNorm2d(int(nnsize * c), track_running_stats=True, affine=bn_affine, momentum=batchnorm_momentum),
            )
        rnni = 0
        self.rnn_cells = 0
        self.rnn_interval = 0
        self.flat_rnn = True
        rnnlist = []
        for i in range(nb_nets):
            resnet_list[i] = nn.ModuleList(resnet_list[i])
            list = []
            if rnn_interval > 0:
              rnni += 1
              while rnni >= rnn_interval:
                  rnni -= rnn_interval
                  if self.flat_rnn:
                      list.append(nn.GRUCell(int(nnsize * c) * h * w, int(nnsize * c) * h * w))
                  else:
                      list.append(nn.GRUCell(int(nnsize * c), int(nnsize * c)))
                  self.rnn_cells += 1
            rnnlist.append(nn.ModuleList(list))
        self.rnnlist = nn.ModuleList(rnnlist)
        self.rnn_channels = int(nnsize * c)
        print("model has %d GRU layers of size %d" % (self.rnn_cells, self.rnn_channels))
        self.mono = nn.Sequential(*mono)
        self.resnets = nn.ModuleList(resnet_list)
        self.v = nn.Linear(2 * int(nnsize * c), 2 * int(nnsize * c))
        self.v_bn = nn.BatchNorm1d(2 * int(nnsize * c), momentum=batchnorm_momentum)
        self.v2 = nn.Linear(2 * int(nnsize * c), 1)
        self.logit_value = False
        self.pi_logit = nn.Conv2d(
            int(nnsize * c), c_prime, nnks, stride=stride, padding=padding, dilation=dilation
        )
        if self.predicts > 0:
          self.predict_pi_logit = nn.Conv2d(
              int(nnsize * c), r_c * self.predicts, nnks, stride=stride, padding=padding, dilation=dilation
          )
        else:
          self.predict_pi_logit = None

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, rnn_state: torch.Tensor, rnn_state_mask: Optional[torch.Tensor], return_logit: bool):
        # x is B, S, C, H, W
        # state is B, C, H, W
        # rnn_state_mask is B, S
        af = self.af
        global_pooling = self.global_pooling
        gruindex = 0
        states = []
        if x.dim() == 4:
          x = x.unsqueeze(1)
        B = x.size(0)
        S = x.size(1)
        x = x.flatten(0, 1)
        previous_block = self.mono(x)  # linear transformation only
        for resnet, rnns in zip(self.resnets, self.rnnlist):
            sublayer_no = 0
            h = previous_block
            if global_pooling > 0:
                hpart = h.narrow(1, 0, int(h.size(1) * global_pooling))
                h = torch.cat((h, F.adaptive_max_pool2d(hpart, 1).expand_as(hpart), F.adaptive_avg_pool2d(hpart, 1).expand_as(hpart)), 1)
            h = af(h)  # initial activation
            for net in resnet:
                if sublayer_no < self.nb_layers_per_net - 1:
                    h = af(net(h))
                else:
                    h = net(h) + previous_block  #  linear transformation only
                    previous_block = h
                sublayer_no = sublayer_no + 1
            for net in rnns:
                if self.flat_rnn:
                  C = h.size(1)
                  H = h.size(2)
                  W = h.size(3)

                  # B * S, C, H, W
                  h = h.view(B, S, C, H, W)
                  # B, S, C, H, W
                  h = h.transpose(0, 1).flatten(2)
                  # S, B, C * H * W
                  sx = rnn_state.select(1, gruindex).view(B, C * H * W)
                  hout = []
                  if rnn_state_mask is None:
                    for s in range(S):
                      #print("in sx.sum() is ", sx.sum())
                      sx = net(h[s], sx)
                      #print("out sx.sum() is ", sx.sum())
                      hout.append(sx)
                  else:
                    for s in range(S):
                      sx = sx * rnn_state_mask[:, s].view(B, 1)
                      #print("in sx.sum() is ", sx.sum())
                      sx = net(h[s], sx)
                      #print("out sx.sum() is ", sx.sum())
                      hout.append(sx)
                  h = torch.stack(hout)
                  # S, B, C * H * W
                  h = h.view(S, B, C, H, W)
                  # S, B, C, H, W
                  h = h.transpose(0, 1).flatten(0, 1)
                  # B * S, C, H, W
                  previous_block = h

                  gruindex += 1
                  states.append(sx.view(B, C, H, W))
                else:
                  C = h.size(1)
                  H = h.size(2)
                  W = h.size(3)
  
                  # B * S, C, H, W
                  h = h.view(B, S, C, H, W)
                  # B, S, C, H, W
                  h = h.permute(1, 0, 3, 4, 2)
                  # S, B, H, W, C
                  h = h.flatten(1, 3).contiguous()
                  # S, B * H * W, C
  
                  sx = rnn_state.select(1, gruindex)
                  # B, C, H, W
                  sx = sx.permute(0, 2, 3, 1)
                  # B, H, W, C
                  sx = sx.flatten(0, 2).contiguous()
                  # B * H * W, C

                  hout = []
                  if rnn_state_mask is None:
                    for s in range(S):
                      print("in sx.sum() is ", sx.sum())
                      sx = net(h[s], sx)
                      print("out sx.sum() is ", sx.sum())
                      hout.append(sx)
                  else:
                    for s in range(S):
                      sx = (sx.view(B, H, W, C) * rnn_state_mask[:, s].view(B, 1, 1, 1)).flatten(0, 2)
                      sx = net(h[s], sx)
                      hout.append(sx)
                  h = torch.stack(hout)
                  # S, B * H * W, C
                  h = h.view(S, B, H, W, C)
                  # S, B, H, W, C
                  h = h.permute(1, 0, 4, 2, 3)
                  # B, S, C, H, W
                  h = h.flatten(0, 1).contiguous()
                  # B * S, C, H, W
                  previous_block = h
  
                  # B * H * W, C
                  sx = sx.view(B, H, W, C)
                  # B, H, W, C
                  sx = sx.permute(0, 3, 1, 2).contiguous()
                  # B, C, H, W
  
                  gruindex += 1
                  states.append(sx.view(B, C, H, W))
        if gruindex != len(states) or gruindex != self.rnn_cells:
            raise RuntimeError("GRU count mismatch")
        h = af(previous_block)  # final activation
        pool = torch.cat((F.adaptive_max_pool2d(h, 1).flatten(-2), F.adaptive_avg_pool2d(h, 1).flatten(-2)), -1).flatten(-2)
        pi_logit = self.pi_logit(h)
        v = af(self.v_bn(self.v(pool)))
        #v = af(self.v(pool))
        v = self.v2(v)
        if return_logit:
            if self.predict_pi_logit is not None:
                predict_pi_logit = self.predict_pi_logit(h)
                return v, pi_logit, predict_pi_logit, torch.stack(states, 1)
            else:
                return v, pi_logit, None, torch.stack(states, 1)
        raise RuntimeError("bad")
        s = pi_logit.shape
        pi = F.softmax(pi_logit, 1).reshape(s)
        v = F.softmax(v.float(), 1)
        return v, pi, None, torch.stack(states, 1)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, rnnState: torch.Tensor):
        v, pi, _, state = self._forward(x, rnnState, None, True)
        #pi = pi.view(-1, self.c_prime, x.size(2), x.size(3))
        reply = {"v": v, "pi_logit": pi, "rnn_state": state}
        return reply

    def loss(
        self,
        model,
        x: torch.Tensor,
        rnn_initial_state: torch.Tensor,
        rnn_state_mask: torch.Tensor,
        v: torch.Tensor,
        pi: torch.Tensor,
        pi_mask: torch.Tensor,
        predict_pi: torch.Tensor = None,
        predict_pi_mask: torch.Tensor = None,
        #stat: utils.MultiCounter,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # tensor shapes are [batch, sequence, ...]
        # if no rnn then it is simply [batch, ...]

        bs = x.size(0)
        seqlen = x.size(1)

        result = []
        #state = torch.zeros([bs, self.rnn_cells, self.rnn_channels, x.size(3), x.size(4)], device=x.device)
        rnn_initial_state = rnn_initial_state.view(bs, self.rnn_cells, self.rnn_channels, x.size(3), x.size(4))

#        pred_vs = []
#        pred_logits = []
#        pred_predict_logits = []
#        for i in range(seqlen):
#          pred_v, pred_logit, pred_predict_logit, state = model._forward(x[:, i], state * rnn_state_mask[:, i].view(bs, 1, 1, 1, 1), return_logit=True)
#          pred_vs.append(pred_v)
#          pred_logits.append(pred_logit)
#          pred_predict_logits.append(pred_predict_logit)

#        pred_v = torch.stack(pred_vs, 1).flatten(0, 1)
#        pred_logit = torch.stack(pred_logits, 1).flatten(0, 1)
#        pred_predict_logit = torch.stack(pred_predict_logits, 1).flatten(0, 1)

        pred_v, pred_logit, pred_predict_logit, state = model._forward(x, rnn_initial_state, rnn_state_mask, return_logit=True)

        #pred_v = pred_v.flatten(0, 1)
        #pred_logit = pred_logit.flatten(0, 1)
        #pred_predict_logit = pred_predict_logit.flatten(0, 1)

        v = v.flatten(0, 1)
        pi = pi.flatten(0, 1)
        pi_mask = pi_mask.flatten(0, 1)

        pi = pi.flatten(1)

        pi_mask = pi_mask.view(pred_logit.shape);
        pred_logit = pred_logit * pi_mask - 400 * (1 - pi_mask)
        if self.predicts > 0:
          predict_pi = predict_pi.flatten(0, 1)
          predict_pi_mask = predict_pi_mask.flatten(0, 1)
          predict_pi_err = (F.mse_loss(pred_predict_logit, predict_pi, reduction="none") * predict_pi_mask).flatten(2).sum(2).flatten(1).mean(1)

        v_err = -(nn.functional.log_softmax(pred_v, dim=1) * v).sum(1)
        pred_log_pi = nn.functional.log_softmax(pred_logit.flatten(1), dim=1).view_as(pred_logit) * pi_mask
        pi_err = -(pred_log_pi * pi).sum(1)

        #pi_err = (-((1 - pred_log_pi.exp().clamp_max(0.9999)).log() * -pi.clamp_max(0)) + -(pred_log_pi * pi.clamp_min(0))).sum(1)

        #err = v_err + pi_err + (predict_pi_err * 0.15 if self.predicts > 0 else 0)
        err = state.flatten(1).sum(1).unsqueeze(1).expand(bs, seqlen).flatten(0, 1) * 0.0 + v_err * 1.5 + pi_err + (predict_pi_err * 0.1 if self.predicts > 0 else 0)

        return err.mean(), v_err.detach().mean(), pi_err.detach().mean(), (predict_pi_err.detach().mean() if self.predicts > 0 else None)

    def compute_baseline_loss(self, advantages):
        return 0.5 * advantages ** 2


    def compute_entropy_loss(self, logits, mask):
        """Return the entropy loss, i.e., the negative entropy of the policy."""
        policy = F.softmax(logits * mask - 400 * (1 - mask), dim=-1)
        log_policy = F.log_softmax(logits * mask - 400 * (1 - mask), dim=-1)
        return policy * log_policy * mask


    def compute_policy_gradient_loss(self, logits, mask, actions, advantages):
        mask = mask.flatten(0, 1)
        cross_entropy = F.nll_loss(
            F.log_softmax(torch.flatten(logits, 0, 1) * mask - 400 * (1 - mask), dim=-1),
            target=torch.flatten(actions, 0, 1),
            reduction="none",
        )
        cross_entropy = cross_entropy.view_as(advantages)
        return cross_entropy * advantages.detach()

    def loss2(
        self,
        model,
        batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = batch["s"]
        v = batch["v"]
        pi = batch["pi"]
        pi_mask = batch["pi_mask"]
        predict_pi = batch["predict_pi"] if self.predicts > 0 else None
        predict_pi_mask = batch["predict_pi_mask"] if self.predicts > 0 else None
        action_pi = batch["action_pi"]
        rnn_initial_state = batch["rnn_initial_state"]
        rnn_state_mask = batch["rnn_state_mask"]

        # tensor shapes are [batch, sequence, ...]
        # if no rnn then it is simply [batch, ...]

        bs = x.size(0)
        seqlen = x.size(1)

        result = []
        rnn_initial_state = rnn_initial_state.view(bs, self.rnn_cells, self.rnn_channels, x.size(3), x.size(4))

        pred_v, pred_logit, pred_predict_logit, state = model._forward(x, rnn_initial_state, rnn_state_mask, return_logit=True)

        #print("pi is ", pi.shape)
        #print("pred_logit is ", pred_logit.shape)
        #print("behavior_pred_v is ", batch["pred_v"].shape)
        #print("pred_v is ", pred_v.shape)

        #print("pi - pred_logit is ", (pi.flatten() - pred_logit.flatten()).sum())

        #v = v.flatten(0, 1)
        #pi = pi.flatten(0, 1)
        #pi_mask = pi_mask.flatten(0, 1)
        #action_pi = action_pi.flatten(0, 1)

        #pi_mask = pi_mask.view(pred_logit.shape);
        #pred_logit = pred_logit * pi_mask - 400 * (1 - pi_mask)
        #pi = pi * pi_mask - 400 * (1 - pi_mask)
        if self.predicts > 0:
          predict_pi = predict_pi.flatten(0, 1)
          predict_pi_mask = predict_pi_mask.flatten(0, 1)
          predict_pi_err = (F.mse_loss(pred_predict_logit, predict_pi, reduction="none") * predict_pi_mask).flatten(2).sum(2).flatten(1).mean(1)

        #print("pi is ", pi)
        #print("pred_logit is ", pred_logit)

        if True:

          device = v.device

          batch_policy_logits = pi.transpose(0, 1).flatten(2)
          learner_outputs_policy_logits = pred_logit.view(bs, seqlen, -1).transpose(0, 1).flatten(2)
          batch_action = action_pi.transpose(0, 1).flatten(2).max(-1).indices
          #discounts = torch.zeros(seqlen, bs).to(device)
          discounts = rnn_state_mask.transpose(0, 1).squeeze(-1) * 0
          clipped_rewards = v.transpose(0, 1).squeeze(-1)
          learner_outputs_baseline = pred_v.view(bs, seqlen, 1).transpose(0, 1).squeeze(-1)
          bootstrap_value = clipped_rewards[-1]

          policy_mask = pi_mask.transpose(0, 1).flatten(2)

          #batch_policy_logits = batch_policy_logits[1:]
          #batch_action = batch_action[1:]
          #clipped_rewards = clipped_rewards[1:]

          #learner_outputs_policy_logits = learner_outputs_policy_logits[:-1]
          #learner_outputs_baseline = learner_outputs_baseline[:-1]

          #print("batch_policy_logits is ", batch_policy_logits)
          #print("learner_outputs_policy_logits is ", learner_outputs_policy_logits)
          #print("batch_policy_logits - learner_outputs_policy_logits is ", (batch_policy_logits - learner_outputs_policy_logits).sum())

          #print("clipped_rewards.shape is ", clipped_rewards.shape)
          #print("bootstrap_value.shape is ", bootstrap_value.shape)
          #print("learner_outputs_baseline.shape is ", learner_outputs_baseline.shape)

          vtrace_returns = vtrace.from_logits(
              behavior_policy_logits=batch_policy_logits,
              target_policy_logits=learner_outputs_policy_logits,
              policy_mask=policy_mask,
              actions=batch_action,
              discounts=discounts,
              rewards=clipped_rewards,
              values=learner_outputs_baseline,
              bootstrap_value=bootstrap_value,
          )

          pi_err = self.compute_policy_gradient_loss(
              learner_outputs_policy_logits,
              policy_mask,
              batch_action,
              vtrace_returns.pg_advantages,
          ).view(seqlen, bs, -1).transpose(0, 1).flatten(0, 1).squeeze(1)
          v_err = self.compute_baseline_loss(
              vtrace_returns.vs - learner_outputs_baseline
          ).transpose(0, 1).flatten(0, 1)
          entropy =  self.compute_entropy_loss(
              learner_outputs_policy_logits, policy_mask
          ).transpose(0, 1).flatten(0, 1).sum(1)

          #print("vtrace_returns.vs is", vtrace_returns.vs)

          #print("pi_err.shape is ", pi_err.shape)
          #print("v_err.shape is ", v_err.shape)
          #print("entropy.shape is ", entropy.shape)

          #trad_v_err = F.mse_loss(pred_v, v.flatten(0, 1), reduction="none")

          #print("v avg: %f min: %f max: %f" % (v.mean().item(), v.min().item(), v.max().item()))
          #print("pred_v avg: %f min: %f max: %f" % (pred_v.mean().item(), pred_v.min().item(), pred_v.max().item()))
          #print("trad_v_err avg: %f min: %f max: %f" % (trad_v_err.mean().item(), trad_v_err.min().item(), trad_v_err.max().item()))

          #print("policy   avg: %f min: %f max: %f" % (pi_err.mean().item(), pi_err.min().item(), pi_err.max().item()))
          #print("baseline avg: %f min: %f max: %f" % (v_err.mean().item(), v_err.min().item(), v_err.max().item()))
          #print("entropy  avg: %f min: %f max: %f" % (entropy.mean().item(), entropy.min().item(), entropy.max().item()))

          #raise RuntimeError("stop")

        elif True:

          v = v.flatten(0, 1)
          pi = pi.flatten(0, 1)
          pi_mask = pi_mask.flatten(0, 1)
          action_pi = action_pi.flatten(0, 1)

          pred_log_pi = pred_logit.flatten(1).log_softmax(1)
          ratio = ((pred_log_pi - pi.flatten(1).log_softmax(1).detach()) * action_pi.flatten(1)).sum(1).exp()

          #print("pred ", (pred_log_pi * action_pi.flatten(1)).sum(1))
          #print("pi ", (pi.flatten(1).log_softmax(1) * action_pi.flatten(1)).sum(1))

          #reward = v.narrow(1, 0, 1) - v.narrow(1, 1, 1);
          ##reward = v.narrow(1, 0, 1);
          pred_v_sf = pred_v.softmax(1)
          #pred_v_reward = pred_v_sf.narrow(1, 0, 1) - pred_v_sf.narrow(1, 1, 1)
          pred_v_reward = pred_v_sf.narrow(1, 0, 1)
          #advantage = (v.narrow(1, 0, 1) - pred_v_reward.detach()).squeeze(-1)
          advantage = v.squeeze(-1) - batch["pred_v"].flatten(0, 1).squeeze(-1)
          #advantage = (v - pred_v.detach()).squeeze(-1)

          #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

          #print("v is ", v)

          #print("reward is ", reward)
          #print("pred_v_reward is ", pred_v_reward)
          #print("advantage is ", advantage)

          print("ratio avg: %f min: %f max: %f" % (ratio.mean().item(), ratio.min().item(), ratio.max().item()))
          print("advantage avg: %f min: %f max: %f" % (advantage.mean().item(), advantage.min().item(), advantage.max().item()))

          #raise RuntimeError("stop")

          ppo_clip = 0.2

          #print("advantage shape is ", advantage.shape)
          #print("ratio shape is ", ratio.shape)

          surr1 = ratio * advantage
          surr2 = ratio.clamp(1 - ppo_clip, 1 + ppo_clip) * advantage

          #print("surr1 shape is ", surr1.shape)

          entropy = (-pred_log_pi * pred_log_pi.exp()).sum(1)

          pi_err = -torch.min(surr1, surr2)
          v_err = F.mse_loss(pred_v, v, reduction="none")
          #v_err = -(nn.functional.log_softmax(pred_v, dim=1) * v).sum(1)

          #print("pi_err is ", pi_err)
          print("pi_err avg: %f min: %f max: %f" % (pi_err.mean().item(), pi_err.min().item(), pi_err.max().item()))
          print("entropy avg: %f min: %f max: %f" % (entropy.mean().item(), entropy.min().item(), entropy.max().item()))

          #raise RuntimeError("stop")
        else:
          advantage = v - pred_v.detach()
          v_err = F.mse_loss(pred_v, v, reduction="none")
          pred_log_pi = nn.functional.log_softmax(pred_logit.flatten(1), dim=1).view_as(pred_logit) * pi_mask
          pi_err = -(pred_log_pi.flatten(1) * action_pi.flatten(1) * v.clamp(0, 1)).sum(1)

        #pi_err = (-((1 - pred_log_pi.exp().clamp_max(0.9999)).log() * -pi.clamp_max(0)) + -(pred_log_pi * pi.clamp_min(0))).sum(1)

        err = state.flatten(1).sum(1).unsqueeze(1).expand(bs, seqlen).flatten(0, 1).sum() * 0.0 + v_err * 0.5 + pi_err - 0.01 * entropy + (predict_pi_err * 0.1 if self.predicts > 0 else 0)

        #print("err is ", err)

        return err.mean(), v_err.detach().mean(), pi_err.detach().mean(), (predict_pi_err.detach().mean() if self.predicts > 0 else None)
