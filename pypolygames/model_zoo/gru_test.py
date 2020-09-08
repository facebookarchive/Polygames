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

class GRUMasked(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        torch.nn.Module.__init__(self)
        self.gru = nn.GRUCell(input_size, hidden_size)
    def forward(self, x, hx, mask: Optional[torch.Tensor]):
        B = x.size(0)
        S = x.size(1)
        hout = []
        for s in range(S):
            if mask is not None:
                hx = hx * mask.select(1, s)
            hx = self.gru(x.select(1, s), hx)
            hout.append(hx)
        h = torch.stack(hout, 1)
        return h, hx

class ConvBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        torch.nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=True)
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        #self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return self.conv(x)
        #return self.bn(self.conv(x))

class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, hidden_layers, af=F.celu):
        torch.nn.Module.__init__(self)
        self.af = af
        self.cin = ConvBN(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.conv = nn.ModuleList([ConvBN(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels) for _ in range(hidden_layers)])
        self.cout = ConvBN(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        if in_channels != out_channels:
            self.resconv = ConvBN(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.out_channels = out_channels
    def forward(self, x):
        B = x.size(0)
        S = x.size(1)
        C = x.size(2)
        H = x.size(3)
        W = x.size(4)

        x = x.flatten(0, 1)
        h = self.af(self.cin(x), inplace=True)
        for net in self.conv:
            h = self.af(net(h), inplace=True)
        h = self.cout(h)
        if hasattr(self, "resconv"):
            x = self.resconv(x)
        return self.af(h + x, inplace=True).view(B, S, self.out_channels, H, W)

class GruBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, hidden_layers, gru_channels, gru_size, af=F.celu):
        torch.nn.Module.__init__(self)
        self.af = af
        self.cin = ConvBN(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.conv = nn.ModuleList([ConvBN(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels) for _ in range(hidden_layers)])
        self.gru_channels = gru_channels
        self.gru_size = gru_size
        self.gruin = ConvBN(hidden_channels, gru_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.grunet = GRUMasked(gru_size, gru_size)
        self.gruout = ConvBN(gru_channels, hidden_channels, kernel_size=1, stride=1, padding=0, groups=1)
        #self.gruout = nn.Linear(gru_size, hidden_channels)
        self.cout = ConvBN(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        if in_channels != out_channels:
            self.resconv = ConvBN(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.out_channels = out_channels
    def forward(self, x, state, mask:Optional[torch.Tensor]=None):
        B = x.size(0)
        S = x.size(1)
        C = x.size(2)
        H = x.size(3)
        W = x.size(4)

        x = x.flatten(0, 1)
        h = self.af(self.cin(x), inplace=True)
        for net in self.conv:
            h = self.af(net(h), inplace=True)
        g = self.af(self.gruin(h), inplace=True)
        g, hg = self.grunet(g.view(B, S, self.gru_size), state, mask)
        h = self.gruout(g.view(B * S, self.gru_channels, h.size(2), h.size(3)))
        #h = h * g.view(B * S, h.size(1), 1, 1).sigmoid()

        h = self.cout(h)
        if hasattr(self, "resconv"):
            x = self.resconv(x)
        return self.af(h + x, inplace=True).view(B, S, self.out_channels, H, W), hg

@zutils.register_model
class GruTest(torch.jit.ScriptModule):
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

        self.gru_channels = 4
        self.gru_size = self.gru_channels * h * w
        self.gru_num = 1
        self.rnn_state_shape = [self.gru_num, self.gru_size]
        self.predicts = 0

        layers = 6

        hc = 512
        self.sc = sc = 128

        blocklist = []
        for i in range(layers):
            blocklist.append(Block(in_channels=c if i == 0 else sc, out_channels=sc, hidden_channels=hc, hidden_layers=2, af=F.celu))
        self.blocks = nn.ModuleList(blocklist)
        self.grublock = GruBlock(in_channels=sc, out_channels=sc, hidden_channels=hc, hidden_layers=2, gru_channels=self.gru_channels, gru_size=self.gru_size, af=F.celu)
        blocklist = []
        for i in range(2):
            blocklist.append(Block(in_channels=sc, out_channels=sc, hidden_channels=hc, hidden_layers=2, af=F.celu))
        self.blocks2 = nn.ModuleList(blocklist)
        self.v = nn.Linear(sc * 2, 1)
        self.logit = nn.Conv2d(sc, c_prime, kernel_size=3, stride=1, padding=1)

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, rnn_state: torch.Tensor, rnn_state_mask: Optional[torch.Tensor], return_logit: bool):
        # x is B, S, C, H, W
        # rnn_state is B, N, C, H, W
        # rnn_state_mask is B, S

        #print("input x is ", x.shape)

        if x.dim() == 4:
          x = x.unsqueeze(1)
        B = x.size(0)
        S = x.size(1)
        C = x.size(2)
        H = x.size(3)
        W = x.size(4)

        #print(B, S, C, H, W, x.shape)

        #print("rnn_state_mask is ", rnn_state_mask)
        #print("rnn_state is ", rnn_state)

        rnn_state_mask = rnn_state_mask.view(B, S, 1) if rnn_state_mask is not None else None

        states = []
        index = 0
        h = x
        #for net in self.blocks:
        #  h, hx = net(h, rnn_state.select(1, index), rnn_state_mask)
        #  states.append(hx)
        #  index += 1
        for net in self.blocks:
            h = net(h)
        #print("rnn_state.sum() is ", rnn_state.sum())
        h, hx = self.grublock(h, rnn_state.select(1, 0), rnn_state_mask)
        #print("hx.sum() is ", hx.sum())
        states.append(hx)
        for net in self.blocks2:
            h = net(h)

        h = h.view(B * S, self.sc, H, W)
        pool = torch.cat((F.adaptive_max_pool2d(h, 1).flatten(-2), F.adaptive_avg_pool2d(h, 1).flatten(-2)), -1).flatten(-2)

        #print(h.min())
        #print(h.max())

        v = self.v(pool)
        logit = self.logit(h)

        #print(logit.min())
        #print(logit.max())

        #print(logit)

        allstates = []

        if return_logit:
            return v, logit, None, torch.stack(states, 1), allstates
        raise RuntimeError("bad")

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, rnnState: torch.Tensor):
        v, pi, _, state, _ = self._forward(x, rnnState, None, True)
        #pi = pi.view(-1, self.c_prime, x.size(2), x.size(3))
        reply = {"v": v, "pi_logit": pi, "rnn_state": state}
        return reply

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
        #rnn_initial_state = rnn_initial_state.view(bs, self.gru_num, self.gru_size)

        pred_v, pred_logit, pred_predict_logit, state, allstates = model._forward(x, rnn_initial_state, rnn_state_mask, return_logit=True)

        #for b in range(bs):
        #  print("mask ", rnn_state_mask.select(0, b))
        #  print("logit ", pred_logit.view(bs, seqlen, -1).select(0, b))

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

        if False:

          device = v.device

          batch_policy_logits = pi.transpose(0, 1).flatten(2)
          learner_outputs_policy_logits = pred_logit.view(bs, seqlen, -1).transpose(0, 1).flatten(2)
          batch_action = action_pi.transpose(0, 1).flatten(2).max(-1).indices
          #discounts = torch.zeros(seqlen, bs).to(device)
          discounts = rnn_state_mask.transpose(0, 1).squeeze(-1) * 0.99
          clipped_rewards = v.transpose(0, 1).squeeze(-1)
          learner_outputs_baseline = pred_v.view(bs, seqlen, 1).transpose(0, 1).squeeze(-1)
          #bootstrap_value = clipped_rewards[-1]
          bootstrap_value = learner_outputs_baseline[-1]

          #clipped_rewards = (clipped_rewards - clipped_rewards.mean()) / (clipped_rewards.std() + 1e-5)

          policy_mask = pi_mask.transpose(0, 1).flatten(2)

          #policy_mask = policy_mask[1:]
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
          entropy = self.compute_entropy_loss(
              learner_outputs_policy_logits, policy_mask
          ).transpose(0, 1).flatten(0, 1).sum(1)

          #print("vtrace_returns.vs is", vtrace_returns.vs)

          #print("pi_err.shape is ", pi_err.shape)
          #print("v_err.shape is ", v_err.shape)
          #print("entropy.shape is ", entropy.shape)

          trad_v_err = F.mse_loss(pred_v, v.flatten(0, 1), reduction="none")

          print("v avg: %f min: %f max: %f" % (v.mean().item(), v.min().item(), v.max().item()))
          print("pred_v avg: %f min: %f max: %f" % (pred_v.mean().item(), pred_v.min().item(), pred_v.max().item()))
          print("trad_v_err avg: %f min: %f max: %f" % (trad_v_err.mean().item(), trad_v_err.min().item(), trad_v_err.max().item()))

          print("policy   avg: %f min: %f max: %f" % (pi_err.mean().item(), pi_err.min().item(), pi_err.max().item()))
          print("baseline avg: %f min: %f max: %f" % (v_err.mean().item(), v_err.min().item(), v_err.max().item()))
          print("entropy  avg: %f min: %f max: %f" % (entropy.mean().item(), entropy.min().item(), entropy.max().item()))

          #raise RuntimeError("stop")

        elif True:

          v = v.flatten(0, 1)
          pi = pi.flatten(0, 1)
          pi_mask = pi_mask.flatten(0, 1)
          action_pi = action_pi.flatten(0, 1)

          mask = pi_mask.flatten(1)

          pred_log_pi = (pred_logit.flatten(1) * mask - 400 * (1 - mask)).log_softmax(1) * mask
          pi_log_pi = (pi.flatten(1) * mask - 400 * (1 - mask)).log_softmax(1).detach() * mask
          ratio = ((pred_log_pi - pi_log_pi) * action_pi.flatten(1)).sum(1).exp()

          #s = ((1 - mask) * action_pi.flatten(1)).sum()
          #if s != 0:
          #  raise RuntimeError("sum is " + str(s))

          #print("pred ", (pred_log_pi * action_pi.flatten(1)).sum(1))
          #print("pi ", (pi.flatten(1).log_softmax(1) * action_pi.flatten(1)).sum(1))

          #reward = v.narrow(1, 0, 1) - v.narrow(1, 1, 1);
          ##reward = v.narrow(1, 0, 1);
          #pred_v_sf = pred_v.softmax(1)
          #pred_v_reward = pred_v_sf.narrow(1, 0, 1) - pred_v_sf.narrow(1, 1, 1)
          #pred_v_reward = pred_v_sf.narrow(1, 0, 1)
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

          entropy = -(-pred_log_pi * pred_log_pi.exp() * pi_mask.flatten(1)).sum(1)

          pi_err = -torch.min(surr1, surr2)
          #v_err = 2 * F.mse_loss(pred_v, v, reduction="none")
          #v_err = -(nn.functional.log_softmax(pred_v, dim=1) * v).sum(1)
          v_err = (v - pred_v).pow(2)

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

        #err = state.flatten(1).sum(1).unsqueeze(1).expand(bs, seqlen).flatten(0, 1).sum() * 0.0 + v_err * 0.5 + pi_err - 0.01 * entropy + (predict_pi_err * 0.1 if self.predicts > 0 else 0)
        err = v_err * 0.5 + pi_err + 0.001 * entropy + (predict_pi_err * 0.1 if self.predicts > 0 else 0)

        #print("err is ", err)

        #print("allstates is ", allstates)

        return err.mean(), v_err.detach().mean(), pi_err.detach().mean(), (predict_pi_err.detach().mean() if self.predicts > 0 else None), allstates
