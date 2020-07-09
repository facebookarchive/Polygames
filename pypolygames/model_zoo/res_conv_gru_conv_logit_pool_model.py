# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple

from . import utils as zutils
from ..params import GameParams, ModelParams
from .. import utils


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
        rnnlist = []
        for i in range(nb_nets):
            resnet_list[i] = nn.ModuleList(resnet_list[i])
            list = []
            if rnn_interval > 0:
              rnni += 1
              while rnni >= rnn_interval:
                  rnni -= rnn_interval
                  list.append(nn.GRUCell(int(nnsize * c), int(nnsize * c)))
                  self.rnn_cells += 1
            rnnlist.append(nn.ModuleList(list))
        self.rnnlist = nn.ModuleList(rnnlist)
        self.rnn_channels = int(nnsize * c)
        print("model has %d GRU layers of size %d" % (self.rnn_cells, self.rnn_channels))
        self.mono = nn.Sequential(*mono)
        self.resnets = nn.ModuleList(resnet_list)
        self.v = nn.Linear(2 * int(nnsize * c), 2 * int(nnsize * c))
        self.v2 = nn.Linear(2 * int(nnsize * c), 1)
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
    def _forward(self, x: torch.Tensor, rnn_state: torch.Tensor, rnn_state_mask: torch.Tensor, return_logit: bool):
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
                C = h.size(1)
                H = h.size(2)
                W = h.size(3)

                # B * S, C, H, W
                h = h.view(B, S, C, H, W)
                # B, S, C, H, W
                h = h.permute(1, 0, 3, 4, 2)
                # S, B, H, W,  C
                h = h.flatten(1, 3)
                # S, B * H * W, C

                sx = rnn_state.select(1, gruindex)
                # B, C, H, W
                sx = sx.permute(0, 2, 3, 1)
                # B, H, W, C
                sx = sx.flatten(0, 2)
                # B * H * W, C

                hout = []
                if rnn_state_mask.dim() != 2:
                  for s in range(S):
                    sx = net(h[s], sx)
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
                h = h.permute(0, 3, 1, 2)
                # B, C, H, W

                # B * H * W, C
                sx = sx.view(B, H, W, C)
                # B, H, W, C
                sx = sx.permute(0, 3, 1, 2)
                # B, C, H, W

                gruindex += 1
                states.append(sx.view(B, C, H, W))
        if gruindex != len(states) or gruindex != self.rnn_cells:
            raise RuntimeError("GRU count mismatch")
        h = af(previous_block)  # final activation
        pool = torch.cat((F.adaptive_max_pool2d(h, 1), F.adaptive_avg_pool2d(h, 1)), 1)
        pi_logit = self.pi_logit(h).flatten(1)
        v = af(self.v(pool.flatten(1)))
        v = torch.tanh(self.v2(v))
        if return_logit:
            if self.predict_pi_logit is not None:
                predict_pi_logit = self.predict_pi_logit(h)
                return v, pi_logit, predict_pi_logit, torch.stack(states, 1)
            else:
                return v, pi_logit, torch.empty(0), torch.stack(states, 1)
        s = pi_logit.shape
        pi = F.softmax(pi_logit, 1).reshape(s)
        return v, pi, torch.empty(0), torch.stack(states, 1)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, rnnState: torch.Tensor):
        v, pi, _, state = self._forward(x, rnnState, torch.empty(0), False)
        pi = pi.view(-1, self.c_prime, x.size(2), x.size(3))
        reply = {"v": v, "pi": pi, "rnn_state": state}
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
        state = rnn_initial_state.view(bs, self.rnn_cells, self.rnn_channels, x.size(3), x.size(4))

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

        pred_v, pred_logit, pred_predict_logit, state = model._forward(x, state, rnn_state_mask, return_logit=True)

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

        v_err = F.mse_loss(pred_v, v, reduction="none").squeeze(1)
        pred_log_pi = nn.functional.log_softmax(pred_logit.flatten(1), dim=1).view_as(pred_logit) * pi_mask
        pi_err = -(pred_log_pi * pi).sum(1)

        #pi_err = (-((1 - pred_log_pi.exp().clamp_max(0.9999)).log() * -pi.clamp_max(0)) + -(pred_log_pi * pi.clamp_min(0))).sum(1)

        #err = v_err + pi_err + (predict_pi_err * 0.15 if self.predicts > 0 else 0)
        err = state.flatten(1).sum(1).unsqueeze(1).expand(bs, seqlen).flatten(0, 1) * 0.0 + v_err * 1.65 + pi_err * 0.9 + (predict_pi_err * 0.15 if self.predicts > 0 else 0)

        return err.mean(), v_err.detach().mean(), pi_err.detach().mean(), (predict_pi_err.detach().mean() if self.predicts > 0 else None)
