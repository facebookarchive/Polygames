# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F

from . import utils as zutils
from ..params import GameParams, ModelParams
from .. import utils


@zutils.register_model
class DeepConvFCLogitModel(torch.jit.ScriptModule):
    __constants__ = ["c_prime", "h_prime", "w_prime", "mono", "conv_nets"]

    DEFAULT_NB_NETS = 13
    DEFAULT_NNSIZE = 2
    DEFAULT_NNKS = 3
    DEFAULT_STRIDE = 1
    DEFAULT_DILATION = 1
    DEFAULT_POOLING = False
    DEFAULT_BN = False
    # DEFAULT_BN_AFFINE = False

    default_game_name = "Connect4"

    def __init__(self, game_params: GameParams, model_params: ModelParams):
        torch.jit.ScriptModule.__init__(self)
        if game_params.game_name is None:
            game_params.game_name = self.__class__.default_game_name
        self.game_name = game_params.game_name
        self.game_params = game_params
        info = zutils.get_game_info(game_params)
        c, h, w = self.c, self.h, self.w = info["feature_size"][:3]
        c_prime, h_prime, w_prime = self.c_prime, self.h_prime, self.w_prime = info[
            "action_size"
        ][:3]

        # nb identical hidden layers (first layer excepted)
        if model_params.nb_nets is None:
            model_params.nb_nets = self.DEFAULT_NB_NETS
        nb_nets = model_params.nb_nets
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
        conv_nets = [
            nn.Conv2d(
                int(nnsize * c),
                int(nnsize * c),
                nnks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not bn_affine,
            )
            for _ in range(nb_nets)
        ]
        if pooling:
            for i in range(nb_nets):
                conv_nets[i] = nn.Sequential(
                    conv_nets[i],
                    nn.MaxPool2d(
                        kernel_size=nnks,
                        padding=padding,
                        stride=stride,
                        dilation=dilation,
                    ),
                )
        if bn or bn_affine:
            mono.append(
                nn.BatchNorm2d(int(nnsize * c), track_running_stats=True, affine=bn_affine)
            )
            for i in range(nb_nets):
                conv_nets[i] = nn.Sequential(
                    conv_nets[i],
                    nn.BatchNorm2d(
                        int(nnsize * c), track_running_stats=True, affine=bn_affine
                    ),
                )
        self.mono = nn.Sequential(*mono)
        self.conv_nets = nn.ModuleList(conv_nets)
        self.v = nn.Linear(int(nnsize * c) * h * w, 1)
        self.pi_logit = nn.Linear(int(nnsize * c) * h * w, c_prime * h_prime * w_prime)

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):
        bs = x.shape[0]
        h = F.relu(self.mono(x))
        for conv_net in self.conv_nets:
            h = F.relu(conv_net(h))
        v = torch.tanh(self.v(h.flatten(1)))
        pi_logit = self.pi_logit(h.flatten(1))
        if return_logit:
            return v, pi_logit
        s = pi_logit.shape
        pi = F.softmax(pi_logit.flatten(1), 1).reshape(s)

        return v, pi

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        v, pi_logit = self._forward(x, True)
        pi_logit = pi_logit.view(-1, self.c_prime, self.h_prime, self.w_prime)
        reply = {"v": v, "pi_logit": pi_logit}
        return reply
