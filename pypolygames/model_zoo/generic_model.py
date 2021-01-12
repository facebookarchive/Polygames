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
class GenericModel(torch.jit.ScriptModule):
    __constants__ = [
        "c_prime",
        "h_prime",
        "w_prime",
        "net1",
        "net2",
        "net3",
        "net4",
        "v1",
        "v2",
        "pi1",
        "pi2",
    ]

    DEFAULT_FCSIZE = 1024
    DEFAULT_NNSIZE = 2
    DEFAULT_NNKS = 3
    DEFAULT_STRIDE = 1
    DEFAULT_DILATION = 1
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

        # fc size
        if model_params.fcsize is None:
            model_params.fcsize = self.DEFAULT_FCSIZE
        fcsize = model_params.fcsize
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

        net1 = [
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
        net2 = [
            nn.Conv2d(
                int(nnsize * c),
                int(nnsize * c),
                nnks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not bn_affine,
            )
        ]
        net3 = [
            nn.Conv2d(
                int(nnsize * c),
                int(nnsize * c),
                nnks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not bn_affine,
            )
        ]
        net4 = [
            nn.Conv2d(
                int(nnsize * c),
                int(nnsize * c),
                nnks,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not bn_affine,
            )
        ]
        v1 = [nn.Linear(int(nnsize * c) * h * w, fcsize)]
        v2 = [nn.Linear(fcsize, fcsize)]
        pi1 = [nn.Linear(int(nnsize * c) * h * w, fcsize)]
        pi2 = [nn.Linear(fcsize, fcsize)]
        if bn or bn_affine:
            net1.append(
                nn.BatchNorm2d(int(nnsize * c), track_running_stats=True, affine=bn_affine)
            )
            net2.append(
                nn.BatchNorm2d(int(nnsize * c), track_running_stats=True, affine=bn_affine)
            )
            net3.append(
                nn.BatchNorm2d(int(nnsize * c), track_running_stats=True, affine=bn_affine)
            )
            net4.append(
                nn.BatchNorm2d(int(nnsize * c), track_running_stats=True, affine=bn_affine)
            )
            v1.append(
                nn.BatchNorm1d(fcsize, track_running_stats=True, affine=bn_affine)
            )
            v2.append(
                nn.BatchNorm1d(fcsize, track_running_stats=True, affine=bn_affine)
            )
            pi1.append(
                nn.BatchNorm1d(fcsize, track_running_stats=True, affine=bn_affine)
            )
            pi2.append(
                nn.BatchNorm1d(fcsize, track_running_stats=True, affine=bn_affine)
            )
        self.net1 = nn.Sequential(*net1)
        self.net2 = nn.Sequential(*net2)
        self.net3 = nn.Sequential(*net3)
        self.net4 = nn.Sequential(*net4)
        self.v1 = nn.Sequential(*v1)
        self.v2 = nn.Sequential(*v2)
        self.pi1 = nn.Sequential(*pi1)
        self.pi2 = nn.Sequential(*pi2)
        self.v3 = nn.Linear(fcsize, 1)
        self.pi3 = nn.Linear(fcsize, c_prime * h_prime * w_prime)

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):
        h1 = F.relu(self.net1(x))
        h2 = F.relu(self.net2(h1)) + h1
        h3 = F.relu(self.net3(h2)) + h2
        h4 = F.relu(self.net4(h3)) + h3
        v1 = F.relu(self.v1(h4.flatten(1)))
        v2 = F.relu(self.v2(v1))
        v = torch.tanh(self.v3(v2))
        pi_logit1 = F.relu(self.pi1(h4.flatten(1)))
        pi_logit2 = F.relu(self.pi2(pi_logit1))
        pi_logit = self.pi3(pi_logit2)
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
