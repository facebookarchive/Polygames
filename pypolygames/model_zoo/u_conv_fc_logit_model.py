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
class UConvFCLogitModel(torch.jit.ScriptModule):
    __constants__ = [
        "c_prime",
        "h_prime",
        "w_prime",
        "nb_layers_per_net",
        "mono",
        "nb_unets_div_by_2",
        "unets",
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

        # nb unets
        if model_params.nb_nets is None:
            model_params.nb_nets = self.DEFAULT_NB_NETS
        nb_nets = model_params.nb_nets
        if nb_nets % 2 == 0:
            raise RuntimeError(
                f'The model "{self.__class__.__name__}" accepts only odd numbers '
                f'for "nb_nets" while it was set to {nb_nets}'
            )
        self.nb_unets_div_by_2 = nb_unets_div_by_2 = nb_nets // 2
        # nb layers per unet
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
        if bn or bn_affine:
            mono.append(
                nn.BatchNorm2d(int(nnsize * c), track_running_stats=True, affine=bn_affine)
            )
        self.mono = nn.Sequential(*mono)

        unet_list = [None] * (2 * nb_unets_div_by_2 + 1)
        for i in range(nb_unets_div_by_2):
            nets1 = [
                nn.Conv2d(
                    int(nnsize * c),
                    int(nnsize * c),
                    nnks,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=not bn_affine,
                )
                for _ in range(nb_layers_per_net)
            ]
            nets2 = [
                nn.Conv2d(
                    int(nnsize * c),
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
                    nets1[j] = nn.Sequential(
                        nets1[j],
                        nn.BatchNorm2d(
                            int(nnsize * c), track_running_stats=True, affine=bn_affine
                        ),
                    )
                    nets2[j] = nn.Sequential(
                        nets2[j],
                        nn.BatchNorm2d(
                            int(nnsize * c), track_running_stats=True, affine=bn_affine
                        ),
                    )
            if pooling:
                for j in range(nb_layers_per_net):
                    nets1[j] = nn.Sequential(
                        nets1[j],
                        nn.MaxPool2d(
                            kernel_size=nnks,
                            padding=padding,
                            stride=stride,
                            dilation=dilation,
                        ),
                    )
                    nets2[j] = nn.Sequential(
                        nets2[j],
                        nn.MaxPool2d(
                            kernel_size=nnks,
                            padding=padding,
                            stride=stride,
                            dilation=dilation,
                        ),
                    )
            unet_list[i] = nn.ModuleList(nets1)
            unet_list[-i - 1] = nn.ModuleList(nets2)
        middle_nets = [
            nn.Conv2d(
                int(nnsize * c),
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
                middle_nets[j] = nn.Sequential(
                    middle_nets[j],
                    nn.BatchNorm2d(
                        int(nnsize * c), track_running_stats=True, affine=bn_affine
                    ),
                )
        if pooling:
            for j in range(nb_layers_per_net):
                middle_nets[j] = nn.Sequential(
                    middle_nets[j],
                    nn.MaxPool2d(
                        kernel_size=nnks,
                        padding=padding,
                        stride=stride,
                        dilation=dilation,
                    ),
                )
        unet_list[nb_unets_div_by_2] = nn.ModuleList(middle_nets)
        self.unets = nn.ModuleList(unet_list)

        self.v = nn.Linear(int(nnsize * c) * h * w, 1)
        self.pi_logit = nn.Linear(int(nnsize * c) * h * w, c_prime * h_prime * w_prime)

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):
        h = self.mono(x)  # linear transformation only
        saved_h = [
            h
        ] * self.nb_unets_div_by_2  # saves output of last linear transformation
        layer_no = 0
        for unet in self.unets:
            sublayer_no = 0
            for net in unet:
                h = F.relu(h)  # activation on previous block
                h = net(h)
                if sublayer_no == self.nb_layers_per_net - 1:
                    if layer_no < self.nb_unets_div_by_2:
                        saved_h[layer_no] = h
                    elif layer_no > self.nb_unets_div_by_2:
                        h = h + saved_h[layer_no - self.nb_unets_div_by_2 - 1]
                sublayer_no = sublayer_no + 1
            layer_no = layer_no + 1
        h = F.relu(h)  # final activation
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
