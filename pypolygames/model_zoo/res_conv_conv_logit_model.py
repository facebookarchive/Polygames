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
class ResConvConvLogitModel(torch.jit.ScriptModule):
    __constants__ = [
        "c_prime",
        "h_prime",
        "w_prime",
        "nb_layers_per_net",
        "mono",
        "resnets",
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
                    nets[j] = nn.Sequential(
                        nets[j],
                        nn.BatchNorm2d(
                            int(nnsize * c), track_running_stats=True, affine=bn_affine
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
                nn.BatchNorm2d(int(nnsize * c), track_running_stats=True, affine=bn_affine),
            )
            for i in range(nb_nets):
                for j in range(nb_layers_per_net):
                    resnet_list[i][j] = nn.Sequential(
                        resnet_list[i][j],
                        nn.BatchNorm2d(
                            int(nnsize * c), track_running_stats=True, affine=bn_affine
                        ),
                    )
        for i in range(nb_nets):
            resnet_list[i] = nn.ModuleList(resnet_list[i])
        self.mono = nn.Sequential(*mono)
        self.resnets = nn.ModuleList(resnet_list)
        self.v = nn.Linear(int(nnsize * c) * h * w, 1)
        self.pi_logit = nn.Conv2d(
            int(nnsize * c), c_prime, nnks, stride=stride, padding=padding, dilation=dilation
        )

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):
        previous_block = self.mono(x)  # linear transformation only
        for resnet in self.resnets:
            sublayer_no = 0
            h = F.relu(previous_block)  # initial activation
            for net in resnet:
                if sublayer_no < self.nb_layers_per_net - 1:
                    h = F.relu(net(h))
                else:
                    h = net(h) + previous_block  #  linear transformation only
                    previous_block = h
                sublayer_no = sublayer_no + 1
        h = F.relu(previous_block)  # final activation
        v = torch.tanh(self.v(h.flatten(1)))
        pi_logit = self.pi_logit(h).flatten(1)
        if return_logit:
            return v, pi_logit
        s = pi_logit.shape
        pi = F.softmax(pi_logit, 1).reshape(s)
        return v, pi

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        v, pi = self._forward(x, False)
        pi = pi.view(-1, self.c_prime, self.h_prime, self.w_prime)
        reply = {"v": v, "pi": pi}
        return reply

    def loss(
        self,
        model,
        x: torch.Tensor,
        v: torch.Tensor,
        pi: torch.Tensor,
        pi_mask: torch.Tensor,
        stat: utils.MultiCounter,
    ) -> float:
        pi = pi.flatten(1)
        pred_v, pred_logit = model._forward(x, return_logit=True)
        utils.assert_eq(v.size(), pred_v.size())
        utils.assert_eq(pred_logit.size(), pi.size())
        utils.assert_eq(pred_logit.dim(), 2)

        pi_mask = pi_mask.view(pred_logit.shape);
        pred_logit = pred_logit * pi_mask - 40 * (1 - pi_mask)
        #pi = pi * pi_mask

        v_err = F.mse_loss(pred_v, v, reduction="none").squeeze(1)
        pred_log_pi = nn.functional.log_softmax(pred_logit.flatten(1), dim=1).view_as(
            pred_logit
        ) * pi_mask
        pi_err = -(pred_log_pi * pi).sum(1)

        utils.assert_eq(v_err.size(), pi_err.size())
        err = v_err + pi_err

        stat["v_err"].feed(v_err.detach().mean().item())
        stat["pi_err"].feed(pi_err.detach().mean().item())
        return err.mean()
