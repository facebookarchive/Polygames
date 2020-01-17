# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from . import utils as zutils
from ..params import GameParams, ModelParams
from .. import utils


@zutils.register_model
class AmazonsModel(torch.jit.ScriptModule):
    __constants__ = ["c_prime", "h_prime", "w_prime"]

    DEFAULT_FCSIZE = 1024
    DEFAULT_NNSIZE = 4
    DEFAULT_NNKS = 3
    DEFAULT_STRIDE = 1
    DEFAULT_DILATION = 1

    default_game_name = "GameOfTheAmazons"

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
        self.model_params = model_params

        self.net1 = nn.Conv2d(
            c, int(nnsize * c), nnks, stride=stride, padding=padding, dilation=dilation
        )
        self.net2 = nn.Conv2d(
            int(nnsize * c),
            int(nnsize * c),
            nnks,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.net3 = nn.Conv2d(
            int(nnsize * c),
            int(nnsize * c),
            nnks,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.net4 = nn.Conv2d(
            int(nnsize * c),
            int(nnsize * c),
            nnks,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.v1 = nn.Linear(int(nnsize * c) * h * w, fcsize)
        self.v2 = nn.Linear(fcsize, fcsize)
        self.v3 = nn.Linear(fcsize, 1)
        self.pi1 = nn.Linear(int(nnsize * c) * h * w, fcsize)
        self.pi2 = nn.Linear(fcsize, fcsize)
        self.pi3 = nn.Linear(fcsize, c_prime + h_prime + w_prime)

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):
        c_prime, h_prime, w_prime = self.c_prime, self.h_prime, self.w_prime
        bs = x.shape[0]
        h1 = nn.functional.relu(self.net1(x))
        h2 = nn.functional.relu(self.net2(h1)) + h1
        h3 = nn.functional.relu(self.net3(h2)) + h2
        h4 = nn.functional.relu(self.net4(h3)) + h3
        v = nn.functional.relu(self.v1(h4.flatten(1)))
        v = nn.functional.relu(self.v2(v))
        v = torch.tanh(self.v3(v))
        pi_logit = nn.functional.relu(self.pi1(h4.flatten(1)))
        pi_logit = nn.functional.relu(self.pi2(pi_logit))
        pi_logit = nn.functional.relu(self.pi3(pi_logit))
        if return_logit:
            v1 = pi_logit[:, :c_prime].reshape(-1, c_prime, 1, 1)
            v2 = pi_logit[:, c_prime : c_prime + h_prime].reshape(-1, 1, h_prime, 1)
            v3 = pi_logit[:, c_prime + h_prime :].reshape(-1, 1, 1, w_prime)
            # This representation is not sparse, that's a temporary hack
            # for testing the idea of a cartesian product.
            pi_logit = v1 + v2 + v3
            return v, pi_logit
        # TODO(oteytaud): remove duplicate reshaping.
        v1 = nn.functional.softmax(pi_logit[:, :c_prime].reshape(-1, c_prime, 1, 1), 1)
        v2 = nn.functional.softmax(
            pi_logit[:, c_prime : c_prime + h_prime].reshape(-1, 1, h_prime, 1), 2
        )
        v3 = nn.functional.softmax(
            pi_logit[:, c_prime + h_prime :].reshape(-1, 1, 1, w_prime), 3
        )
        pi = v1 * v2 * v3
        # pi = nn.functional.softmax(pi.view(pi.shape[0], -1), 1).reshape(pi.shape)
        # This representation is not sparse, that's a temporary hack
        # for testing the idea of a cartesian product.
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
        stat: utils.MultiCounter
    ) -> float:
        # print(x.size())
        # print(x[0])
        # print(v)
        # print(pi.size())
        batchsize = pi.shape[0]
        # pi = pi.view(batchsize, -1)
        pred_v, pred_logit = self._forward(x, True)
        utils.assert_eq(v.size(), pred_v.size())
        utils.assert_eq(pred_logit.size(), pi.size())
        utils.assert_eq(pred_logit.dim(), 4)

        pred_logit = pred_logit * pi_mask.view(pred_logit.shape)

        # pred_logit = pred_logit.view(batchsize, -1)
        v_err = 0.5 * (v - pred_v).pow(2).squeeze(1)
        s = pred_logit.shape
        bs = x.shape[0]
        pred_log_pi = nn.functional.log_softmax(pred_logit.flatten(1), 1).reshape(s)
        # pred_log_pi = nn.functional.log_softmax(pred_logit, 1)
        pi_err = -(pred_log_pi * pi).reshape(bs, -1).sum(1)

        # why would these quantities be equal ?
        utils.assert_eq(v_err.size(), pi_err.size())
        err = v_err + pi_err

        stat["v_err"].feed(v_err.detach().mean().item())
        stat["pi_err"].feed(pi_err.detach().mean().item())
        return err.mean()
