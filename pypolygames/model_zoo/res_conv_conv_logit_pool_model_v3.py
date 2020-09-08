# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple, List

from . import utils as zutils
from ..params import GameParams, ModelParams
from .. import utils


@zutils.register_model
class ResConvConvLogitPoolModelV3(torch.jit.ScriptModule):
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
        self.nb_nets = nb_nets
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

        groups = self.groups = 1
        sereduction = 8

        hc = int((nnsize * c + groups - 1) // groups * groups)
        #poolhc = int(int(hc * self.global_pooling + groups / 2 + 1) // (groups / 2) * (groups / 2)) * 2
        poolhc = int(hc * self.global_pooling)

        print("global pooling ", self.global_pooling)
        print("af ", model_params.activation_function)
        print("groups ", groups)
        print("hc ", hc)
        print("poolhc ", poolhc)

        self.downsample: Dict[int, float] = {}
        self.upsample: Dict[int, bool] = {}

        self.downsample[-1] = 1.0
        self.upsample[-1] = False

        #self.downsample[int(nb_nets * 0.125)] = 0.5
        #self.downsample[int(nb_nets * 0.33)] = 0.5
        #self.upsample[int(nb_nets * 0.66)] = True
        #self.upsample[int(nb_nets * 0.875)] = True

        #self.downsample[int(nb_nets * 0.25)] = 0.5
        #self.upsample[int(nb_nets * 0.75)] = True

        mono = [
            nn.Conv2d(
                c,
                hc,
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
                    hc + (poolhc if _ == 0 else 0),
                    hc,
                    nnks,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=not bn_affine,
                    groups=groups,
                )
                for _ in range(nb_layers_per_net)
            ]
            if bn or bn_affine:
                for j in range(nb_layers_per_net):
                    nets[j] = nn.Sequential(
                        nets[j],
                        nn.BatchNorm2d(
                            hc, track_running_stats=True, affine=bn_affine, momentum=batchnorm_momentum
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
            resnet_list.append(nn.ModuleList(nets))
        if bn or bn_affine:
            mono.append(
                nn.BatchNorm2d(hc, track_running_stats=True, affine=bn_affine, momentum=batchnorm_momentum),
            )
        self.mono = nn.Sequential(*mono)
        self.resnets = nn.ModuleList(resnet_list)
        self.seblocks = nn.ModuleList([nn.ModuleList([nn.Linear(2 * hc, int(hc / sereduction)), nn.Linear(int(hc / sereduction), hc)]) for i in range(nb_nets)])
        if self.global_pooling > 0 or True:
            self.poolblocks = nn.ModuleList([nn.Linear(2 * hc, poolhc) for i in range(nb_nets)])
        else:
            self.poolblocks = nn.ModuleList([])
        self.v = nn.Linear(2 * hc, 2 * hc)
        self.v_bn = nn.BatchNorm1d(2 * hc, momentum=batchnorm_momentum)
        self.v2 = nn.Linear(2 * hc, 3)
        self.logit_value = True
        self.pi_logit = nn.Conv2d(
            hc, c_prime, nnks, stride=stride, padding=padding, dilation=dilation
        )
        if self.predicts > 0:
          self.predict_pi_logit = nn.Conv2d(
              hc, r_c * self.predicts, nnks, stride=stride, padding=padding, dilation=dilation
          )
        else:
          self.predict_pi_logit = None

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):
        af = self.af
        global_pooling = self.global_pooling
        groups = self.groups
        h = af(self.mono(x))
        i = 0
        sizestack: List[Tuple[int, int]] = []
        for resnet, seblock, poolblock in zip(self.resnets, self.seblocks, self.poolblocks):
            if i in self.downsample:
                sizestack.append((h.size(-2), h.size(-1)))
                h = F.interpolate(h, scale_factor=self.downsample[i], mode="bilinear", recompute_scale_factor=True, align_corners=False)
            elif i in self.upsample:
                h = F.interpolate(h, size=sizestack.pop(), mode="bilinear", align_corners=False)
            previous_block = h
            if global_pooling > 0:
                maxpool = torch.cat((F.adaptive_max_pool2d(h, 1).flatten(-2), F.adaptive_avg_pool2d(h, 1).flatten(-2)), -1).flatten(-2)
                x = af(seblock[0](maxpool))
                x = seblock[1](x).sigmoid()
                x = x.unsqueeze(-1).unsqueeze(-1)
                h = h * x

                x = af(poolblock(maxpool))
                h = torch.cat((h, x.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), x.size(1), h.size(2), h.size(3))), 1)
            sublayer_no = 0
            for net in resnet:
                if sublayer_no < self.nb_layers_per_net - 1:
                    h = af(net(h))
                else:
                    h = af(net(h) + previous_block)
                sublayer_no = sublayer_no + 1

            i += 1
        pool = torch.cat((F.adaptive_max_pool2d(h, 1).flatten(-2), F.adaptive_avg_pool2d(h, 1).flatten(-2)), -1).flatten(-2)
        pi_logit = self.pi_logit(h).flatten(1)
        v = af(self.v_bn(self.v(pool)))
        v = self.v2(v)
        if return_logit:
            if self.predict_pi_logit is not None:
                predict_pi_logit = self.predict_pi_logit(h)
                return v, pi_logit, predict_pi_logit
            else:
                return v, pi_logit, torch.empty(0)
        s = pi_logit.shape
        pi = F.softmax(pi_logit.float(), 1).reshape(s)
        v = F.softmax(v.float(), 1)
        return v, pi, torch.empty(0)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        v, pi, _ = self._forward(x, False)
        pi = pi.view(-1, self.c_prime, x.size(2), x.size(3))
        reply = {"v": v, "pi": pi}
        return reply

    def loss(
        self,
        model,
        x: torch.Tensor,
        v: torch.Tensor,
        pi: torch.Tensor,
        pi_mask: torch.Tensor,
        predict_pi: torch.Tensor = None,
        predict_pi_mask: torch.Tensor = None,
        #stat: utils.MultiCounter,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi = pi.flatten(1)
        pred_v, pred_logit, pred_predict_logit = model._forward(x, return_logit=True)
        #utils.assert_eq(v.size(), pred_v.size())
        #utils.assert_eq(pred_logit.size(), pi.size())
        #utils.assert_eq(pred_logit.dim(), 2)

        pi_mask = pi_mask.view(pred_logit.shape);
        pred_logit = pred_logit * pi_mask - 400 * (1 - pi_mask)
        if self.predicts > 0:
            predict_pi_err = (F.mse_loss(pred_predict_logit, predict_pi, reduction="none") * predict_pi_mask).flatten(2).sum(2).flatten(1).mean(1)

        v_err = -(nn.functional.log_softmax(pred_v, dim=1) * v).sum(1)
        pred_log_pi = nn.functional.log_softmax(pred_logit.flatten(1), dim=1).view_as(pred_logit) * pi_mask
        pi_err = -(pred_log_pi * pi).sum(1)

        #utils.assert_eq(v_err.size(), pi_err.size())
        err = v_err * 1.5 + pi_err + (predict_pi_err * 0.1 if self.predicts > 0 else 0)

        #stat["v_err"].feed(v_err.detach().mean().item())
        #stat["pi_err"].feed(pi_err.detach().mean().item())
        return err.mean(), v_err.detach().mean(), pi_err.detach().mean(), (predict_pi_err.detach().mean() if self.predicts > 0 else None)
