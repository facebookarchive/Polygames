# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from . import utils as zutils
from ..params import GameParams, ModelParams

# import utils

@zutils.register_model
class Connect4BenchModel(torch.jit.ScriptModule):
    def __init__(self, game_params: GameParams, model_params: ModelParams):
    # def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6 * 7 * 2, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc_pi = nn.Linear(200, 7)
        self.fc_val = nn.Linear(200, 1)

    @torch.jit.script_method
    def _forward(self, x: torch.Tensor, return_logit: bool):
        x = x[:, :2, :, :]
        # print(x.size())
        x = x.view(-1, 84)
        h = nn.functional.relu(self.fc1(x))
        h = nn.functional.relu(self.fc2(h))
        h = nn.functional.relu(self.fc3(h))
        v = self.fc_val(h)
        pi_logit = self.fc_pi(h)
        if return_logit:
            return v, pi_logit
        pi = nn.functional.softmax(pi_logit, 1)
        return v, pi

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        v, pi_logit = self._forward(x, True)
        pi_logit = pi_logit.view(-1, 7, 1, 1)
        reply = {"v": v, "pi_logit": pi_logit}
        return reply

