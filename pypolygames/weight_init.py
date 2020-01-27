# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch


def _init_weight_from_method(init_method):
    def wrapped_init_method(net):
        if getattr(net, "weight", None) is not None:
            # with batch norm affine some weight have dim == 1
            if net.weight.dim() > 1:
                return init_method(net)

    return wrapped_init_method


WEIGHT_INIT = OrderedDict(
    xavier_uniform=_init_weight_from_method(
        lambda net: torch.nn.init.xavier_uniform_(net.weight, gain=1.0)
    ),
    xavier_normal=_init_weight_from_method(
        lambda net: torch.nn.init.xavier_normal_(net.weight, gain=1.0)
    ),
    kaiming_uniform=_init_weight_from_method(
        lambda net: torch.nn.init.kaiming_uniform_(
            net.weight, a=0, mode="fan_in", nonlinearity="relu"
        )
    ),
    kaiming_normal=_init_weight_from_method(
        lambda net: torch.nn.init.kaiming_normal_(
            net.weight, a=0, mode="fan_in", nonlinearity="relu"
        )
    ),
)
