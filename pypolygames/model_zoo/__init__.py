# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .generic_model import GenericModel
from .amazons_model import AmazonsModel
from .nano_fc_logit_model import NanoFCLogitModel
from .nano_conv_logit_model import NanoConvLogitModel
from .deep_conv_fc_logit_model import DeepConvFCLogitModel
from .deep_conv_conv_logit_model import DeepConvConvLogitModel
from .res_conv_fc_logit_model import ResConvFCLogitModel
from .res_conv_conv_logit_model import ResConvConvLogitModel
from .res_conv_conv_logit_pool_model import ResConvConvLogitPoolModel
from .res_conv_conv_logit_pool_model_v2 import ResConvConvLogitPoolModelV2
from .u_conv_fc_logit_model import UConvFCLogitModel
from .u_conv_conv_logit_model import UConvConvLogitModel
from .connect4_benchmark_model import Connect4BenchModel

from .utils import MODELS  # directory where models are registered
