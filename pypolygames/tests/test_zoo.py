# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import SkipTest
import pytest
import torch
from .. import model_zoo
from ..model_zoo.utils import get_game_info
from .. import params
from .. import utils


@pytest.mark.parametrize("model_name", [n for n in model_zoo.MODELS])
def test_models(model_name) -> None:
    if model_name in ["Connect4BenchModel", "ResConvConvLogitPoolModelV2"]:
        raise SkipTest(f"Skipping {model_name}")
    game_params = params.GameParams(
        game_name="Tristannogo"
        if "GameOfTheAmazons" not in model_name
        else "GameOfTheAmazons"
    )
    model_params = params.ModelParams(model_name=model_name)
    info = get_game_info(game_params)
    model = model_zoo.MODELS[model_name](game_params, model_params)
    model.eval()  # necessary for batch norm as it expects more than 1 ex in training
    feature_size = info["feature_size"][:3]
    action_size = info["action_size"][:3]
    input_data = torch.zeros([1] + feature_size, device=torch.device("cpu"))
    outputs = model.forward(input_data)
    assert list(outputs["v"].shape) == [1, 1]
    assert list(outputs["pi"].shape) == [1] + action_size
    # loss
    multi_counter = utils.MultiCounter(root=None)
    pi_mask = torch.ones(outputs["pi"].shape)
    model.loss(
        model, input_data, outputs["v"], outputs["pi"], pi_mask, multi_counter
    )  # make sure it computes something
