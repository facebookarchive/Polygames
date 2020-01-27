# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator

import torch
import random
import numpy as np


def generate_random_seeds(seed: int) -> Iterator[int]:
    # set-up all seeds
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    # generate random seeds
    generator = random.Random(seed)
    while True:
        yield generator.randint(0, 2 ** 31 - 1)


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device).detach()
    elif isinstance(batch, dict):
        return {key: to_device(batch[key], device) for key in batch}
    else:
        assert False, "unsupported type: %s" % type(batch)
