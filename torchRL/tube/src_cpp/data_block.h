/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/extension.h>
#include <vector>

#include "utils.h"

namespace tube {

class DataBlock {
 public:
  DataBlock(const std::string& name,
            const std::vector<int64_t>& sizes,
            torch::ScalarType dtype)
      : name(std::move(name))
      , data(torch::zeros(sizes, dtype)) {
  }

  torch::Tensor& getBuffer() {
    return data;
  }

  std::vector<int64_t> sizes() {
    return data.sizes().vec();
  }

  torch::ScalarType dtype() {
    return data.scalar_type();
  }

  const std::string name;
  torch::Tensor data;
};
}

#include "episodic_trajectory.h"
#include "fixed_len_trajectory.h"
#include "indefinite_trajectory.h"
