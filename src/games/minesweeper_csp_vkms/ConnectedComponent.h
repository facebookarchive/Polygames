/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../minesweeper_common.h"

namespace csp {
namespace vkms {

struct ConnectedComponent {
  Minesweeper::SparseMask _constraints;
  Minesweeper::SparseMask _variables;
}; // struct ConnectedComponent

} // namespace vkms
} // namespace csp
