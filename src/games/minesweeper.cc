/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <iostream>
#include <ctime>

#include "minesweeper_common.h"

namespace Minesweeper {

static std::ostream& timestamp(std::ostream& os) {
  std::time_t result = std::time(nullptr);
  char buf[100];
  if (std::strftime(buf, sizeof(buf), "%c", std::localtime(&result))) {
    os << buf;
  }
  return os;
} // timestamp

std::ostream& debug(std::ostream& os) {
  timestamp(os) << " [DEBUG] [Minesweeper] ";
  return os;
} // debug

std::string sparseMaskToString(const SparseMask& mask) {
  std::ostringstream oss;
  oss << '[';
  for (const auto& v: mask) {
    oss << '(' << v.row() << ',' << v.col() << "), ";
  }
  oss << ']';
  return oss.str();
} // sparseMaskToString

} // namespace Minesweeper
