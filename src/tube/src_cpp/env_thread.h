/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <tuple>
#include <unordered_map>

namespace tube {

class EnvThread {
 public:
  using StatsValue = std::tuple<double, double, double>;
  using Stats = std::unordered_map<std::string, StatsValue>;

  EnvThread() = default;
  EnvThread(EnvThread&& n) {
    terminate_ = n.terminate_.load();
  }

  virtual ~EnvThread() {
  }

  virtual void mainLoop() = 0;

  virtual void terminate() {
    terminate_ = true;
  }

  /// Get various statistics associated with this thread
  virtual Stats get_stats() {
    return Stats();
  }

  std::atomic_bool terminate_{false};
};
}  // namespace tube
