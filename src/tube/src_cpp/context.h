/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "env_thread.h"

namespace tube {

class Context {
 public:
  Context()
      : started_(false)
      , numTerminatedThread_(0) {
  }

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  ~Context() {
    for (auto& v : envs_) {
      v->terminate();
    }
    for (auto& v : threads_) {
      v.join();
    }
  }

  int pushEnvThread(std::shared_ptr<EnvThread> env) {
    assert(!started_);
    envs_.push_back(std::move(env));
    return (int)envs_.size();
  }

  void start() {
    for (int i = 0; i < (int)envs_.size(); ++i) {
      threads_.emplace_back([this, i]() {
        envs_[i]->mainLoop();
        ++numTerminatedThread_;
      });
    }
  }

  bool terminated() {
    // std::cout << ">>> " << numTerminatedThread_ << std::endl;
    return numTerminatedThread_ == (int)envs_.size();
  }

  std::string getStatsStr() const {
    EnvThread::Stats cum_stats;
    for (const auto& env : envs_) {
      const auto& stats = env->get_stats();
      for (const auto& key2stat : stats) {
        auto& cum_stats_val = cum_stats[key2stat.first];
        const auto& stat_val = key2stat.second;
        std::get<0>(cum_stats_val) += std::get<0>(stat_val);
        std::get<1>(cum_stats_val) += std::get<1>(stat_val);
        std::get<2>(cum_stats_val) += std::get<2>(stat_val);
      }
    }
    std::ostringstream oss;
    for (const auto& key2stat : cum_stats) {
      const auto f0 = std::get<0>(key2stat.second);
      const auto f1 = std::get<1>(key2stat.second);
      const auto f2 = std::get<2>(key2stat.second);
      const auto mean = (f0 > 0 ? f1 / f0 : 0);
      const auto stddev = (f0 > 0 ? std::sqrt(f2 / f0 - mean * mean) : 0);
      oss << key2stat.first << ": N=" << f0 << ", avg=" << mean
          << ", std=" << stddev << std::endl;
    }
    return oss.str();
  }

 private:
  bool started_;
  std::atomic<int> numTerminatedThread_;
  std::vector<std::shared_ptr<EnvThread>> envs_;
  std::vector<std::thread> threads_;
};
}  // namespace tube
