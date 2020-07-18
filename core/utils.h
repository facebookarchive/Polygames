/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/torch.h>

#include "state.h"

inline void getFeatureInTensor(const State& state, float* dest) {
  auto& feat = state.GetFeatures();
  memcpy(dest, feat.data(), sizeof(float) * feat.size());
}

inline void getFeatureInTensor(const State& state, torch::Tensor dest) {
  assert(dest.dtype() == torch::kFloat32);
  auto& feat = state.GetFeatures();
  torch::Tensor temp = torch::from_blob(
      (void*)feat.data(), state.GetFeatureSize(), dest.dtype());
  dest.copy_(temp);
}

inline torch::Tensor getFeatureInTensor(const State& state) {
  torch::Tensor t = torch::zeros(state.GetFeatureSize(), torch::kFloat32);
  getFeatureInTensor(state, t);
  return t;
}

inline void getPolicyInTensor(const State& state,
                              const std::unordered_map<mcts::Action, float>& pi,
                              torch::Tensor& dest,
                              torch::Tensor& mask) {
  assert(dest.dtype() == torch::kFloat32);
  assert(state.GetActionSize().size() == 3);
  auto accessor = dest.accessor<float, 3>();
  auto maskaccessor = mask.accessor<float, 3>();

  const auto& legalAction = state.GetLegalActions();
  for (const auto& a2pi : pi) {
    mcts::Action actionIdx = a2pi.first;
    if (actionIdx >= (int)legalAction.size() || actionIdx < 0) {
      std::terminate();
      std::cout << "Wrong action in getPolicyTargetInTensor, "
                << "action idx: " << actionIdx
                << ", num legal: " << legalAction.size() << std::endl;
      assert(false);
    }
    const auto& action = legalAction[actionIdx];
    float pi = a2pi.second;
    // std::cout << "action: " << action << ", pi: " << pi << std::endl;
    accessor[action->GetX()][action->GetY()][action->GetZ()] = pi;
    maskaccessor[action->GetX()][action->GetY()][action->GetZ()] = 1;
  }
}

inline std::pair<torch::Tensor, torch::Tensor> getPolicyInTensor(
    const State& state, const std::unordered_map<mcts::Action, float>& pi) {
  torch::Tensor t = torch::zeros(state.GetActionSize(), torch::kFloat32);
  torch::Tensor mask = torch::zeros(state.GetActionSize(), torch::kFloat32);
  getPolicyInTensor(state, pi, t, mask);
  return std::make_pair(t, mask);
}

inline void normalize(std::unordered_map<mcts::Action, float>& a2pi) {
  float sumProb = 0.0f;
  for (const auto& p : a2pi) {
    sumProb += p.second;
  }

  if (sumProb != 0.0f) {
    for (auto& p : a2pi) {
      p.second /= sumProb;
    }
  }
}

inline std::unordered_map<mcts::Action, float> getLegalPi(
    const State& state, torch::TensorAccessor<float, 3> accessor) {
  std::unordered_map<mcts::Action, float> a2pi;
  const auto& legalActions = state.GetLegalActions();
  auto as = state.GetActionSize();
  for (const auto& action : legalActions) {
    if ((action->GetX() >= as[0]) || (action->GetY() >= as[1]) ||
        (action->GetZ() >= as[2])) {
      std::cout << action->GetX() << "," << action->GetY() << ","
                << action->GetZ() << " / " << as[0] << "," << as[1] << ","
                << as[2] << std::endl;
    }
    assert(action->GetX() < as[0]);
    assert(action->GetY() < as[1]);
    assert(action->GetZ() < as[2]);
    float pi = accessor[action->GetX()][action->GetY()][action->GetZ()];
    auto it = a2pi.insert({action->GetIndex(), pi});
    assert(it.second == true);
  }
  normalize(a2pi);
  return a2pi;
}

inline std::unordered_map<mcts::Action, float> getLegalPi(
    const State& state, const torch::Tensor& pi) {
  // assert pi is prob not logit
  // std::cout << pi.sum().item<float>() << std::endl;
  float pi_sum = pi.sum().item<float>();
  assert(std::abs(pi_sum - 1) < 1e-2);
  assert(pi.min().item<float>() >= (float)0);

  auto accessor = pi.accessor<float, 3>();
  return getLegalPi(state, accessor);
}

inline int64_t product(const std::vector<int64_t>& nums) {
  int64_t p = 1;
  for (auto n : nums) {
    p *= n;
  }
  return p;
}
