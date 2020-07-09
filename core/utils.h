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
  if (feat.size() != temp.numel()) {
    throw std::runtime_error("getFeatureInTensor size mismatch");
  }
  dest.copy_(temp);
}

inline torch::Tensor getFeatureInTensor(const State& state) {
  torch::Tensor t = torch::zeros(state.GetFeatureSize(), torch::kFloat32);
  getFeatureInTensor(state, t);
  return t;
}

inline void getRawFeatureInTensor(const State& state, torch::Tensor dest) {
  assert(dest.dtype() == torch::kFloat32);
  auto& feat = state.GetRawFeatures();
  torch::Tensor temp = torch::from_blob(
      (void*)feat.data(), state.GetRawFeatureSize(), dest.dtype());
  if (feat.size() != temp.numel()) {
    throw std::runtime_error("getRawFeatureInTensor size mismatch");
  }
  dest.copy_(temp);
}

inline torch::Tensor getRawFeatureInTensor(const State& state) {
  torch::Tensor t = torch::zeros(state.GetRawFeatureSize(), torch::kFloat32);
  getRawFeatureInTensor(state, t);
  return t;
}

inline void getPolicyMaskInTensor(
    const State& state, torch::TensorAccessor<float, 3> maskaccessor) {
  for (const auto& action : state.GetLegalActions()) {
    maskaccessor[action.GetX()][action.GetY()][action.GetZ()] = 1;
  }
}

inline void getPolicyMaskInTensor(const State& state, torch::Tensor& mask) {
  assert(state.GetActionSize().size() == 3);
  auto maskaccessor = mask.accessor<float, 3>();
  getPolicyMaskInTensor(state, maskaccessor);
}

inline torch::Tensor getPolicyMaskInTensor(const State& state) {
  torch::Tensor mask = torch::zeros(state.GetActionSize(), torch::kFloat32);
  getPolicyMaskInTensor(state, mask);
  return mask;
}

inline void getPolicyInTensor(const State& state,
                              const std::vector<float>& pi,
                              torch::Tensor& dest,
                              torch::Tensor& mask) {
  assert(dest.dtype() == torch::kFloat32);
  assert(state.GetActionSize().size() == 3);
  auto accessor = dest.accessor<float, 3>();
  auto maskaccessor = mask.accessor<float, 3>();

  const auto& legalAction = state.GetLegalActions();
  for (mcts::Action actionIdx = 0; actionIdx != pi.size(); ++actionIdx) {
    if (actionIdx >= (int)legalAction.size() || actionIdx < 0) {
      std::cout << "Wrong action in getPolicyTargetInTensor, "
                << "action idx: " << actionIdx
                << ", num legal: " << legalAction.size() << std::endl;
      std::terminate();
      assert(false);
    }
    const auto& action = legalAction[actionIdx];
    float piVal = pi[actionIdx];
    int x = action.GetX();
    int y = action.GetY();
    int z = action.GetZ();
    accessor[x][y][z] += piVal;
    maskaccessor[x][y][z] = 1;
  }
}

inline std::pair<torch::Tensor, torch::Tensor> getPolicyInTensor(
    const State& state, const std::vector<float>& pi) {
  torch::Tensor t = torch::zeros(state.GetActionSize(), torch::kFloat32);
  torch::Tensor mask = torch::zeros(state.GetActionSize(), torch::kFloat32);
  getPolicyInTensor(state, pi, t, mask);
  return std::make_pair(t, mask);
}

inline void normalize(std::vector<float>& a2pi) {
  float sumProb = 0.0f;
  for (auto& p : a2pi) {
    sumProb += p;
  }

  if (sumProb != 0.0f) {
    for (auto& p : a2pi) {
      p /= sumProb;
    }
  }
}

inline void getLegalPi(const State& state,
                       torch::TensorAccessor<float, 3> accessor,
                       std::vector<float>& out) {
  const auto& legalActions = state.GetLegalActions();
  out.resize(legalActions.size());
  for (size_t i = 0; i != legalActions.size(); ++i) {
    const auto& action = legalActions[i];
    float pi = accessor[action.GetX()][action.GetY()][action.GetZ()];
    out[i] = pi;
  }
  normalize(out);
}

inline void getLegalPi(const State& state,
                       const torch::Tensor& pi,
                       std::vector<float>& out) {
  auto accessor = pi.accessor<float, 3>();
  return getLegalPi(state, accessor, out);
}

inline int64_t product(const std::vector<int64_t>& nums) {
  int64_t p = 1;
  for (auto n : nums) {
    p *= n;
  }
  return p;
}
