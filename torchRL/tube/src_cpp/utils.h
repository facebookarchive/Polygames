/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace tube {

namespace utils {

inline int getProduct(const std::vector<int64_t>& nums) {
  int prod = 1;
  for (auto v : nums) {
    prod *= v;
  }
  return prod;
}

inline std::vector<int64_t> pushLeft(int64_t left,
                                     const std::vector<int64_t>& nums) {
  std::vector<int64_t> vec;
  vec.push_back(left);
  for (auto v : nums) {
    vec.push_back(v);
  }
  return vec;
}

template <typename T> inline void printVector(const std::vector<T>& vec) {
  for (const auto& v : vec) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;
}

template <typename T> inline void printMapKey(const T& map) {
  for (const auto& name2sth : map) {
    std::cout << name2sth.first << ", ";
  }
  std::cout << std::endl;
}

inline void verifyTensors(
    const std::unordered_map<std::string, torch::Tensor>& src,
    const std::unordered_map<std::string, torch::Tensor>& dest) {
  if (src.size() != dest.size()) {
    std::cout << "src.size()[" << src.size() << "] != dest.size()["
              << dest.size() << "]" << std::endl;
    std::cout << "src keys: ";
    for (const auto& p : src)
      std::cout << p.first << " ";
    std::cout << "dest keys: ";
    for (const auto& p : dest)
      std::cout << p.first << " ";
    std::cout << std::endl;
    assert(false);
  }

  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    // std::cout << "in copy: trying to get: " << name << std::endl;
    // std::cout << "dest map keys" << std::endl;
    // printMapKey(dest);
    const auto& destTensor = dest.at(name);
    // if (destTensor.sizes() != srcTensor.sizes()) {
    //   std::cout << "copy size-mismatch: "
    //             << destTensor.sizes() << ", " << srcTensor.sizes() <<
    //             std::endl;
    // }
    if (destTensor.sizes() != srcTensor.sizes()) {
      std::cout << name << ", dstSize: " << destTensor.sizes()
                << ", srcSize: " << srcTensor.sizes() << std::endl;
      assert(false);
    }

    if (destTensor.dtype() != srcTensor.dtype()) {
      std::cout << name << ", dstType: " << destTensor.dtype()
                << ", srcType: " << srcTensor.dtype() << std::endl;
      assert(false);
    }
  }
}

inline void copyTensors(
    const std::unordered_map<std::string, torch::Tensor>& src,
    std::unordered_map<std::string, torch::Tensor>& dest) {
  verifyTensors(src, dest);
  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    // std::cout << "in copy: trying to get: " << name << std::endl;
    // std::cout << "dest map keys" << std::endl;
    // printMapKey(dest);
    auto& destTensor = dest.at(name);
    // if (destTensor.sizes() != srcTensor.sizes()) {
    //   std::cout << "copy size-mismatch: "
    //             << destTensor.sizes() << ", " << srcTensor.sizes() <<
    //             std::endl;
    // }
    destTensor.copy_(srcTensor);
  }
}

// TODO: maybe merge these two functions?
inline void copyTensors(
    const std::unordered_map<std::string, torch::Tensor>& src,
    std::unordered_map<std::string, torch::Tensor>& dest,
    std::vector<int64_t>& index) {
  assert(src.size() == dest.size());
  assert(!index.empty());
  torch::Tensor indexTensor =
      torch::from_blob(index.data(), {(int64_t)index.size()}, torch::kInt64);

  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    auto& destTensor = dest.at(name);
    // assert(destTensor.sizes() == srcTensor.sizes());
    assert(destTensor.dtype() == srcTensor.dtype());
    destTensor.index_copy_(0, indexTensor, srcTensor);
  }
}

}  // namespace utils
}
