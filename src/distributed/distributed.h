/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <torch/torch.h>
#include <unordered_map>

namespace rpc {

template <typename X, typename A, typename B>
void serialize(X& x, const std::pair<A, B>& v) {
  x(v.first, v.second);
}

template <typename X, typename A, typename B>
void serialize(X& x, std::pair<A, B>& v) {
  x(v.first, v.second);
}

template <typename X, typename T>
void serialize(X& x, const std::optional<T>& v) {
  x(v.has_value());
  if (v.has_value()) {
    x(v.value());
  }
}

template <typename X, typename T> void serialize(X& x, std::optional<T>& v) {
  if (x.template read<bool>()) {
    v.emplace();
    x(v.value());
  } else {
    v.reset();
  }
}

template <typename X, typename T>
void serialize(X& x, const std::vector<T>& v) {
  x(v.size());
  for (auto& v2 : v) {
    x(v2);
  }
}

template <typename X, typename T> void serialize(X& x, std::vector<T>& v) {
  size_t n = x.template read<size_t>();
  v.resize(n);
  for (size_t i = 0; i != n; ++i) {
    x(v[i]);
  }
}

template <typename X, typename Key, typename Value>
void serialize(X& x, const std::unordered_map<Key, Value>& v) {
  x(v.size());
  for (auto& v2 : v) {
    x(v2.first, v2.second);
  }
}

template <typename X, typename Key, typename Value>
void serialize(X& x, std::unordered_map<Key, Value>& v) {
  v.clear();
  size_t n = x.template read<size_t>();
  for (; n; --n) {
    auto k = x.template read<Key>();
    v.emplace(std::move(k), x.template read<Value>());
  }
}

template <typename X> void serialize(X& x, const torch::Tensor& v) {
  if (!v.is_contiguous()) {
    serialize(x, v.contiguous());
    return;
  }
  x(v.scalar_type(),
    std::basic_string_view<int64_t>(v.sizes().data(), v.sizes().size()));
  void* data = v.data_ptr();
  size_t size = v.numel() * v.dtype().itemsize();
  x(std::string_view((const char*)data, size));
}

template <typename X> void serialize(X& x, torch::Tensor& v) {
  torch::ScalarType dtype;
  std::basic_string_view<int64_t> sizes;
  x(dtype, sizes);
  if (v.defined() && v.scalar_type() == dtype) {
    v.resize_(torch::IntArrayRef(sizes.begin(), sizes.end()));
  } else {
    v = torch::empty(torch::IntArrayRef(sizes.begin(), sizes.end()), dtype);
  }
  std::string_view data;
  x(data);
  if ((size_t)v.numel() != data.size() / v.dtype().itemsize()) {
    throw std::runtime_error("numel mismatch in tensor deserialize");
  }
  std::memcpy(v.data_ptr(), data.data(), data.size());
}

}  // namespace rpc

namespace distributed {

class ServerImpl;
class ClientImpl;

class Server {
  std::unique_ptr<ServerImpl> impl;

 public:
  Server();
  ~Server();

  void setOnTrainData(
      std::function<
          void(const std::unordered_map<std::string, torch::Tensor>)>);
  void start(std::string endpoint);
  void updateModel(const std::string& id,
                   std::unordered_map<std::string, torch::Tensor> stateDict);
};

class Client {
  std::unique_ptr<ClientImpl> impl;

 public:
  Client();
  ~Client();

  void setOnUpdateModel(
      std::function<void(std::string_view,
                         std::unordered_map<std::string, torch::Tensor>)>);
  void connect(std::string endpoint);
  void requestModel(bool isTournamentOpponent);
  void sendTrainData(
      const std::unordered_map<std::string, torch::Tensor>& data);
  bool wantsTournamentResult();
  std::string_view getModelId();
  void sendResult(float reward,
                  std::unordered_map<std::string_view, float> models);
};

}  // namespace distributed
