#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string_view>

namespace network {

class PeerImpl;
class ServerImpl;
class NetworkImpl;

class Peer {
 public:
  Peer() = default;
  Peer(Peer&&) = default;
  Peer(std::unique_ptr<PeerImpl, std::function<void(PeerImpl*)>> impl);
  ~Peer();
  Peer& operator=(Peer&&) = default;

  void send(const void* data, size_t n);
  void send(std::string_view buf);

  void setOnReceive(std::function<void(const void* data, size_t n)> callback);
  void setOnReceive(std::function<void(std::string_view)> callback);

  void sendMessage(const void* data, size_t n);
  void sendMessage(std::string_view buf);

  void setOnMessage(std::function<void(const void* data, size_t n)> callback);
  void setOnMessage(std::function<void(std::string_view)> callback);
  void setOnMessage(std::nullptr_t);

  void setOnConnectionClosed(std::function<void()> callback);

  explicit operator bool() const {
    return impl != nullptr;
  }

  bool connected() const;
  void close();
  void post_close();

  std::unique_lock<std::mutex> lock();

 private:
  std::unique_ptr<PeerImpl, std::function<void(PeerImpl*)>> impl;
};

class Server {
 public:
  Server() = default;
  Server(Server&&) = default;
  Server(std::unique_ptr<ServerImpl, std::function<void(ServerImpl*)>> impl);
  ~Server();
  Server& operator=(Server&&) = default;

  void setOnPeer(std::function<void(Peer)> callback);
  void close();
  std::unique_lock<std::mutex> lock();
  void listen(std::string_view endpoint);

 private:
  std::unique_ptr<ServerImpl, std::function<void(ServerImpl*)>> impl;
};

class Network {
 public:
  Network();
  Network(Network&&) = default;
  Network(std::unique_ptr<NetworkImpl, std::function<void(NetworkImpl*)>> impl);
  ~Network();

  Peer connect(std::string_view endpoint);
  Server listen(std::string_view endpoint);

  bool run_one();
  void post(std::function<void()> f);

 private:
  std::unique_ptr<NetworkImpl, std::function<void(NetworkImpl*)>> impl;
};

}  // namespace network
