#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>

namespace rdma {

class Error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Endpoint {
  uint32_t lid;
  uint32_t qpnum;
};

struct Buffer {
  virtual ~Buffer() {
  }
  virtual uint32_t key() = 0;
};

struct Host {
  virtual ~Host() {
  }
  virtual Endpoint init() = 0;
  virtual void connect(Endpoint ep) = 0;

  virtual void read(Buffer& localBuffer,
                    void* localAddress,
                    uint32_t remoteKey,
                    uintptr_t remoteAddress,
                    size_t size) = 0;
  virtual void wait() = 0;
};

struct Context {
  virtual ~Context() {
  }
  virtual std::unique_ptr<Host> createHost() = 0;
  virtual std::unique_ptr<Buffer> createBuffer(void* address, size_t size) = 0;
};

std::unique_ptr<Context> create();

}  // namespace rdma
