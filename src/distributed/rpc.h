
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd/lib/zstd.h"

#include "network.h"

#include "string_view"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace rpc {

// This is not a cross platform serializer
struct Serializer {
  std::vector<char> buf;
  void write(const void* data, size_t len) {
    size_t offset = buf.size();
    if (buf.capacity() < offset + len) {
      buf.reserve(
          std::max(offset + len, std::max(buf.capacity() * 2, (size_t)16)));
    }
    buf.resize(offset + len);
    std::memcpy(buf.data() + offset, data, len);
  }
  template <typename T, std::enable_if_t<std::is_trivial_v<T>>* = nullptr>
  void write(T v) {
    write((void*)&v, sizeof(v));
  }

  void write(std::string_view str) {
    write(str.size());
    write(str.data(), str.size());
  }

  template<typename T>
  void write(std::basic_string_view<T> str) {
    write(str.size());
    write(str.data(), sizeof(T) * str.size());
  }

  //  template <typename Key, typename Value, typename... A>
  //  void write(const std::unordered_map<Key, Value, A...>& v) {
  //    writeMap(v);
  //  }

  //  void write(torch::Tensor v) {
  //    std::ostringstream os;
  //    torch::save(v, os);
  //    write(os.str());
  //  }

  //  template <typename Key, typename Value, typename... A>
  //  void writeMap(const std::unordered_map<Key, Value, A...>& v) {
  //    write(v.size());
  //    for (auto& x : v) {
  //      write(x.first);
  //      write(x.second);
  //    }
  //  }

  void clear() {
    buf.clear();
  }
  const char* data() const {
    return buf.data();
  }
  size_t size() const {
    return buf.size();
  }

  void compress(int level = 0) {
    std::vector<char> newbuf;
    newbuf.resize(sizeof(size_t) + ZSTD_compressBound(buf.size()));
    auto n = ZSTD_compress(newbuf.data() + sizeof(size_t),
                           newbuf.size() - sizeof(size_t),
                           buf.data(),
                           buf.size(),
                           level);
    if (!ZSTD_isError(n)) {
      size_t sn = buf.size();
      std::memcpy(newbuf.data(), &sn, sizeof(sn));
      newbuf.resize(sizeof(size_t) + n);
      std::swap(buf, newbuf);
    } else {
      buf.clear();
    }
  }
};
struct Deserializer {
  std::string_view buf;
  std::vector<char> ownbuf;
  Deserializer() = default;
  Deserializer(std::string_view buf)
      : buf(buf) {
  }
  Deserializer(const void* data, size_t len)
      : buf((const char*)data, len) {
  }
  void consume(size_t len) {
    buf = {buf.data() + len, buf.size() - len};
  }
  template<typename T>
  std::basic_string_view<T> readStringView() {
    size_t len = read<size_t>();
    if (buf.size() < sizeof(T) * len) {
      len = buf.size() / sizeof(T);
    }
    T* data = (T*)buf.data();
    consume(sizeof(T) * len);
    return {data, len};
  }
  std::string_view readString() {
    size_t len = read<size_t>();
    if (buf.size() < len) {
      len = buf.size();
    }
    const char* data = buf.data();
    consume(len);
    return {data, len};
  }
  template <typename T, std::enable_if_t<std::is_trivial_v<T>>* = nullptr>
  void read(T& r) {
    if (buf.size() < sizeof(T)) {
      consume(buf.size());
      r = {};
      return;
    }
    std::memcpy(&r, buf.data(), sizeof(T));
    consume(sizeof(T));
  }
  void read(std::string_view& r) {
    r = readString();
  }
  void read(std::string& r) {
    r = readString();
  }
  template<typename T>
  void read(std::basic_string_view<T>& r) {
    r = readStringView<T>();
  }
  //  template <typename Key, typename Value, typename... A>
  //  void read(std::unordered_map<Key, Value, A...>& v) {
  //    readMap(v);
  //  }

  //  void read(torch::Tensor& v) {
  //    auto s = read();
  //    std::string str(s.data(), s.size());
  //    std::istringstream is(str);
  //    torch::load(v, is);
  //  }

  template <typename T> T read() {
    T r;
    read(r);
    return r;
  }
  std::string_view read() {
    return readString();
  }
  //  template <typename Key, typename Value, typename... A>
  //  void readMap(std::unordered_map<Key, Value, A...>& v) {
  //    v.clear();
  //    size_t n = read<size_t>();
  //    for (; n; --n) {
  //      auto k = read<Key>();
  //      v.emplace(std::move(k), read<Value>());
  //    }
  //  }

  bool empty() {
    return buf.empty();
  }

  void decompress() {
    size_t sn = read<size_t>();
    std::vector<char> newbuf;
    newbuf.resize(sn);
    auto n =
        ZSTD_decompress(newbuf.data(), newbuf.size(), buf.data(), buf.size());
    if (!ZSTD_isError(n)) {
      std::swap(ownbuf, newbuf);
      buf = {ownbuf.data(), ownbuf.size()};
    } else {
      buf = {};
    }
  }
};

template <typename T> struct AssertHelper { static const bool value = false; };

struct Serialize {
  Serialize(Serializer& ser)
      : ser(ser) {
  }
  Serializer& ser;

  template <typename T> static std::false_type has_serialize_f(...);
  template <typename T,
            typename = decltype(
                std::declval<T>().serialize(std::declval<Serialize&>()))>
  static std::true_type has_serialize_f(int);
  template <typename T>
  static const bool has_serialize =
      decltype(Serialize::has_serialize_f<T>(0))::value;
  template <typename T> static std::false_type has_builtin_write_f(...);
  template <
      typename T,
      typename = decltype(std::declval<Serializer>().write(std::declval<T>()))>
  static std::true_type has_builtin_write_f(int);
  template <typename T>
  static const bool has_builtin_write =
      decltype(Serialize::has_builtin_write_f<T>(0))::value;
  template <typename T> void operator()(const T& v) {
    if constexpr (has_serialize<const T>) {
      v.serialize(*this);
    } else if constexpr (has_builtin_write<const T>) {
      ser.write(std::forward<const T>(v));
    } else {
      serialize(*this, std::forward<const T>(v));
      // static_assert(AssertHelper<T>::value, "Don't know how to serialize this
      // type");
    }
  }

  template <typename... T> void operator()(const T&... v) {
    (int[]){((*this)(std::forward<const T>(v)), 0)...};
  }
};

struct Deserialize {
  Deserialize(Deserializer& des)
      : des(des) {
  }
  Deserializer& des;

  template <typename T> static std::false_type has_serialize_f(...);
  template <typename T,
            typename = decltype(
                std::declval<T>().serialize(std::declval<Deserialize&>()))>
  static std::true_type has_serialize_f(int);
  template <typename T>
  static const bool has_serialize =
      decltype(Deserialize::has_serialize_f<T>(0))::value;
  template <typename T> static std::false_type has_builtin_read_f(...);
  template <typename T,
            typename =
                decltype(std::declval<Deserializer>().read(std::declval<T&>()))>
  static std::true_type has_builtin_read_f(int);
  template <typename T>
  static const bool has_builtin_read =
      decltype(Deserialize::has_builtin_read_f<T>(0))::value;
  template <typename T> void operator()(T& v) {
    if constexpr (has_serialize<T>) {
      v.serialize(*this);
    } else if constexpr (has_builtin_read<T>) {
      des.read(v);
    } else {
      serialize(*this, v);
      // static_assert(AssertHelper<T>::value, "Don't know how to serialize this
      // type");
    }
  }

  template <typename... T> void operator()(T&... v) {
    (int[]){((*this)(v), 0)...};
  }

  template <typename T> T read() {
    if constexpr (has_serialize<T>) {
      T r;
      r.serialize(*this);
      return r;
    } else if constexpr (has_builtin_read<T>) {
      return des.read<T>();
    } else {
      T r;
      serialize(*this, r);
      return r;
      // static_assert(AssertHelper<T>::value, "Don't know how to serialize this
      // type");
    }
  }
};

struct RPCException : std::exception {};

struct RPCExceptionConnectionError : RPCException {
  virtual const char* what() const noexcept override {
    return "RPC connection error";
  }
};

struct RPCExceptionFunctionNotFound : RPCException {
  virtual const char* what() const noexcept override {
    return "RPC function not found";
  }
};

struct RPCExceptionRemoteException : RPCException {
  virtual const char* what() const noexcept override {
    return "RPC remote exception";
  }
};

class Client : public std::enable_shared_from_this<Client> {
 public:
  Client() = default;
  Client(network::Peer peer)
      : peer(std::move(peer)) {
    this->peer.setOnMessage([this](std::string_view buf) {
      bytesReceived_ += buf.size();
      Deserializer des(buf);
      des.decompress();
      Deserialize x(des);
      uint32_t id;
      uint8_t status;
      x(id, status);
      std::unique_lock l(reqmut);
      auto i = requests.find(id);
      if (i != requests.end()) {
        auto r = std::move(i->second);
        requests.erase(i);
        l.unlock();
        lastLatency_ = std::chrono::steady_clock::now() - r->timestamp;
        if (status == 0xff) {
          r->exception(std::make_exception_ptr(RPCExceptionFunctionNotFound()));
        } else if (status == 0xfe) {
          r->exception(std::make_exception_ptr(RPCExceptionRemoteException()));
        } else if (status != 0) {
          r->exception(std::make_exception_ptr(RPCExceptionConnectionError()));
        } else {
          r->handle(x);
        }
      }
    });
    this->peer.setOnConnectionClosed([this]() {
      std::unique_lock l(reqmut);
      for (auto& v : requests) {
        v.second->exception(
            std::make_exception_ptr(RPCExceptionConnectionError()));
      }
      requests.clear();
    });
  }
  ~Client() {
    peer.close();
  }

  void close() {
    peer.close();
  }

  template <typename... Args>
  void async(std::string_view funcname, Args&&... args) {
    Serializer ser;
    Serialize x(ser);
    uint32_t id = ++reqcounter;
    x(id, funcname, std::forward<Args>(args)...);
    ser.compress();
    peer.sendMessage(ser.data(), ser.size());
    bytesSent_ += ser.size();
    ++numRpcCalls_;
  }

  struct RequestBase {
    std::chrono::steady_clock::time_point timestamp;
    virtual ~RequestBase() {
    }
    virtual void handle(Deserialize&) noexcept = 0;
    virtual void exception(std::exception_ptr e) noexcept = 0;
  };

  template <typename R> struct RequestImpl : RequestBase {
    std::promise<R> p;
    std::atomic_bool handled = false;
    virtual ~RequestImpl() {
    }
    virtual void handle(Deserialize& x) noexcept override {
      if (handled.exchange(true)) {
        return;
      }
      if constexpr (std::is_same_v<void, R>) {
        p.set_value();
      } else {
        R r;
        x(r);
        p.set_value(r);
      }
    }
    virtual void exception(std::exception_ptr e) noexcept override {
      if (handled.exchange(true)) {
        return;
      }
      p.set_exception(e);
    }
  };

  template <typename R, typename... Args>
  std::future<R> async(std::string_view funcname, Args&&... args) {
    Serializer ser;
    Serialize x(ser);
    uint32_t id = ++reqcounter;
    x(id, funcname, std::forward<Args>(args)...);
    auto req = std::make_unique<RequestImpl<R>>();
    auto fut = req->p.get_future();
    std::unique_lock l(reqmut);
    req->timestamp = std::chrono::steady_clock::now();
    requests[id] = std::move(req);
    l.unlock();
    ser.compress();
    peer.sendMessage(ser.data(), ser.size());
    bytesSent_ += ser.size();
    ++numRpcCalls_;
    return fut;
  }

  template <typename R, typename... Args>
  R sync(std::string_view funcname, Args&&... args) {
    auto f = async<R>(funcname, std::forward<Args>(args)...);
    return f.get();
  }

  template <typename... Args>
  void sync(std::string_view funcname, Args&&... args) {
    return sync<void>(funcname, std::forward<Args>(args)...);
  }

  size_t bytesSent() const {
    return bytesSent_;
  }
  size_t bytesReceived() const {
    return bytesReceived_;
  }
  size_t numRpcCalls() const {
    return numRpcCalls_;
  }
  std::chrono::steady_clock::duration lastLatency() const {
    return lastLatency_;
  }

 private:
  std::mutex reqmut;
  std::unordered_map<uint32_t, std::unique_ptr<RequestBase>> requests;
  std::atomic<uint32_t> reqcounter = 0;
  network::Peer peer;

  std::atomic_size_t bytesSent_ = 0;
  std::atomic_size_t bytesReceived_ = 0;
  std::atomic_size_t numRpcCalls_ = 0;
  std::atomic<std::chrono::steady_clock::duration> lastLatency_{};
};

class Server {
 public:
  Server() = default;
  Server(network::Server server)
      : server(std::move(server)) {
    this->server.setOnPeer([this](network::Peer peer) {
      auto ref = std::make_shared<Peer>();
      ref->peer = std::move(peer);
      auto l = this->server.lock();
      peers.push_back(ref);
      l.unlock();
      ref->peer.setOnMessage(
          [this, ref](std::string_view buf) { handle(*ref, buf); });
      ref->peer.setOnConnectionClosed([this, ref]() {
        auto l = this->server.lock();
        for (auto i = peers.begin(); i != peers.end(); ++i) {
          if (*i == ref) {
            peers.erase(i);
            break;
          }
        }
        ref->peer.post_close();
      });
    });
  }
  ~Server() {
    server.close();
    for (auto& v : peers) {
      v->peer.close();
    }
    peers.clear();
  }

  Server(Server&&) = delete;
  Server(Server&) = delete;

  void listen(std::string_view endpoint) {
    server.listen(endpoint);
  }

  struct FBase {
    virtual ~FBase(){};
    virtual void call(Deserialize& x, Serialize& sx) = 0;
  };

  template <typename R, typename... Args> struct FImpl : FBase {
    std::function<R(Args...)> f;
    FImpl(std::function<R(Args...)> f)
        : f(std::move(f)) {
    }
    virtual ~FImpl(){};
    virtual void call(Deserialize& x, Serialize& sx) override {
      std::tuple<Args...> args;
      unfold<0>(x, args);
      if constexpr (std::is_same_v<void, R>) {
        std::apply(f, std::move(args));
      } else {
        sx(std::apply(f, std::move(args)));
      }
    }
    template <size_t n, typename T> void unfold(Deserialize& x, T& tuple) {
      x(std::get<n>(tuple));
      if constexpr (n + 1 != std::tuple_size_v<T>) {
        unfold<n + 1>(x, tuple);
      }
    }
  };

  std::unordered_set<std::string> funcnames;
  std::unordered_map<std::string_view, std::unique_ptr<FBase>> funcs;

  template <typename R, typename... Args>
  void define(std::string_view name, std::function<R(Args...)> f) {
    auto ff = std::make_unique<FImpl<R, Args...>>(std::move(f));
    funcs[*funcnames.emplace(name).first] = std::move(ff);
  }

  template <typename R, typename... Args>
  void define(std::string_view name, R (*f)(Args...)) {
    define(name, std::function<R(Args...)>(f));
  }

  template <typename R, typename M, typename... Args>
  void define(std::string_view name, R (M::*f)(Args...), M* self) {
    define(name, std::function<R(Args...)>([self, f](Args&&... args) -> R {
             return (self->*f)(std::forward<Args>(args)...);
           }));
  }

  template <typename R, typename... Args, typename T>
  void define(std::string_view name, T f) {
    auto ff = std::make_unique<FImpl<R, Args...>>(std::move(f));
    funcs[*funcnames.emplace(name).first] = std::move(ff);
  }

  size_t bytesSent() const {
    return bytesSent_;
  }
  size_t bytesReceived() const {
    return bytesReceived_;
  }
  size_t numRpcCalls() const {
    return numRpcCalls_;
  }

 private:
  struct Peer {
    network::Peer peer;
  };

  struct Message {
    Message* next = nullptr;
    std::vector<char> buf;
  };

  void handle(Peer& peer, std::string_view buf) {
    bytesReceived_ += buf.size();
    Deserializer des(buf.data(), buf.size());
    des.decompress();
    Deserialize x(des);
    uint32_t id;
    std::string_view name;
    x(id, name);
    Serializer ser;
    Serialize sx(ser);
    ++numRpcCalls_;
    auto i = funcs.find(name);
    if (i != funcs.end()) {
      sx(id);
      sx((uint8_t)0);
      try {
        i->second->call(x, sx);
      } catch (...) {
        ser.clear();
        sx(id);
        sx((uint8_t)0xfe);
        ser.compress();
        peer.peer.sendMessage(ser.data(), ser.size());
        bytesSent_ += ser.size();
        throw;
      }
    } else {
      sx(id);
      sx((uint8_t)0xff);
    }
    ser.compress();
    peer.peer.sendMessage(ser.data(), ser.size());
    bytesSent_ += ser.size();
  }

  network::Server server;
  std::vector<std::shared_ptr<Peer>> peers;

  std::atomic_size_t bytesSent_ = 0;
  std::atomic_size_t bytesReceived_ = 0;
  std::atomic_size_t numRpcCalls_ = 0;
};

class Rpc {
 public:
  Rpc() = default;
  Rpc(Rpc&&) = delete;
  Rpc(Rpc&) = delete;
  ~Rpc() {
    terminate = true;
    for (auto& _ : threads) {
      (void)_;
      net.post([] {});
    }
  }

  std::shared_ptr<Server> listen(std::string_view endpoint) {
    return std::make_shared<Server>(net.listen(endpoint));
  }

  std::shared_ptr<Client> connect(std::string_view endpoint) {
    return std::make_shared<Client>(net.connect(endpoint));
  }

  bool run_one() {
    return net.run_one();
  }

  void asyncRun(int nThreads = 1) {
    for (; nThreads; --nThreads) {
      threads.emplace_back([this]() {
        while (!terminate) {
          net.run_one();
        }
      });
    }
  }

 private:
  network::Network net;

  std::atomic_bool terminate = false;
  std::vector<std::thread> threads;
};

}  // namespace rpc
