
#include "network.h"

#include "asio.hpp"

#include <deque>
#include <list>
#include <vector>

namespace network {

template <typename T> struct Handle {
  T* obj = nullptr;
  Handle() {
  }
  Handle(T& obj)
      : obj(&obj) {
    ++obj.refcount;
  }
  Handle(std::nullptr_t) {
  }
  Handle(Handle&& n) {
    obj = std::exchange(n.obj, nullptr);
  }
  Handle(const Handle& n) {
    acquire(n.obj);
  }
  Handle& operator=(Handle&& n) {
    std::swap(obj, n.obj);
    return *this;
  }
  Handle& operator=(const Handle& n) {
    acquire(n.obj);
    return *this;
  }
  void acquire(T* newobj) {
    release();
    obj = newobj;
    if (obj)
      ++obj->refcount;
  }
  void release() {
    if (obj) {
      if (--obj->refcount == 0) {
        obj->owner->free(obj);
      }
      obj = nullptr;
    }
  }
  ~Handle() {
    release();
  }
  T& operator*() const {
    return *obj;
  }
  T* operator->() const {
    return obj;
  }
  explicit operator bool() const {
    return obj;
  }
};

template <typename T> struct Ref {
  std::atomic_int refcount = 0;
  Handle<T> ref() {
    return *(T*)this;
  }
};

template <typename T> struct Cache {
  struct entry {
    entry* freenext = nullptr;
    Cache* owner = nullptr;
    std::aligned_storage_t<sizeof(T), alignof(T)> buf;
    entry* storagenext = nullptr;
  };
  std::atomic<entry*> storagelist;
  std::atomic<entry*> freelist;
  static entry* get(T* ptr) {
    uintptr_t v = (uintptr_t)(void*)ptr;
    v -= offsetof(entry, buf);
    return (entry*)v;
  }
  template <typename... A> T* allocate(A&&... args) {
    T* r;
    entry* e = freelist;
    while (e && !freelist.compare_exchange_weak(e, e->freenext))
      ;
    if (!e) {
      e = new entry();
      entry* s = storagelist;
      do {
        e->storagenext = s;
      } while (!storagelist.compare_exchange_weak(s, e));
    }
    r = (T*)&e->buf;
    new (r) T(std::forward<A>(args)...);
    e->owner = this;
    return r;
  }
  void free(T* obj) {
    entry* e = get(obj);
    if (e->owner != this) {
      std::terminate();
    }
    e->owner = nullptr;
    obj->~T();
    entry* f = freelist;
    do {
      e->freenext = f;
    } while (!freelist.compare_exchange_weak(f, e));
  }

  ~Cache() {
    for (entry* e = storagelist; e; e = e->storagenext) {
      if (e->owner) {
        ((T&)e->buf).~T();
      }
    }
  }

  template <typename... A> Handle<T> make(A&&... args) {
    return (allocate(this, std::forward<A>(args)...))->ref();
  }
};

template <typename T> struct Wrapper : Ref<Wrapper<T>> {
  Cache<Wrapper<T>>* owner = nullptr;
  T obj;
  template <typename... A>
  Wrapper(Cache<Wrapper<T>>* owner, A&&... args)
      : owner(owner)
      , obj(std::forward<A>(args)...) {
  }
  T* operator->() {
    return &obj;
  }
  T& operator*() {
    return obj;
  }
};

template <size_t maxsize> struct Buffer : Ref<Buffer<maxsize>> {
  Cache<Buffer>* owner = nullptr;
  std::array<char, maxsize> buf;
  size_t begin = 0;
  size_t end = 0;
  Buffer(Cache<Buffer>* owner)
      : owner(owner) {
  }
  size_t space() {
    return buf.size() - end;
  }
  size_t append(const void* data, size_t n) {
    n = std::min(n, space());
    std::memcpy(buf.data() + end, data, n);
    end += n;
    return n;
  }
  void free(size_t n) {
    begin += n;
  }
  const void* data() const {
    return buf.data() + begin;
  }
  size_t size() const {
    return end - begin;
  }
  bool empty() const {
    return size() == 0;
  }
};

std::pair<std::string_view, int> decodeEndpoint(std::string_view endpoint) {
  std::string_view hostname = endpoint;
  int port = 0;
  auto bpos = endpoint.find('[');
  if (bpos != std::string_view::npos) {
    auto bepos = endpoint.find(']', bpos);
    if (bepos != std::string_view::npos) {
      hostname = endpoint.substr(bpos + 1, bepos - (bpos + 1));
      endpoint = endpoint.substr(bepos + 1);
    }
  }
  auto cpos = endpoint.find(':');
  if (cpos != std::string_view::npos) {
    if (hostname == endpoint)
      hostname = endpoint.substr(0, cpos);
    ++cpos;
    while (cpos != endpoint.size()) {
      char c = endpoint[cpos];
      if (c < '0' || c > '9')
        break;
      port *= 10;
      port += c - '0';
      ++cpos;
    }
  }
  return {hostname, port};
}

class PeerImpl : public Ref<PeerImpl> {
 public:
  Cache<PeerImpl>* owner = nullptr;
  asio::io_context& context;
  Cache<Wrapper<asio::ip::tcp::resolver>> resolverCache;
  Cache<Wrapper<asio::steady_timer>> timerCache;
  Cache<Wrapper<asio::ip::tcp::socket>> socketCache;
  Cache<Buffer<0x10000>> bufferCache;
  asio::ip::tcp::socket socket;
  std::string connectEndpoint;
  std::string remoteHost;
  int remotePort = 0;
  bool connected = false;
  bool closed = false;
  std::vector<char> readBuffer;
  Handle<Buffer<0x10000>> writeBuffer;
  std::vector<Handle<Buffer<0x10000>>> writeBufferQueue;
  std::mutex mutex;
  PeerImpl(Cache<PeerImpl>* owner, asio::io_context& context)
      : owner(owner)
      , context(context)
      , socket(context) {
  }
  void connect(std::string_view endpoint) {
    auto l = lock();
    if (connected)
      return;
    if (endpoint.empty())
      return;
    if (connectEndpoint != endpoint)
      connectEndpoint = endpoint;
    auto [hostname, port] = decodeEndpoint(endpoint);

    remoteHost = hostname;
    remotePort = port;

    asio::error_code ec;
    asio::ip::address address = asio::ip::make_address(hostname, ec);
    if (ec) {
      auto resolver = resolverCache.make(context);
      (*resolver)->async_resolve(
          remoteHost, "",
          [this, str = std::string(remoteHost), peer = ref(), resolver,
           port = port](const asio::error_code& ec,
                        asio::ip::tcp::resolver::results_type results) mutable {
            if (!ec) {
              int n = 0;
              for (auto ep : results) {
                auto timer = timerCache.make(context);
                (*timer)->expires_from_now(std::chrono::seconds(n));
                (*timer)->async_wait(
                    [ep, timer, peer, port](const asio::error_code& ec) {
                      if (!ec) {
                        peer->connect(
                            {ep.endpoint().address(), (unsigned short)port});
                      }
                    });
                ++n;
              }
            } else {
              printf("resolve(%s): %s\n", str.c_str(), ec.message().c_str());
            }
          });
    } else {
      connect({address, (unsigned short)port});
    }

    auto timer = timerCache.make(context);
    (*timer)->expires_from_now(std::chrono::seconds(30));
    (*timer)->async_wait(
        [this, timer, peer = ref()](const asio::error_code& ec) {
          if (!ec) {
            connect(connectEndpoint);
          }
        });
  }

  void asyncRead(size_t offset = 0) {
    socket.async_receive(
        asio::buffer(readBuffer.data() + offset, readBuffer.size() - offset),
        [this, peer = ref()](auto&&... args) mutable {
          onReceive(std::forward<decltype(args)>(args)...);
        });
  }

  void setConnected(asio::ip::tcp::socket sock) {
    if (connected)
      return;
    if (closed) {
      sock.close();
      return;
    }
    connected = true;
    socket = std::move(sock);

    if (onReceiveCallback || onMessageCallback)
      asyncRead();
    flush();
  }

  void connect(asio::ip::tcp::endpoint ep) {
    if (connected || closed)
      return;

    auto h = socketCache.make(context);

    (*h)->async_connect(
        ep, [this, ep, h, peer = ref()](const asio::error_code& ec) {
          auto l = lock();
          if (!ec && !connected) {
            setConnected(std::move(**h));
          } else if (ec) {
            printf("connect(%s:%d): %s\n", ep.address().to_string().c_str(),
                   (int)ep.port(), ec.message().c_str());
          }
        });
  }

  void failure() {
    auto l = lock();
    if (!connected)
      return;
    connected = false;
    socket.close();
    writeBuffer = nullptr;
    CallbackCounter cc(activeCallbacks);
    l.unlock();
    if (onConnectionClosed) {
      onConnectionClosed();
    }
    auto timer = timerCache.make(context);
    (*timer)->expires_from_now(std::chrono::seconds(5));
    (*timer)->async_wait(
        [this, timer, peer = ref()](const asio::error_code& ec) {
          if (!ec) {
            connect(connectEndpoint);
          }
        });
  }

  std::atomic<bool> sending = false;

  std::vector<char> sendBuffer;

  void callSend(std::vector<char> buffer, size_t offset) {
    auto ab = asio::buffer(buffer.data() + offset, buffer.size() - offset);
    socket.async_send(
        ab, [this, peer = ref(), buffer = std::move(buffer), offset](
                const asio::error_code& ec, size_t n) mutable {
          if (ec) {
            sending = false;
            failure();
          } else {
            size_t remaining = buffer.size() - offset - n;
            if (remaining) {
              auto l = lock();
              callSend(std::move(buffer), offset + n);
            } else {
              sending = false;
              auto l = lock();
              flush();
            }
          }
        });
  }

  void flush() {
    if (!connected)
      return;
    if (!sendBuffer.empty()) {
      if (sending.exchange(true))
        return;
      callSend(std::move(sendBuffer), 0);
      sendBuffer.clear();
    }
  }

  void sendNoFlush(const void* data, size_t n) {
    size_t offset = sendBuffer.size();
    if (offset + n > sendBuffer.capacity()) {
      sendBuffer.reserve(std::max(offset + n, offset * 2));
    }
    sendBuffer.resize(offset + n);
    std::memcpy(sendBuffer.data() + offset, data, n);
  }

  //  std::vector<asio::const_buffer> sendBuffers;
  //
  //  void flush() {
  //    if (!connected) return;
  //    if (sending) return;
  //    sendBuffers.clear();
  //    if (!writeBufferQueue.empty()) {
  //      for (auto& v : writeBufferQueue) {
  //        sendBuffers.push_back(asio::buffer(v->data(), v->size()));
  //      }
  //      writeBufferQueue.clear();
  //    }
  //    size_t freeN = 0;
  //    if (writeBuffer && !writeBuffer->empty()) {
  //      sendBuffers.push_back(asio::buffer(writeBuffer->data(),
  //      writeBuffer->size())); freeN = writeBuffer->size();
  //    }
  //    if (!sendBuffers.empty()) {
  //      sending = true;
  //      socket.async_send(sendBuffers, [this, freeSrc = writeBuffer->ref(),
  //      freeN, peer = ref()](const asio::error_code& ec, size_t n) {
  //        sending = false;
  //        freeSrc->free(freeN);
  //        if (ec) {
  //          failure();
  //        } else {
  //          flush();
  //        }
  //      });
  //    }
  //  }

  //  void sendNoFlush(const void* data, size_t n) {
  //    if (!writeBuffer || writeBuffer->space() == 0) {
  //      writeBuffer = bufferCache.make();
  //    }
  //    size_t s = writeBuffer->append(data, n);
  //    while (writeBuffer->space() == 0) {
  //      writeBufferQueue.push_back(writeBuffer);
  //      writeBuffer = bufferCache.make();
  //      data = (const char*)data + s;
  //      n -= s;
  //      s = writeBuffer->append(data, n);
  //    }
  //  }

  void send(const void* data, size_t n) {
    auto l = lock();
    sendNoFlush(data, n);
    if (!sending) {
      flush();
    }
  }

  void sendMessage(const void* data, size_t n) {
    auto l = lock();
    uint32_t len = n;
    sendNoFlush(&len, sizeof(len));
    sendNoFlush(data, n);
    flush();
  }

  struct CallbackCounter {
    std::atomic_int& c;
    CallbackCounter(std::atomic_int& c)
        : c(c) {
      ++c;
    }
    ~CallbackCounter() {
      --c;
    }
  };
  std::atomic_int activeCallbacks;

  std::function<void()> onConnectionClosed;
  std::function<void(const void*, size_t)> onReceiveCallback;
  std::function<void(const void*, size_t)> onMessageCallback;

  void setOnReceive(std::function<void(const void*, size_t)> callback,
                    size_t bufferSize = 0x10000) {
    auto l = lock();
    if (!onReceiveCallback && connected) {
      asyncRead();
    }
    readBuffer.resize(bufferSize);
    onReceiveCallback = std::move(callback);
  }

  int messageState = -1;
  size_t messageReceived = 0;
  size_t messageLength = 0;

  void setOnMessage(std::function<void(const void*, size_t)> callback,
                    size_t bufferSize = 0x10000) {
    auto l = lock();
    if (!onMessageCallback && callback) {
      if (connected) {
        asyncRead();
      }
      readBuffer.resize(bufferSize);
      messageState = 0;
    }
    if (!callback) {
      messageState = -1;
    }
    onMessageCallback = std::move(callback);
  }

  void onReceive(const asio::error_code& ec, size_t n) {
    auto l = lock();
    if (closed) {
      return;
    }
    CallbackCounter cc(activeCallbacks);
    if (!ec) {
      while (true) {
        if (messageState == 0) {
          messageReceived += n;
          n = 0;
          if (messageReceived >= 4) {
            messageLength = *(uint32_t*)readBuffer.data();
            readBuffer.resize(std::max(readBuffer.size(), 4 + messageLength));
            messageState = 1;
            continue;
          } else {
            asyncRead(messageReceived);
          }
        } else if (messageState == 1) {
          messageReceived += n;
          n = 0;
          if (messageReceived >= 4 + messageLength) {
            std::vector<char> tmp(
                readBuffer.data() + 4, readBuffer.data() + 4 + messageLength);
            if (messageReceived == 4 + messageLength) {
              messageState = 0;
              messageReceived = 0;
              asyncRead(messageReceived);
              l.unlock();
            } else {
              std::memmove(readBuffer.data(),
                           readBuffer.data() + 4 + messageLength,
                           messageReceived - (4 + messageLength));
              messageReceived -= 4 + messageLength;
              messageState = 0;
              l.unlock();
              context.post([this]() { onReceive({}, 0); });
            }
            onMessageCallback(tmp.data(), tmp.size());
            break;
          } else {
            if (readBuffer.size() - messageReceived == 0) {
              readBuffer.resize(readBuffer.size() * 2);
            }
          }
          asyncRead(messageReceived);
        } else if (onReceiveCallback) {
          asyncRead();
          if (n) {
            l.unlock();
            onReceiveCallback(readBuffer.data(), n);
          }
        }
        break;
      }
    } else {
      l.unlock();
      failure();
    }
  }

  std::unique_lock<std::mutex> lock() {
    return std::unique_lock(mutex);
  }

  void close() {
    auto l = lock();
    if (connected) {
      connected = false;
      socket.close();
      writeBuffer = nullptr;
    }

    messageState = -1;
    closed = true;

    l.unlock();
    while (activeCallbacks) {
      std::this_thread::yield();
    }
  }

  void post_close() {
    context.post([peer = ref()] { peer->close(); });
  }

  void setOnConnectionClosed(std::function<void()> callback) {
    onConnectionClosed = std::move(callback);
  }
};

class ServerImpl : public Ref<ServerImpl> {
 public:
  Cache<ServerImpl>* owner = nullptr;
  asio::io_context& context;
  Cache<Wrapper<asio::ip::tcp::resolver>> resolverCache;
  Cache<Wrapper<asio::steady_timer>> timerCache;
  Cache<Wrapper<asio::ip::tcp::acceptor>> acceptorCache;
  Cache<Wrapper<asio::ip::tcp::socket>> socketCache;
  Cache<PeerImpl> peerCache;
  std::string listenEndpoint;
  std::function<void(Handle<PeerImpl> peer)> onPeer;
  bool bound = false;
  std::vector<Handle<Wrapper<asio::ip::tcp::socket>>> sockets;
  std::mutex mutex;
  ServerImpl(Cache<ServerImpl>* owner, asio::io_context& context)
      : owner(owner)
      , context(context) {
  }
  ~ServerImpl() {
    close();
  }

  void asyncAccept(Handle<Wrapper<asio::ip::tcp::acceptor>> h,
                   Handle<Wrapper<asio::ip::tcp::socket>> socket) {
    (*h)->async_accept(**socket, [this, server = ref(), h,
                                  socket](const asio::error_code& ec) {
      if (!ec) {
        if (onPeer) {
          auto peer = peerCache.make(context);
          peer->setConnected(std::move(**socket));
          onPeer(peer);
        }
      }
      asyncAccept(h, socket);
    });
  }

  void bind(asio::ip::tcp::endpoint ep) {
    auto retry = [&](asio::error_code ec) {
      printf("bind(%s:%d): %s\n", ep.address().to_string().c_str(),
             (int)ep.port(), ec.message().c_str());
      auto timer = timerCache.make(context);
      (*timer)->expires_from_now(std::chrono::seconds(30));
      (*timer)->async_wait(
          [this, ep, timer, server = ref()](const asio::error_code& ec) {
            if (!ec) {
              bind(ep);
            }
          });
    };

    auto h = acceptorCache.make(context);

    asio::error_code ec;
    (*h)->open(ep.protocol());
    (*h)->set_option(asio::socket_base::reuse_address(true));
    (*h)->bind(ep, ec);
    if (ec)
      return retry(ec);
    (*h)->listen(asio::socket_base::max_connections, ec);
    if (ec)
      return retry(ec);

    auto socket = socketCache.make(context);

    asyncAccept(h, socket);

    auto l = lock();
    sockets.push_back(socket);
  }

  void bind(std::string_view endpoint) {
    if (endpoint.empty()) {
      return;
    }
    bound = true;
    auto [hostname, port] = decodeEndpoint(endpoint);

    asio::error_code ec;
    asio::ip::address address;
    if (hostname != "*") {
      address = asio::ip::make_address(hostname, ec);
    }
    if (ec) {
      auto resolver = resolverCache.make(context);
      (*resolver)->async_resolve(
          hostname, "",
          [this, peer = ref(), resolver, port = port](
              const asio::error_code& ec,
              asio::ip::tcp::resolver::results_type results) mutable {
            if (!ec) {
              for (auto ep : results) {
                bind({ep.endpoint().address(), (unsigned short)port});
              }
            } else {
              auto timer = timerCache.make(context);
              (*timer)->expires_from_now(std::chrono::seconds(30));
              (*timer)->async_wait(
                  [this, timer, peer = ref()](const asio::error_code& ec) {
                    if (!ec) {
                      auto l = lock();
                      bind(listenEndpoint);
                    }
                  });
            }
          });
    } else {
      context.post([this, address, port = port, server = ref()] {
        bind({address, (unsigned short)port});
      });
    }
  }

  void listen(std::string_view endpoint) {
    auto l = lock();
    listenEndpoint = endpoint;

    if (onPeer)
      bind(endpoint);
  }

  void setOnPeer(std::function<void(Handle<PeerImpl>)> callback) {
    auto l = lock();
    onPeer = callback;
    if (!bound)
      bind(listenEndpoint);
  }

  std::unique_lock<std::mutex> lock() {
    return std::unique_lock(mutex);
  }

  void close() {
    auto l = lock();
    onPeer = nullptr;
    for (auto& v : sockets) {
      (*v)->close();
    }
    sockets.clear();
  }
};

class NetworkImpl {
 public:
  Cache<PeerImpl> peerCache;
  Cache<ServerImpl> serverCache;
  asio::io_context context;
  asio::executor_work_guard<asio::io_context::executor_type> work{
      context.get_executor()};

  ~NetworkImpl() {
    work.reset();
  }

  Handle<PeerImpl> connect(std::string_view endpoint) {
    auto h = peerCache.make(context);
    h->connect(endpoint);
    return h;
  }

  Handle<ServerImpl> listen(std::string_view endpoint) {
    auto h = serverCache.make(context);
    h->listen(endpoint);
    return h;
  }

  template <typename T>
  static std::unique_ptr<T, std::function<void(T*)>> wrap(Handle<T> h) {
    auto* ptr = &*h;
    return std::unique_ptr<T, std::function<void(T*)>>(
        ptr, [h = std::move(h)](T* ptr) mutable { h = nullptr; });
  }

  bool run_one() {
    return context.run_one() != 0;
  }

  void post(std::function<void()> f) {
    asio::post(std::move(f));
  }
};

Peer::Peer(std::unique_ptr<PeerImpl, std::function<void(PeerImpl*)>> impl)
    : impl(std::move(impl)) {
}

Peer::~Peer() {
}

void Peer::send(const void* data, size_t n) {
  return impl->send(data, n);
}

void Peer::send(std::string_view buf) {
  return send(buf.data(), buf.size());
}

void Peer::setOnReceive(std::function<void(const void*, size_t)> callback) {
  return impl->setOnReceive(std::move(callback));
}

void Peer::setOnReceive(std::function<void(std::string_view)> callback) {
  setOnReceive([callback = std::move(callback)](const void* data, size_t n) {
    return callback(std::string_view((const char*)data, n));
  });
}

void Peer::sendMessage(const void* data, size_t n) {
  return impl->sendMessage(data, n);
}

void Peer::sendMessage(std::string_view buf) {
  return sendMessage(buf.data(), buf.size());
}

void Peer::setOnMessage(std::function<void(const void*, size_t)> callback) {
  return impl->setOnMessage(std::move(callback));
}

void Peer::setOnMessage(std::function<void(std::string_view)> callback) {
  if (!callback)
    setOnMessage(nullptr);
  else
    setOnMessage([callback = std::move(callback)](const void* data, size_t n) {
      return callback(std::string_view((const char*)data, n));
    });
}

void Peer::setOnMessage(std::nullptr_t) {
  impl->setOnMessage(nullptr);
}

void Peer::setOnConnectionClosed(std::function<void()> callback) {
  impl->setOnConnectionClosed(std::move(callback));
}

bool Peer::connected() const {
  return impl->connected;
}

void Peer::close() {
  impl->close();
}

void Peer::post_close() {
  impl->post_close();
}

std::unique_lock<std::mutex> Peer::lock() {
  return impl->lock();
}

Server::Server(
    std::unique_ptr<ServerImpl, std::function<void(ServerImpl*)>> impl)
    : impl(std::move(impl)) {
}

Server::~Server() {
}

void Server::setOnPeer(std::function<void(Peer)> callback) {
  impl->setOnPeer([callback = std::move(callback)](Handle<PeerImpl> peer) {
    callback(NetworkImpl::wrap(peer));
  });
}

void Server::close() {
  impl->close();
}

std::unique_lock<std::mutex> Server::lock() {
  return impl->lock();
}

void Server::listen(std::string_view endpoint) {
  return impl->listen(endpoint);
}

Network::Network() {
  impl = std::make_unique<NetworkImpl>();
}

Network::Network(
    std::unique_ptr<NetworkImpl, std::function<void(NetworkImpl*)>> impl)
    : impl(std::move(impl)) {
}

Network::~Network() {
}

Peer Network::connect(std::string_view endpoint) {
  return impl->wrap(impl->connect(endpoint));
}

Server Network::listen(std::string_view endpoint) {
  return impl->wrap(impl->listen(endpoint));
}

bool Network::run_one() {
  return impl->run_one();
}

void Network::post(std::function<void()> f) {
  return impl->post(std::move(f));
}

}  // namespace network
