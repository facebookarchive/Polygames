#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

#include <semaphore.h>

namespace async {

struct Function {
  Function* next = nullptr;
  int priority = 0;
  void* storage;
  size_t allocated = 0;
  void (*dtor)(void*);
  void (*call)(void*);
  Function() = default;
  Function(const Function&) = delete;
  Function& operator=(const Function&) = delete;
  Function(Function&& n) {
    storage = n.storage;
    allocated = n.allocated;
    dtor = n.dtor;
    call = n.call;
    n.storage = nullptr;
  }
  Function& operator=(Function&& n) {
    std::swap(storage, n.storage);
    std::swap(allocated, n.allocated);
    std::swap(dtor, n.dtor);
    std::swap(call, n.call);
    return *this;
  }
  template <typename F> Function(F&& f) {
    storage = std::malloc(sizeof(F));
    if (!storage) {
      throw std::bad_alloc();
    }
    allocated = sizeof(F);
    try {
      new (storage) F(std::forward<F>(f));
    } catch (...) {
      std::free(storage);
      throw;
    }
    dtor = [](void* ptr) noexcept {
      ((F*)ptr)->~F();
    };
    call = [](void* ptr) noexcept {
      (*(F*)ptr)();
    };
  }
  template <typename F> Function& operator=(F&& f) {
    if (allocated < sizeof(F)) {
      void* newStorage = std::malloc(sizeof(F));
      if (!newStorage) {
        throw std::bad_alloc();
      }
      if (storage) {
        dtor(storage);
        std::free(storage);
      }
      storage = newStorage;
      allocated = sizeof(F);
    } else {
      if (storage) {
        dtor(storage);
      }
    }
    try {
      new (storage) F(std::forward<F>(f));
    } catch (...) {
      std::free(storage);
      throw;
    }
    dtor = [](void* ptr) { ((F*)ptr)->~F(); };
    call = [](void* ptr) { (*(F*)ptr)(); };

    return *this;
  }
  ~Function() {
    if (storage) {
      dtor(storage);
      std::free(storage);
    }
  }
  void operator()() {
    call(storage);
  }
};

template<typename Thread>
struct HandleT {
  Function* func = nullptr;
  Thread* thread = nullptr;
  HandleT() = default;
  HandleT(Function* func, Thread* thread)
      : func(func)
      , thread(thread) {
  }
  HandleT(HandleT&& n) {
    func = std::exchange(n.func, nullptr);
    thread = std::exchange(n.thread, nullptr);
  }
  HandleT(const HandleT&) = delete;
  HandleT& operator=(HandleT&& n) {
    std::swap(func, n.func);
    std::swap(thread, n.thread);
    return *this;
  }
  HandleT& operator=(const HandleT&) = delete;
  ~HandleT() {
    if (func) {
      Function* ftmp = thread->freelist;
      do {
        func->next = ftmp;
      } while (!thread->freelist.compare_exchange_weak(ftmp, func));
    }
  }
  void setPriority(int value) {
    func->priority = value;
  }
  explicit operator bool() const {
    return func;
  }
};

using Handle = HandleT<struct Thread>;

struct Thread {

  std::thread thread;
  std::mutex mut;
  std::condition_variable cv;
  std::atomic<Function*> queue = nullptr;
  std::atomic<Function*> freelist = nullptr;
  Function* internalqueue = nullptr;
  bool dead = false;
  std::atomic<bool> busy = true;

  // std::list<Thread>::iterator it;

  sem_t sem;

  Thread() {
    sem_init(&sem, 0, 0);
  }

  void threadEntry() {
    // std::unique_lock<std::mutex> l(mut);
    while (true) {
      //        busy = false;
      //        while (!queue) {
      //          if (dead) {
      //            return;
      //          }
      //          cv.wait(l);
      //        }
      //        while (!queue.load()) {
      //          std::this_thread::yield();
      //        }
      Function* f = queue;
      while (!f) {
        if (dead) {
          return;
        }
        sem_wait(&sem);
        f = queue;
      }
      while (!queue.compare_exchange_weak(f, f->next)) {
        f = queue;
      }
      busy = true;
      // l.unlock();
      if (internalqueue || queue) {
        do {
          while (f) {
            Function** insert = &internalqueue;
            Function* next = internalqueue;
            while (next && next->priority <= f->priority) {
              insert = &next->next;
              next = next->next;
            }
            f->next = next;
            *insert = f;

            do {
              f = queue;
            } while (f && !queue.compare_exchange_weak(f, f->next));
          }

          f = internalqueue;
          internalqueue = f->next;

          (*f)();

          do {
            f = queue;
          } while (f && !queue.compare_exchange_weak(f, f->next));
        } while (f || internalqueue);
      } else {
        (*f)();
      }

      // l.lock();
    }
  }

  void enqueue(Function* func) {
    Function* qtmp = queue;
    do {
      func->next = qtmp;
    } while (!queue.compare_exchange_weak(qtmp, func));
    sem_post(&sem);
    //      if (!busy) {
    //        std::unique_lock<std::mutex> l(mut, std::defer_lock);
    //        if (l.try_lock()) {
    //          if (!busy) {
    //            cv.notify_all();
    //          }
    //        }
    //      }
  }

  template <typename F> Handle getHandle(F&& f) {
    Function* func = freelist;
    while (func && !freelist.compare_exchange_weak(func, func->next))
      ;
    if (!func) {
      func = new Function();
    }
    Handle h(func, this);
    *func = std::forward<F>(f);
    return h;
  }
};

struct Threads {

  std::atomic_size_t nextThread = 0;
  std::deque<Thread> threads;

  size_t size() const {
    return threads.size();
  }

  Thread& getThread() {
    return threads[nextThread++ % threads.size()];
  }

  void enqueue(const Handle& h) {
    h.thread->enqueue(h.func);
  }

  Threads() = default;
  Threads(int nThreads) {
    start(nThreads);
  }

  void start(int nThreads) {
    for (int i = 0; i != nThreads; ++i) {
      threads.emplace_back();
      //auto it = std::prev(threads.end());
      Thread* t = &threads.back();
      // t->it = it;
      threads.back().thread = std::thread([t]() { t->threadEntry(); });
    }
  }

  ~Threads() {
    for (auto& v : threads) {
      std::lock_guard<std::mutex> l(v.mut);
      v.dead = true;
      v.cv.notify_all();
      sem_post(&v.sem);
    }
    for (auto& v : threads) {
      v.thread.join();
    }
  }
};

struct Task {

  std::mutex mut;
  std::condition_variable cv;

  std::atomic_int liveCount{0};

  Threads* threads = nullptr;
  Task() = default;
  Task(Threads& threads)
      : threads(&threads) {
  }
  ~Task() {
    wait();
  }
  Task& operator=(const Task& n) {
    if (liveCount || n.liveCount) {
      throw std::runtime_error("attempt to copy active Task object");
    }
    threads = n.threads;
    return *this;
  }

  template <typename F>
  Handle getHandle(Thread& thread, F&& f) {
    return thread.getHandle([f = std::forward<F>(f), this]() mutable {
      f();
      if (--liveCount == 0) {
        std::unique_lock l(mut);
        cv.notify_all();
      }
    });
  }

  void enqueue(const Handle& h) {
    ++liveCount;
    threads->enqueue(h);
  }

  void wait() {
    std::unique_lock l(mut);
    while (liveCount != 0) {
      cv.wait(l);
    }
  }
};

}
