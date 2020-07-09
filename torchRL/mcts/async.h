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

struct AsyncThreads {

  struct Thread {
    std::thread thread;

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

    struct Handle {
      Function* func = nullptr;
      Thread* thread = nullptr;
      Handle() = default;
      Handle(Function* func, Thread* thread)
          : func(func)
          , thread(thread) {
      }
      Handle(Handle&& n) {
        func = std::exchange(n.func, nullptr);
        thread = std::exchange(n.thread, nullptr);
      }
      Handle(const Handle&) = delete;
      Handle& operator=(Handle&& n) {
        std::swap(func, n.func);
        std::swap(thread, n.thread);
        return *this;
      }
      Handle& operator=(const Handle&) = delete;
      ~Handle() {
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
    };

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

  std::atomic_size_t nextThread = 0;
  std::deque<Thread> threads;

  Thread& getThread() {
    return threads[nextThread++ % threads.size()];
  }

  void enqueue(const Thread::Handle& h) {
    h.thread->enqueue(h.func);
  }

  AsyncThreads(int nThreads) {
    for (int i = 0; i != nThreads; ++i) {
      threads.emplace_back();
      auto it = std::prev(threads.end());
      Thread* t = &threads.back();
      // t->it = it;
      threads.back().thread = std::thread([t]() { t->threadEntry(); });
    }
  }

  ~AsyncThreads() {
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

struct AsyncTask {

  std::mutex mut;
  std::condition_variable cv;

  std::atomic_int liveCount{0};

  AsyncThreads* threads = nullptr;
  AsyncTask() = default;
  AsyncTask(AsyncThreads& threads)
      : threads(&threads) {
  }
  ~AsyncTask() {
    wait();
  }

  template <typename F>
  AsyncThreads::Thread::Handle getHandle(AsyncThreads::Thread& thread, F&& f) {
    return thread.getHandle([f = std::forward<F>(f), this]() mutable {
      f();
      if (--liveCount == 0) {
        std::unique_lock l(mut);
        cv.notify_all();
      }
    });
  }

  void enqueue(const AsyncThreads::Thread::Handle& h) {
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
