
#include "threads.h"

namespace threads {

async::Threads threads;
std::once_flag flag;

void init(int nThreads) {

  std::call_once(flag, [nThreads](){

    threads.start(nThreads);

    async::Task task(threads);
    std::vector<async::Handle> handles;

    for (int i = 0; i != nThreads; ++i) {
      auto& thread = threads.getThread();
      auto h = task.getHandle(thread, [i]() {
        setCurrentThreadName("async " + std::to_string(i));
      });
      task.enqueue(h);
      handles.push_back(std::move(h));
    }

    task.wait();
  });

}

void setCurrentThreadName(const std::string& name) {
#ifdef __APPLE__
  pthread_setname_np(name.c_str());
#elif __linux__
  pthread_setname_np(pthread_self(), name.c_str());
#endif
}

}
