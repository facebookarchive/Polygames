
#include "threads.h"

namespace threads {

async::Threads threads;
std::once_flag flag;

void init(int nThreads) {

  std::call_once(
      flag,
      [](int nThreads) {
        if (nThreads <= 0) {
          nThreads = std::thread::hardware_concurrency();
          if (nThreads <= 0) {
            throw std::runtime_error("Could not automatically determine the "
                                     "number of hardware threads :(");
          }
          printf("Starting %d threads (automatically configured)\n", nThreads);
        } else {
          printf("Starting %d threads\n", nThreads);
        }

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
      },
      nThreads);
}

void setCurrentThreadName(const std::string& name) {
#ifdef __APPLE__
  pthread_setname_np(name.c_str());
#elif __linux__
  pthread_setname_np(pthread_self(), name.c_str());
#endif
}

}  // namespace threads
