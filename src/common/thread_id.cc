 
#include <atomic>

namespace {
std::atomic_int threadIdCounter{0};
thread_local int threadId = ++threadIdCounter;
}

namespace common {

int getThreadId() {
  return threadId;
}

}
