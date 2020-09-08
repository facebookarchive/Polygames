
#include "async.h"

#include <string>

namespace threads {

extern async::Threads threads;

void init(int nThreads);
void setCurrentThreadName(const std::string& name);

}
