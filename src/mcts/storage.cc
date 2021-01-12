
#include "storage.h"

namespace mcts {

std::mutex freeStoragesMutex;
std::list<Storage*> freeStorages;

Node* Storage::newNode() {
  if (chunkIndex >= chunks.size()) {
    Node* newChunk = (Node*)std::aligned_alloc(128, sizeof(Node) * chunkSize);
    new (newChunk) Node[chunkSize];
    for (size_t i = 0; i != chunkSize; ++i) {
      newChunk[i].setStorageAndId(this, i);
    }
    chunks.push_back(newChunk);
  }
  Node* r = chunks[chunkIndex] + subIndex;
  ++subIndex;
  if (subIndex == chunkSize) {
    subIndex = 0;
    ++chunkIndex;
  }
  ++allocated;
  return r;
}

void Storage::freeNode(Node* node) {
  --allocated;
  if (allocated == 0) {
    chunkIndex = 0;
    subIndex = 0;
    std::lock_guard l(freeStoragesMutex);
    freeStorages.push_back(this);
  }
}

Storage* Storage::getStorage() {
  std::unique_lock l(freeStoragesMutex);
  if (freeStorages.empty()) {
    l.unlock();
    return new Storage();
  }
  Storage* r = freeStorages.back();
  freeStorages.pop_back();
  return r;
}

}  // namespace mcts
