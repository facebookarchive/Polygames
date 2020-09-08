/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <sstream>

#include "replay_buffer.h"

#define ZSTD_STATIC_LINKING_ONLY
#include "zstd/lib/zstd.h"

namespace {
struct cctx {
  ZSTD_CCtx* ctx;
  cctx() {
    ctx = ZSTD_createCCtx();
    if (!ctx) {
      throw std::runtime_error("Failed to allocate zstd context");
    }
  }
  ~cctx() {
    ZSTD_freeCCtx(ctx);
  }
};

struct dctx {
  ZSTD_DCtx* ctx;
  dctx() {
    ctx = ZSTD_createDCtx();
    if (!ctx) {
      throw std::runtime_error("Failed to allocate zstd context");
    }
  }
  ~dctx() {
    ZSTD_freeDCtx(ctx);
  }
};
}  // namespace

namespace core {

void ReplayBuffer::add(
    std::unordered_map<std::string, at::Tensor> input) {
  if (input.empty()) {
    return;
  }
  if (!hasKeys) {
    std::lock_guard l(keyMutex);
    if (keys.empty()) {
      for (auto& v : input) {
        torch::ArrayRef<int64_t> x = v.second.sizes();
        x = torch::ArrayRef<int64_t>(x.begin() + 1, x.end());
        keys.push_back({v.first,
                        std::vector<int64_t>(x.begin(), x.end()),
                        v.second.dtype()});
      }
      hasKeys = true;

      for (auto& vx : input) {
        printf("  keys  key '%s' shape %s\n",
               vx.first.c_str(),
               ss(vx.second.sizes()).c_str());
      }
    }
  }

  std::vector<char> tmpbuf;

  auto n = input.begin()->second.size(0);

  cctx ctx;

  for (int i = 0; i != n; ++i) {
    if (input.size() != keys.size()) {
      for (auto& vx : input) {
        printf("  add key '%s' shape %s\n",
               vx.first.c_str(),
               ss(vx.second.sizes()).c_str());
      }
      throw std::runtime_error("replay buffer keys mismatch");
    }
    BufferEntry* newEntry = new BufferEntry[input.size()];

    size_t index = 0;
    for (const auto& [key, shape, dtype] : keys) {
      auto t = input.at(key)[i];

      if (!t.is_contiguous()) {
        throw std::runtime_error("replay buffer input is not contiguous");
      }

      void* data = t.data_ptr();
      size_t datasize = dtype.itemsize() * t.numel();

      tmpbuf.resize(sizeof(size_t) + ZSTD_compressBound(datasize));
      auto n = ZSTD_compressCCtx(
          ctx.ctx, tmpbuf.data(), tmpbuf.size(), data, datasize, 0);
      if (ZSTD_isError(n)) {
        throw std::runtime_error("replay buffer compress failed");
      }

      auto& e = newEntry[index++];
      e.datasize = datasize;
      e.data.assign(tmpbuf.begin(), tmpbuf.begin() + n);
    }
    //      size_t index = 0;
    //      for (auto& v : input) {
    //        if (v.first != keys.at(index)) {
    //          for (auto& vx : input) {
    //            printf("  add key '%s' shape %s\n", vx.first.c_str(),
    //            ss(vx.second.sizes()).c_str());
    //          }
    //          throw std::runtime_error("replay buffer key mismatch; got " +
    //          v.first + ", expected " + keys.at(index));
    //        }
    //        newEntry[index++] = v.second.select(0, i);
    //      }

    auto slot = numAdd_++ % capacity;
    auto* prev = buffer[slot].exchange(newEntry);
    if (prev) {
      delete[] prev;
    }
  }
}

std::unordered_map<std::string, at::Tensor> ReplayBuffer::sampleImpl(
    int sampleSize) {
  if (!hasKeys) {
    return {};
  }
  int siz = size();
  std::unordered_map<std::string, torch::Tensor> r;
  std::vector<char*> pointers;
  // size_t index = 0;
  for (auto& [k, shape, dtype] : keys) {
    // auto& t = buffer[0][index];
    std::vector<int64_t> sizes;
    sizes.assign(shape.begin(), shape.end());
    sizes.insert(sizes.begin(), sampleSize);
    // printf("keys %d is %s\n", index, ss(sizes).c_str());
    auto tensor = torch::empty(sizes, dtype);
    pointers.push_back((char*)tensor.data_ptr());
    r[k] = tensor;
    //++index;
  }
  dctx ctx;
  size_t nCopies = 0;
  auto copy = [&](size_t srcIndex) {
    auto src = buffer[srcIndex].exchange(nullptr);
    if (!src) {
      return 0;
    }
    if (nCopies >= (size_t)sampleSize) {
      throw std::runtime_error(
          "replay buffer internal error: copied too many samples");
    }
    ++nCopies;
    for (size_t i = 0; i != keys.size(); ++i) {
      //        if (src[i].sizes() != buffer[0][i].sizes()) {
      //          printf("src[%d] is %s\n", i, ss(src[i].sizes()).c_str());
      //          printf("buffer[0][%d] is %s\n", i,
      //          ss(buffer[0][i].sizes()).c_str()); throw
      //          std::runtime_error("replay buffer size mismatch");
      //        }
      size_t datasize = src[i].datasize;
      auto n = ZSTD_decompressDCtx(ctx.ctx,
                                   pointers[i],
                                   datasize,
                                   src[i].data.data(),
                                   src[i].data.size());
      if (ZSTD_isError(n)) {
        throw std::runtime_error("replay buffer decompress failed");
      }
      pointers[i] += datasize;
    }
    BufferEntry* nullref = nullptr;
    if (!buffer[srcIndex].compare_exchange_strong(nullref, src)) {
      delete[] src;
    }
    return 1;
  };
  int64_t seq = std::min(numAdd_ - prevSampleNumAdd_, (int64_t)sampleSize);
  int64_t i = 0;
  //    for (;i != seq;) {
  //      i += copy((prevSampleNumAdd_ + i) % siz);
  //    }
  prevSampleNumAdd_ += seq;
  std::vector<size_t> indices;
  while (i != sampleSize) {
    indices.clear();
    std::unique_lock l(sampleMutex);
    for (size_t ii = i; ii != sampleSize;) {
      if (sampleOrderIndex >= sampleOrder.size()) {
        size_t p = sampleOrder.size();
        if (p != capacity) {
          sampleOrder.resize(siz);
          for (size_t i = p; i != sampleOrder.size(); ++i) {
            sampleOrder[i] = i;
          }
        }
        std::shuffle(sampleOrder.begin(), sampleOrder.end(), rng_);
        sampleOrderIndex = 0;
      }
      indices.push_back(sampleOrder.at(sampleOrderIndex++));
      ++ii;
    }
    l.unlock();
    for (size_t index : indices) {
      i += copy(index);
    }
  }
  numSample_ += sampleSize;
  return r;
}

std::unordered_map<std::string, at::Tensor> ReplayBuffer::sample(int sampleSize) {
  // return sampleImpl(sampleSize);
  std::unique_lock l(mut);
  if (sampleThreads.empty()) {
    for (int i = 0; i != 8; ++i) {
      sampleThreads.emplace_back([this]() {
        std::unique_lock l(mut);
        while (true) {
          while (results.size() >= 8 || resultsSampleSize == 0) {
            cv.wait(l);
            if (sampleThreadDie) {
              return;
            }
          }
          l.unlock();
          auto tmp = sampleImpl(resultsSampleSize);
          l.lock();
          results.push_back(std::move(tmp));
          cv2.notify_all();
        }
      });
    }
  }
  resultsSampleSize = sampleSize;
  while (results.empty()) {
    cv.notify_all();
    cv2.wait(l);
  }
  auto r = std::move(results.front());
  results.pop_front();
  cv.notify_all();
  return r;
}

}
