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

using tube::old::ReplayBuffer;

std::vector<int64_t> sampleKfromN(int k, int n, std::mt19937& rng) {
  std::unordered_set<int> samples;
  while ((int)samples.size() < k && (int)samples.size() < n) {
    int s = rng() % n;
    samples.insert(s);
  }
  std::vector<int64_t> ret;
  for (int s : samples) {
    ret.push_back((int64_t)s);
  }
  return ret;
}

void ReplayBuffer::add(std::unordered_map<std::string, torch::Tensor> input) {
  std::unique_lock<std::mutex> lk(mBuf_);
  if (size_ == 0) {
    for (auto& it : input) {
      auto t = it.second.sizes();
      std::vector<int64_t> sizes(t.begin(), t.end());
      sizes[0] = capacity;
      buffer_.insert({it.first, torch::zeros(sizes, it.second.dtype())});
    }
  }
  assert(input.size() == buffer_.size());

  // now perform the copying
  torch::Tensor tensorIndices = getNextIndices(input);
  numAdd_ += tensorIndices.size(0);
  for (auto& b : buffer_) {
    const std::string name = b.first;
    auto in = input.find(name);
    assert(in != input.end());
    // Explicit size checking
    auto s1 = b.second.sizes();
    auto s2 = in->second.sizes();
    assert(s1.size() == s2.size());
    for (size_t i = 1; i < s1.size(); i++) {
      assert(s1[i] == s2[i]);
    }
    b.second.index_copy_(0, tensorIndices, in->second);
  }
}

std::unordered_map<std::string, torch::Tensor> ReplayBuffer::sample(
    int sampleSize) {
  std::unique_lock<std::mutex> lk(mBuf_);
  numSample_ += sampleSize;
  assert(sampleSize <= size_);
  auto sampleIndices = torch::tensor(sampleKfromN(sampleSize, size_, rng_));

  std::unordered_map<std::string, torch::Tensor> result;
  for (auto& b : buffer_) {
    const std::string& name = b.first;
    result.insert({name, torch::index_select(b.second, 0, sampleIndices)});
  }
  return result;
}

ReplayBuffer::SerializableState ReplayBuffer::toState() {
  std::unique_lock<std::mutex> lk(mBuf_);
  ReplayBuffer::SerializableState state;
  state.capacity = capacity;
  state.size = size_;
  state.nextIdx = nextIdx_;
  std::ostringstream oss;
  oss << rng_;
  state.rngState = oss.str();
  state.buffer = buffer_;
  return state;
}

void ReplayBuffer::initFromState(const SerializableState& state) {
  std::unique_lock<std::mutex> lk(mBuf_);
  if (state.capacity != capacity) {
    std::ostringstream oss;
    oss << "Attempt to initialize a buffer of capacity " << capacity
        << " from buffer state of capacity " << state.capacity;
    throw std::runtime_error(oss.str());
  }
  size_ = state.size;
  nextIdx_ = state.nextIdx;
  std::istringstream iss(state.rngState);
  iss >> rng_;
  buffer_ = state.buffer;
}

torch::Tensor ReplayBuffer::getNextIndices(
    std::unordered_map<std::string, torch::Tensor>& input) {
  int inSize = -1;
  // these are indices of replay buffer that we will copy into
  std::vector<int64_t> copyIndices;
  for (auto& it : input) {
    if (inSize < 0) {
      inSize = it.second.size(0);
      if (inSize > capacity) {
        std::cerr << "inSize=" << inSize << ", capacity=" << capacity
                  << std::endl;
        assert(inSize <= capacity);
      }
      for (int i = 0; i < inSize; i++) {
        copyIndices.push_back((i + nextIdx_) % capacity);
      }
      if (size_ < capacity) {
        size_ = size_ + inSize > capacity ? capacity : size_ + inSize;
      }
      nextIdx_ = (nextIdx_ + inSize) % capacity;
    } else {
      // all the names should have the same input size
      assert(inSize == it.second.size(0));
    }
  }
  return torch::tensor(copyIndices);
}

// inline bool test() {

//  ReplayBuffer buffer(81, 42);

//  auto start = std::chrono::steady_clock::now();

//  auto rep = [&](std::string s, int n) {
//    double t = std::chrono::duration_cast<
//                   std::chrono::duration<double, std::ratio<1, 1>>>(
//                   std::chrono::steady_clock::now() - start)
//                   .count();
//    printf("%s in %gs, %g/s\n", s.c_str(), t, n / t);
//    start = std::chrono::steady_clock::now();
//  };

//  std::vector<torch::Tensor> data;
//  for (int i = 0; i != 1024 * 1; ++i) {
//    data.push_back(torch::rand({40, 8, 8, 100}));
//  }

//  rep("gen data", data.size());

//  srand(42);

//  buffer.add({{"x", data[rand() % data.size()]}});

//  rep("initial add", 1);

//  int n = 600000;
//  for (int i = 0; i != n; ++i) {
//    buffer.add({{"x", data[rand() % data.size()]}});
//  }

//  rep("add", n);

//  for (int i = 0; i != 10; ++i) buffer.sample(100000);

//  rep("sample", 100000);

//  printf("\n---\n\n");

//  srand(42);

//  tube::ReplayBuffer2 buffer2(81, 42);
//  rep("init", 1);
//  buffer2.add({{"x", data[rand() % data.size()]}});

//  rep("initial add", 1);

//  for (int i = 0; i != n; ++i) {
//    buffer2.add({{"x", data[rand() % data.size()]}});
//  }

//  rep("add", n);

//  for (int i = 0; i != 10; ++i) buffer2.sample(100000);
////  for (auto& [k, v] : tmp) {
////    std::cout << " sample returned " << k << " = " << v.sizes();
////  }

//  rep("sample", 100000);

//  std::terminate();

//}

// inline bool vtest = test();

void tube::ReplayBuffer2::add(
    std::unordered_map<std::string, at::Tensor> input) {
  // std::lock_guard l(sampleMutex);
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
        throw std::runtime_error("replay buffer input is not contigous");
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

std::unordered_map<std::string, at::Tensor> tube::ReplayBuffer2::sampleImpl(
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
    if (!src)
      return 0;
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
