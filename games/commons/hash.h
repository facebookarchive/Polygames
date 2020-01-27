/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

template<
  typename T,
  size_t SIZE>
class HashBook {
public:
  using Storage  = std::array<T, SIZE>;

  HashBook() = default;

  template<typename RngEngine>
  void setup(RngEngine& rng) {
    std::independent_bits_engine<RngEngine, sizeof(T) * 8, T> gen(rng);
    for (size_t i = 0; i < _book.size(); ++i) {
      _book[i] = gen();
    }
  }

  constexpr T operator[](size_t i) const { return _book[i]; }

private:
  HashBook(const HashBook&) = delete;
  HashBook& operator=(const HashBook&) = delete;
  Storage _book;
};

template<typename T, size_t SIZE>
class Hasher {
public:

  Hasher(const HashBook<T, SIZE>& hashBook) : _hashBook(hashBook), _hash(0) {}

  void reset() { _hash = 0; }

  void trigger(size_t i) { _hash ^= _hashBook[i]; }

  uint64_t hash() const { return _hash; }

private:
  const HashBook<T, SIZE>& _hashBook;
  uint64_t _hash;
};
