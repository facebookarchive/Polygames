/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../core/state.h"
#include "commons/hash.h"
#include <array>
#include <mutex>

namespace Othello2 {

template <size_t SIZE> class State : public core::State {

  static_assert(SIZE >= 4, "Board too small");
  static_assert(SIZE % 2 == 0, "Board has odd size");
  static constexpr size_t NUM_PIECE_TYPES = 2;
  static constexpr size_t NUM_FIELD_TYPES = NUM_PIECE_TYPES + 1;
  // using _Action = Action<SIZE>;
  static constexpr size_t HASHBOOK_SIZE = SIZE * SIZE * NUM_FIELD_TYPES + 1;
  static constexpr size_t HASH_BLACK_OFFSET = SIZE * SIZE;
  static constexpr size_t HASH_WHITE_OFFSET = 2 * SIZE * SIZE;
  static constexpr size_t WHITE_INIT_OFFSET_1 =
      SIZE * (SIZE / 2 - 1) + SIZE / 2 - 1;
  static constexpr size_t WHITE_INIT_OFFSET_2 = SIZE * SIZE / 2 + SIZE / 2;
  static constexpr size_t BLACK_INIT_OFFSET_1 =
      SIZE * (SIZE / 2 - 1) + SIZE / 2;
  static constexpr size_t BLACK_INIT_OFFSET_2 = SIZE * SIZE / 2 + SIZE / 2 - 1;
  using _HashBook = HashBook<uint64_t, HASHBOOK_SIZE>;
  using _Hasher = Hasher<uint64_t, HASHBOOK_SIZE>;
  using Cache = std::array<uint8_t, SIZE * SIZE>;

 public:
  using Field = uint8_t;
  using Board = std::array<Field, SIZE * SIZE>;

  static constexpr Field EMPTY = 0;
  static constexpr Field BLACK = 1;
  static constexpr Field WHITE = 2;
  static constexpr char EMPTY_STR[] = ".";
  static constexpr char BLACK_STR[] = "x";
  static constexpr char WHITE_STR[] = "o";

  State(int seed);
  virtual void Initialize() override;
  virtual std::unique_ptr<core::State> clone_() const override;
  virtual void ApplyAction(const ::_Action& action) override;
  virtual void DoGoodAction() override;
  virtual void printCurrentBoard() const override;

  const Board& GetBoard() const {
    return _board;
  }

 private:
  std::string boardToString() const;
  bool CanPutStone(Field stone) const;
  bool CanPutStone(Field stone, size_t row, size_t col) const;
  void PutStone(Field stone, size_t row, size_t col);
  bool boardFilled() const;
  constexpr Field stoneToPlay() const;
  void nextTurn();

  void RefillLegalActions();
  void fillFeatures();
  void initializeBoard();
  void initializeHasher();
  void initializeCache();
  void setTerminalStatus();

  static std::once_flag hashBookConfigured;
  static _HashBook hashBook;
  _Hasher _hasher;
  Board _board;
  Cache _cache;

};  // class State

template <size_t SIZE> std::once_flag State<SIZE>::hashBookConfigured;

template <size_t SIZE> typename State<SIZE>::_HashBook State<SIZE>::hashBook;

}  // namespace Othello2
