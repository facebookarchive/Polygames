/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <random>
#include <sstream>

#include <core/game.h>
#include <core/state.h>

using namespace csp::vkms;
using namespace std;

template <size_t NTOTAL> float good_action_eval(core::State& s) {
  size_t game_count{0};
  float sum_score{0};
  size_t n_actions_cur_game = 0;
  while (game_count < NTOTAL) {
    s.reset();
    n_actions_cur_game = 0;
    while (!s.terminated()) {
      s.DoGoodAction();
      ++n_actions_cur_game;
    }
    sum_score += 0.5 * (1 + s.getReward(0));
    ++game_count;
  }
  const float avg_score = sum_score / float(game_count);
  return avg_score;
}  // good_action_eval

template <size_t W, size_t H, size_t N> std::string get_map_str() {
  ostringstream oss;
  oss << "Minesweeper<W=" << W << ", H=" << H << ", M=" << N << ">";
  return oss.str();
}  // get_map_str

template <size_t W, size_t H, size_t N, int SEED, size_t NTOTAL>
void benchmark_vkms() {
  using namespace std::chrono;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  auto state = Minesweeper::State<W, H, N>(SEED);
  float win_rate = good_action_eval<NTOTAL>(state);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  double duration_s = duration<double>(t2 - t1).count();
  auto map_str = get_map_str<W, H, N>();
  std::cout << map_str << ", win rate = " << win_rate << ", took " << duration_s
            << "s (" << static_cast<double>(NTOTAL) / duration_s << " games/s)"
            << std::endl;
}  // benchmark_vkms

int main(int, char**) {
  static constexpr int master_seed = 999;
  benchmark_vkms<4, 4, 4, master_seed, 100000>();
  benchmark_vkms<3, 1, 1, master_seed, 100000>();
  benchmark_vkms<5, 2, 3, master_seed, 100000>();
  benchmark_vkms<5, 5, 10, master_seed, 100000>();
  benchmark_vkms<10, 1, 5, master_seed, 100000>();
  benchmark_vkms<7, 3, 10, master_seed, 100000>();
  benchmark_vkms<5, 5, 15, master_seed, 100000>();
  benchmark_vkms<8, 8, 10, master_seed, 100000>();
  benchmark_vkms<9, 9, 10, master_seed, 100000>();
  benchmark_vkms<16, 16, 40, master_seed, 10000>();
  benchmark_vkms<30, 16, 99, master_seed, 10000>();
}
