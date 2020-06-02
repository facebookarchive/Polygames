/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <samegame.h>

/*
std::vector<int> _dataGroups;  // nbI*nbJ
std::vector<int> _groupSizes;  // <= nbI*nbJ
std::vector<int> _lastIs;      // nbJ
std::vector<int> _lastJs;      // nbI

bool isTerminated() const;
double getScore() const;
const std::vector<Move> & getMoves() const;
void play(int n);

void findGroups();
void buildGroup(int i0, int j0);
void findMoves();
void contractJ(int j);
void contractI(int i, int j);
*/

struct BoardTest : Samegame::Board {
 BoardTest(int nbI, int nbJ, int nbColors, std::function<int()> fillFunc) :
  Board(nbI, nbJ, nbColors, fillFunc) {}

 const std::vector<int> & testDataGroups() const { return _dataGroups; }
 const std::vector<int> & testGroupSizes() const { return _groupSizes; }
 const std::vector<int> & testLastIs() const { return _lastIs; }
 const std::vector<int> & testLastJs() const { return _lastJs; }

 int testInd(int i, int j) const { return ind(i, j); }

 void testFindGroups() { findGroups(); }
 void testBuildGroup(int i0, int j0) { buildGroup(i0, j0); }
 void testFindMoves() { findMoves(); }
 void testContractJ(int j) { contractJ(j); }
 void testContractI(int i, int j) { contractI(i, j); }
};

TEST(SamegameGroup, move1) {
 Samegame::Board::Move m{1, -2, 0, 3};
 ASSERT_EQ(1, m._i);
 ASSERT_EQ(-2, m._j);
 ASSERT_EQ(0, m._color);
 ASSERT_EQ(3, m._eval);
}

TEST(SamegameGroup, ind1) {
 std::vector<int> colors {
  0, 0, 3,
   2, 1, 1
 };
 auto iter = colors.begin();
 auto fillFunc = [&iter]() { return *iter++; };
 BoardTest game(2, 3, 4, fillFunc);
 ASSERT_EQ(0, game.testInd(0, 0));
 ASSERT_EQ(1, game.testInd(0, 1));
 ASSERT_EQ(2, game.testInd(0, 2));
 ASSERT_EQ(3, game.testInd(1, 0));
 ASSERT_EQ(4, game.testInd(1, 1));
 ASSERT_EQ(5, game.testInd(1, 2));
}

TEST(SamegameGroup, dataColors1) {
 std::vector<int> colors {
  0, 0, 3,
   2, 1, 1
 };
 auto iter = colors.begin();
 auto fillFunc = [&iter]() { return *iter++; };
 BoardTest game(2, 3, 4, fillFunc);
 ASSERT_EQ(0, game.dataColors(0, 0));
 ASSERT_EQ(0, game.dataColors(0, 1));
 ASSERT_EQ(3, game.dataColors(0, 2));
 ASSERT_EQ(2, game.dataColors(1, 0));
 ASSERT_EQ(1, game.dataColors(1, 1));
 ASSERT_EQ(1, game.dataColors(1, 2));
}

TEST(SamegameGroup, lastIs1) {
 std::vector<int> colors {
  0, 0, 3,
   2, 1, 1
 };
 auto iter = colors.begin();
 auto fillFunc = [&iter]() { return *iter++; };
 BoardTest game(2, 3, 4, fillFunc);
 ASSERT_EQ(1, game.testLastIs()[0]);
 ASSERT_EQ(1, game.testLastIs()[1]);
 ASSERT_EQ(1, game.testLastIs()[2]);
}

TEST(SamegameGroup, lastJs1) {
 std::vector<int> colors {
  0, 0, 3,
   2, 1, 1
 };
 auto iter = colors.begin();
 auto fillFunc = [&iter]() { return *iter++; };
 BoardTest game(2, 3, 4, fillFunc);
 ASSERT_EQ(2, game.testLastJs()[0]);
 ASSERT_EQ(2, game.testLastJs()[1]);
}

TEST(SamegameGroup, dataGroups1) {
 std::vector<int> colors {
  0, 0, 3,
   2, 1, 1
 };
 auto iter = colors.begin();
 auto fillFunc = [&iter]() { return *iter++; };
 BoardTest game(2, 3, 4, fillFunc);

 ASSERT_EQ(4, game.testGroupSizes().size());
 ASSERT_EQ(2, game.testGroupSizes()[0]);
 ASSERT_EQ(1, game.testGroupSizes()[1]);
 ASSERT_EQ(1, game.testGroupSizes()[2]);
 ASSERT_EQ(2, game.testGroupSizes()[3]);

 ASSERT_EQ(0, game.testDataGroups()[game.testInd(0, 0)]);
 ASSERT_EQ(0, game.testDataGroups()[game.testInd(0, 1)]);
 ASSERT_EQ(1, game.testDataGroups()[game.testInd(0, 2)]);
 ASSERT_EQ(2, game.testDataGroups()[game.testInd(1, 0)]);
 ASSERT_EQ(3, game.testDataGroups()[game.testInd(1, 1)]);
 ASSERT_EQ(3, game.testDataGroups()[game.testInd(1, 2)]);
}

