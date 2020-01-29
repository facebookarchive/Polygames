/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit tests for the Havannah game.

#include <havannah.h>
#include <gtest/gtest.h>
#include "utils.h"

///////////////////////////////////////////////////////////////////////////////
// helpers
///////////////////////////////////////////////////////////////////////////////

static void CheckHavannahPathInfo(const Havannah::PathInfo & path, Havannah::Color color,
  unsigned borders, unsigned corners, int m) {
 ASSERT_EQ(path._color, color);
 ASSERT_EQ(path._borders, borders);
 ASSERT_EQ(path._corners, corners);
 ASSERT_EQ(path._mainPathIndex, m);
}

namespace Havannah {

 template<int SIZE, bool PIE>
  class BoardTest : public Board<SIZE, PIE> {

   public:

    BoardTest() : Board<SIZE, PIE>() {}

    int getNbIndices() const {
        return Board<SIZE, PIE>::_nbIndices;
    }

    int getNbEmptyIndices() const {
        return Board<SIZE, PIE>::_nbEmptyIndices;
    }

    void getPathIndexAndColorAtIndex(int index, int& pathIndex, Color& color) const {
     return Board<SIZE, PIE>::getPathIndexAndColorAtIndex(index, pathIndex, color);
    }

    const std::array<std::array<int,7>,fullsize(SIZE)*fullsize(SIZE)> & getNeighboursBoard() const {
     return Board<SIZE, PIE>::_neighboursBoard;
    }

    std::array<PathInfo, fullsize(SIZE)*fullsize(SIZE)> & getPaths() {
     return Board<SIZE, PIE>::_paths;
    }

    int & getPathsEnd() {
     return Board<SIZE, PIE>::_pathsEnd;
    }

    std::array<int, fullsize(SIZE)*fullsize(SIZE)> & getPathBoard() {
     return Board<SIZE, PIE>::_pathBoard;
    }

    unsigned computeBorders(int index) const { return Board<SIZE, PIE>::computeBorders(index); }
    unsigned computeCorners(int index) const { return Board<SIZE, PIE>::computeCorners(index); }

  };

}  // namespace Havannah

///////////////////////////////////////////////////////////////////////////////
// unit tests
///////////////////////////////////////////////////////////////////////////////

TEST(HavannahGroup, fullsize) {
 ASSERT_EQ(Havannah::fullsize(5), 9);
 ASSERT_EQ(Havannah::fullsize(6), 11);
 ASSERT_EQ(Havannah::fullsize(7), 13);
 ASSERT_EQ(Havannah::fullsize(8), 15);
 ASSERT_EQ(Havannah::fullsize(9), 17);
 ASSERT_EQ(Havannah::fullsize(10), 19);
}

TEST(HavannahGroup, reset_8) {
 Havannah::BoardTest<8, true> b;
 b.reset();
 ASSERT_EQ(b.getNbIndices(), 15*15-7*8);
 ASSERT_EQ(b.getCurrentPlayer(), PLAYER_0);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_NULL);
 ASSERT_EQ(b.getLastIndex(), std::optional<int>());
 ASSERT_EQ(b.getNbEmptyIndices(), 15*15-7*8);
}

TEST(HavannahGroup, reset_7) {
 Havannah::BoardTest<7, true> b;
 b.reset();
 ASSERT_EQ(b.getNbIndices(), 13*13-6*7);
 ASSERT_EQ(b.getCurrentPlayer(), PLAYER_0);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_NULL);
 ASSERT_EQ(b.getLastIndex(), std::optional<int>());
 ASSERT_EQ(b.getNbEmptyIndices(), 13*13-6*7);
}

TEST(HavannahGroup, reset_5) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 ASSERT_EQ(b.getNbIndices(), 9*9-4*5);
 ASSERT_EQ(b.getCurrentPlayer(), PLAYER_0);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_NULL);
 ASSERT_EQ(b.getLastIndex(), std::optional<int>());
 ASSERT_EQ(b.getNbEmptyIndices(), 9*9-4*5);
}

TEST(HavannahGroup, copyConstructor) {
 Havannah::BoardTest<5, true> b0;
 b0.reset();
 Havannah::BoardTest<5, true> b(b0);
 ASSERT_EQ(b.getNbIndices(), 61);
 ASSERT_EQ(b.getCurrentPlayer(), PLAYER_0);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_NULL);
 ASSERT_EQ(b.getLastIndex(), std::optional<int>());
 ASSERT_EQ(b.getNbEmptyIndices(), 61);
}

TEST(HavannahGroup, resetNeighboursTopLeft) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index = b.convertCellToIndex(Havannah::Cell(0, 4));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Havannah::Cell(0,5)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Havannah::Cell(1,3)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Havannah::Cell(1,4)));
 ASSERT_EQ(refNeighbourIndices[3], -1);
}

TEST(HavannahGroup, resetNeighboursTopCenter) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index = b.convertCellToIndex(Havannah::Cell(0, 6));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Havannah::Cell(0,5)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Havannah::Cell(0,7)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Havannah::Cell(1,5)));
 ASSERT_EQ(refNeighbourIndices[3], b.convertCellToIndex(Havannah::Cell(1,6)));
 ASSERT_EQ(refNeighbourIndices[4], -1);
}

TEST(HavannahGroup, resetNeighboursTopRight) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index = b.convertCellToIndex(Havannah::Cell(0, 8));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Havannah::Cell(0,7)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Havannah::Cell(1,7)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Havannah::Cell(1,8)));
 ASSERT_EQ(refNeighbourIndices[3], -1);
}

TEST(HavannahGroup, resetNeighboursMiddleRight) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index = b.convertCellToIndex(Havannah::Cell(4, 8));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Havannah::Cell(3,8)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Havannah::Cell(4,7)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Havannah::Cell(5,7)));
 ASSERT_EQ(refNeighbourIndices[3], -1);
}

TEST(HavannahGroup, resetNeighboursMiddleRight2) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index = b.convertCellToIndex(Havannah::Cell(6, 6));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Havannah::Cell(5,6)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Havannah::Cell(5,7)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Havannah::Cell(6,5)));
 ASSERT_EQ(refNeighbourIndices[3], b.convertCellToIndex(Havannah::Cell(7,5)));
 ASSERT_EQ(refNeighbourIndices[4], -1);
}

TEST(HavannahGroup, resetNeighboursBottomRight) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index = b.convertCellToIndex(Havannah::Cell(8, 4));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Havannah::Cell(7,4)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Havannah::Cell(7,5)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Havannah::Cell(8,3)));
 ASSERT_EQ(refNeighbourIndices[3], -1);
}

TEST(HavannahGroup, resetNeighboursBottomCenter) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index = b.convertCellToIndex(Havannah::Cell(8, 2));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Havannah::Cell(7,2)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Havannah::Cell(7,3)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Havannah::Cell(8,1)));
 ASSERT_EQ(refNeighbourIndices[3], b.convertCellToIndex(Havannah::Cell(8,3)));
 ASSERT_EQ(refNeighbourIndices[4], -1);
}

TEST(HavannahGroup, resetNeighboursBottomLeft) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index = b.convertCellToIndex(Havannah::Cell(8, 0));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Havannah::Cell(7,0)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Havannah::Cell(7,1)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Havannah::Cell(8,1)));
 ASSERT_EQ(refNeighbourIndices[3], -1);
}

TEST(HavannahGroup, resetNeighboursMiddleLeft) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index = b.convertCellToIndex(Havannah::Cell(4, 0));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Havannah::Cell(3,1)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Havannah::Cell(4,1)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Havannah::Cell(5,0)));
 ASSERT_EQ(refNeighbourIndices[3], -1);
}

TEST(HavannahGroup, resetNeighboursMiddleCenter) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index = b.convertCellToIndex(Havannah::Cell(4, 4));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Havannah::Cell(3,4)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Havannah::Cell(3,5)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Havannah::Cell(4,3)));
 ASSERT_EQ(refNeighbourIndices[3], b.convertCellToIndex(Havannah::Cell(4,5)));
 ASSERT_EQ(refNeighbourIndices[4], b.convertCellToIndex(Havannah::Cell(5,3)));
 ASSERT_EQ(refNeighbourIndices[5], b.convertCellToIndex(Havannah::Cell(5,4)));
 ASSERT_EQ(refNeighbourIndices[6], -1);
}

TEST(HavannahGroup, resetPaths) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 ASSERT_EQ(b.getPathsEnd(), 1);
 const Havannah::PathInfo & p0 = b.getPaths()[0];
 ASSERT_EQ(p0._color, Havannah::Color::COLOR_NONE);
 ASSERT_EQ(p0._borders, 0);
 ASSERT_EQ(p0._corners, 0);
 ASSERT_EQ(p0._mainPathIndex, 0);
}

TEST(HavannahGroup, resetPathBoard) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 for (int i=0; i<61; i++) {
  ASSERT_EQ(b.getPathBoard()[i], 0); 
  ASSERT_EQ(b.getColorAtIndex(i), Havannah::Color::COLOR_NONE);
 }
}

TEST(HavannahGroup, resetGetters1) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 ASSERT_EQ(b.getNbIndices(), 61);
 ASSERT_EQ(b.getNbEmptyIndices(), 61);
 ASSERT_EQ(b.getLastIndex(), std::optional<int>());
 ASSERT_EQ(b.getCurrentPlayer(), PLAYER_0);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_NULL);
}

TEST(HavannahGroup, isValid1) {
 Havannah::BoardTest<5, true> b;
 b.reset();

 ASSERT_EQ(b.isValidIndex(12), true);
 ASSERT_EQ(b.isValidIndex(4), true);
 ASSERT_EQ(b.isValidIndex(0), false);
 ASSERT_EQ(b.isValidIndex(-1), false);
 ASSERT_EQ(b.isValidIndex(60), true);
 ASSERT_EQ(b.isValidIndex(61), false);

 ASSERT_EQ(b.isValidCell(Havannah::Cell(2, 3)), true);
 ASSERT_EQ(b.isValidCell(Havannah::Cell(0, 4)), true);
 ASSERT_EQ(b.isValidCell(Havannah::Cell(0, 3)), false);
 ASSERT_EQ(b.isValidCell(Havannah::Cell(-1, 1)), false);
 ASSERT_EQ(b.isValidCell(Havannah::Cell(2, 0)), false);
 ASSERT_EQ(b.isValidCell(Havannah::Cell(2, 5)), true);
 ASSERT_EQ(b.isValidCell(Havannah::Cell(2, -1)), false);
 ASSERT_EQ(b.isValidCell(Havannah::Cell(3, 9)), false);
 ASSERT_EQ(b.isValidCell(Havannah::Cell(3, 8)), true);
 ASSERT_EQ(b.isValidCell(Havannah::Cell(7, 5)), true);
 ASSERT_EQ(b.isValidCell(Havannah::Cell(7, 6)), false);
}

TEST(HavannahGroup, convertIndexToCell) {
 Havannah::BoardTest<8, true> b;
 b.reset();
 ASSERT_EQ(b.convertIndexToCell(3), Havannah::Cell(0, 3));
 ASSERT_EQ(b.convertIndexToCell(19), Havannah::Cell(1, 4));
 ASSERT_EQ(b.convertIndexToCell(61), Havannah::Cell(4, 1));
}

TEST(HavannahGroup, convertCellToIndex) {
 Havannah::BoardTest<10, true> b;
 b.reset();
 ASSERT_EQ(b.convertCellToIndex(Havannah::Cell(0, 2)), 2);
 ASSERT_EQ(b.convertCellToIndex(Havannah::Cell(1, 0)), 19);
 ASSERT_EQ(b.convertCellToIndex(Havannah::Cell(7, 5)), 138);
}

TEST(HavannahGroup, computeBorderCorner_center) {
 Havannah::BoardTest<8, true> b;
 b.reset();
 int index;
 index = b.convertCellToIndex(Havannah::Cell(7, 7));
 ASSERT_EQ(b.computeBorders(index), 0);
 ASSERT_EQ(b.computeCorners(index), 0);
}

TEST(HavannahGroup, computeBorderCorner_topleft) {
 Havannah::BoardTest<8, true> b;
 b.reset();
 int index;
 index = b.convertCellToIndex(Havannah::Cell(0, 7));
 ASSERT_EQ(b.computeBorders(index), 0);
 ASSERT_EQ(b.computeCorners(index), 1);
 index = b.convertCellToIndex(Havannah::Cell(0, 8));
 ASSERT_EQ(b.computeBorders(index), 1);
 ASSERT_EQ(b.computeCorners(index), 0);
 index = b.convertCellToIndex(Havannah::Cell(1, 6));
 ASSERT_EQ(b.computeBorders(index), 32);
 ASSERT_EQ(b.computeCorners(index), 0);
}

TEST(HavannahGroup, computeBorderCorner_topright) {
 Havannah::BoardTest<8, true> b;
 b.reset();
 int index;
 index = b.convertCellToIndex(Havannah::Cell(0, 14));
 ASSERT_EQ(b.computeBorders(index), 0);
 ASSERT_EQ(b.computeCorners(index), 2);
 index = b.convertCellToIndex(Havannah::Cell(0, 13));
 ASSERT_EQ(b.computeBorders(index), 1);
 ASSERT_EQ(b.computeCorners(index), 0);
 index = b.convertCellToIndex(Havannah::Cell(1, 14));
 ASSERT_EQ(b.computeBorders(index), 2);
 ASSERT_EQ(b.computeCorners(index), 0);
}

TEST(HavannahGroup, computeBorderCorner_middleright) {
 Havannah::BoardTest<8, true> b;
 b.reset();
 int index;
 index = b.convertCellToIndex(Havannah::Cell(7, 14));
 ASSERT_EQ(b.computeBorders(index), 0);
 ASSERT_EQ(b.computeCorners(index), 4);
 index = b.convertCellToIndex(Havannah::Cell(6, 14));
 ASSERT_EQ(b.computeBorders(index), 2);
 ASSERT_EQ(b.computeCorners(index), 0);
 index = b.convertCellToIndex(Havannah::Cell(8, 13));
 ASSERT_EQ(b.computeBorders(index), 4);
 ASSERT_EQ(b.computeCorners(index), 0);
}

TEST(HavannahGroup, computeBorderCorner_bottomright) {
 Havannah::BoardTest<8, true> b;
 b.reset();
 int index;
 index = b.convertCellToIndex(Havannah::Cell(14, 7));
 ASSERT_EQ(b.computeBorders(index), 0);
 ASSERT_EQ(b.computeCorners(index), 8);
 index = b.convertCellToIndex(Havannah::Cell(13, 8));
 ASSERT_EQ(b.computeBorders(index), 4);
 ASSERT_EQ(b.computeCorners(index), 0);
 index = b.convertCellToIndex(Havannah::Cell(14, 6));
 ASSERT_EQ(b.computeBorders(index), 8);
 ASSERT_EQ(b.computeCorners(index), 0);
}

TEST(HavannahGroup, computeBorderCorner_bottomleft) {
 Havannah::BoardTest<8, true> b;
 b.reset();
 int index;
 index = b.convertCellToIndex(Havannah::Cell(14, 0));
 ASSERT_EQ(b.computeBorders(index), 0);
 ASSERT_EQ(b.computeCorners(index), 16);
 index = b.convertCellToIndex(Havannah::Cell(13, 0));
 ASSERT_EQ(b.computeBorders(index), 16);
 ASSERT_EQ(b.computeCorners(index), 0);
 index = b.convertCellToIndex(Havannah::Cell(14, 1));
 ASSERT_EQ(b.computeBorders(index), 8);
 ASSERT_EQ(b.computeCorners(index), 0);
}

TEST(HavannahGroup, computeBorderCorner_middleleft) {
 Havannah::BoardTest<8, true> b;
 b.reset();
 int index;
 index = b.convertCellToIndex(Havannah::Cell(7, 0));
 ASSERT_EQ(b.computeBorders(index), 0);
 ASSERT_EQ(b.computeCorners(index), 32);
 index = b.convertCellToIndex(Havannah::Cell(6, 1));
 ASSERT_EQ(b.computeBorders(index), 32);
 ASSERT_EQ(b.computeCorners(index), 0);
 index = b.convertCellToIndex(Havannah::Cell(8, 0));
 ASSERT_EQ(b.computeBorders(index), 16);
 ASSERT_EQ(b.computeCorners(index), 0);
}


TEST(HavannahGroup, getPathIndexAndPlayerAtIndex) {
 Havannah::BoardTest<7, true> b;
 b.reset();
 b.getPathsEnd() = 3;
 b.getPaths()[1] = Havannah::PathInfo(1, Havannah::Color::COLOR_BLACK, 0, 0);
 b.getPaths()[2] = Havannah::PathInfo(2, Havannah::Color::COLOR_WHITE, 0, 0);

 int index_1_1 = b.convertCellToIndex(Havannah::Cell(1, 1));
 int index_3_1 = b.convertCellToIndex(Havannah::Cell(3, 1));
 int index_2_3 = b.convertCellToIndex(Havannah::Cell(2, 3));

 b.getPathBoard()[index_3_1] = 1;
 b.getPathBoard()[index_2_3] = 2;

 int pathIndex;
 Havannah::Color color;

 b.getPathIndexAndColorAtIndex(index_1_1, pathIndex, color);
 ASSERT_EQ(pathIndex, 0);
 ASSERT_EQ(color, Havannah::Color::COLOR_NONE);

 b.getPathIndexAndColorAtIndex(index_3_1, pathIndex, color);
 ASSERT_EQ(pathIndex, 1);
 ASSERT_EQ(color, Havannah::Color::COLOR_BLACK);

 b.getPathIndexAndColorAtIndex(index_2_3, pathIndex, color);
 ASSERT_EQ(pathIndex, 2);
 ASSERT_EQ(color, Havannah::Color::COLOR_WHITE);
}

TEST(HavannahGroup, play1) {
 Havannah::BoardTest<8, true> b;
 b.reset();
 int index;

 index = b.convertCellToIndex(Havannah::Cell(2, 8));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 2);
 ASSERT_EQ(b.getPathBoard()[index], 1);
 ASSERT_EQ(b.getPaths()[1]._color, Havannah::Color::COLOR_BLACK);
 ASSERT_EQ(b.getPaths()[1]._borders, 0);
 ASSERT_EQ(b.getPaths()[1]._corners, 0);
 ASSERT_EQ(b.getPaths()[1]._mainPathIndex, 1);

 index = b.convertCellToIndex(Havannah::Cell(3, 8));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 3);
 ASSERT_EQ(b.getPathBoard()[index], 2);
 ASSERT_EQ(b.getPaths()[2]._color, Havannah::Color::COLOR_WHITE);
 ASSERT_EQ(b.getPaths()[2]._borders, 0);
 ASSERT_EQ(b.getPaths()[2]._corners, 0);
 ASSERT_EQ(b.getPaths()[2]._mainPathIndex, 2);

 index = b.convertCellToIndex(Havannah::Cell(2, 9));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 3);
 ASSERT_EQ(b.getPathBoard()[index], 1);
 ASSERT_EQ(b.getPaths()[1]._color, Havannah::Color::COLOR_BLACK);
 ASSERT_EQ(b.getPaths()[1]._borders, 0);
 ASSERT_EQ(b.getPaths()[1]._corners, 0);
 ASSERT_EQ(b.getPaths()[1]._mainPathIndex, 1);
}

TEST(HavannahGroup, play2) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index;

 index = b.convertCellToIndex(Havannah::Cell(2, 5));
 b.play(index);
 index = b.convertCellToIndex(Havannah::Cell(3, 5));
 b.play(index);
 index = b.convertCellToIndex(Havannah::Cell(2, 6));
 b.play(index);

 ASSERT_EQ(b.getPathsEnd(), 3);
 ASSERT_EQ(b.getPathBoard()[index], 1);
 ASSERT_EQ(b.getPaths()[1]._color, Havannah::Color::COLOR_BLACK);
 ASSERT_EQ(b.getPaths()[1]._borders, 0);
 ASSERT_EQ(b.getPaths()[1]._corners, 0);
 ASSERT_EQ(b.getPaths()[1]._mainPathIndex, 1);
}

TEST(HavannahGroup, play3) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 int index;

 ASSERT_EQ(b.getNbIndices(), 61);

 ASSERT_EQ(b.getNbEmptyIndices(), 61);
 ASSERT_EQ(b.getLastIndex(), std::optional<int>());
 ASSERT_EQ(b.getCurrentPlayer(), PLAYER_0);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_NULL);

 index = b.convertCellToIndex(Havannah::Cell(2, 6));
 b.play(index);
 ASSERT_EQ(b.getNbEmptyIndices(), 60);
 ASSERT_EQ(b.getLastIndex(), index);
 ASSERT_EQ(b.getCurrentPlayer(), PLAYER_1);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_NULL);

 index = b.convertCellToIndex(Havannah::Cell(3, 6));
 b.play(index);
 ASSERT_EQ(b.getNbEmptyIndices(), 59);
 ASSERT_EQ(b.getLastIndex(), index);
 ASSERT_EQ(b.getCurrentPlayer(), PLAYER_0);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_NULL);

 index = b.convertCellToIndex(Havannah::Cell(2, 7));
 b.play(index);
 ASSERT_EQ(b.getNbEmptyIndices(), 58);
 ASSERT_EQ(b.getLastIndex(), index);
 ASSERT_EQ(b.getCurrentPlayer(), PLAYER_1);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_NULL);
}

TEST(HavannahGroup, findWinnerPath) {
 Havannah::BoardTest<5, true> b;
 b.reset();

 // play cells
 std::vector<Havannah::Cell> gameCells = {
  Havannah::Cell(4, 1), Havannah::Cell(0, 4), 
  Havannah::Cell(4, 3), Havannah::Cell(1, 4), 
  Havannah::Cell(4, 5), Havannah::Cell(2, 4), 
  Havannah::Cell(4, 7), Havannah::Cell(3, 4), 
  Havannah::Cell(4, 2), Havannah::Cell(5, 0), 
  Havannah::Cell(4, 4), Havannah::Cell(5, 1), 
  Havannah::Cell(4, 0), Havannah::Cell(5, 2), 
  Havannah::Cell(4, 6), Havannah::Cell(5, 3), 
  Havannah::Cell(4, 8)
 };
 for (const Havannah::Cell & c : gameCells) {
  int index = b.convertCellToIndex(c);
  b.play(index);
 }
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_0);

 // test cells in winner path
 std::vector<int> winIndices = b.findWinnerPath();
 ASSERT_EQ(int(winIndices.size()), 9);
 std::vector<Havannah::Cell> winCells { 
  Havannah::Cell(4, 0),
   Havannah::Cell(4, 1),
   Havannah::Cell(4, 2),
   Havannah::Cell(4, 3),
   Havannah::Cell(4, 4),
   Havannah::Cell(4, 5),
   Havannah::Cell(4, 6),
   Havannah::Cell(4, 7),
   Havannah::Cell(4, 8)
 };
 for (const Havannah::Cell & c : winCells) {
  int index = b.convertCellToIndex(c);
  ASSERT_NE(winIndices.end(), std::find(winIndices.begin(), 
     winIndices.end(), index));
 }

}

TEST(HavannahGroup, winner_white_fork) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 std::vector<int> gameIndices = {
  28, 21,
  38, 13,
  31, 22,
  23, 57,
  6, 41,
  55, 16,
  24, 42,
  25, 58,
  17, 56,
  14, 49,
  39, 67,
  37
 };
 for (int i : gameIndices)
  b.play(i);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_0);
}

TEST(HavannahGroup, winner_white_ring) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 std::vector<int> gameIndices = {
  37, 22,
  30, 49,
  47, 55,
  46, 38,
  29, 31,
  39
 };
 for (int i : gameIndices)
  b.play(i);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_0);
}

TEST(HavannahGroup, winner_white_bridge) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 std::vector<int> gameIndices = {
  72, 47,
  65, 8,
  75, 26,
  76, 58,
  66, 55,
  64
 };
 for (int i : gameIndices)
  b.play(i);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_0);
}

TEST(HavannahGroup, winner_black_fork) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 std::vector<int> gameIndices = {
  39, 6,
  23, 16,
  31, 33,
  72, 43,
  50, 52,
  26, 17,
  35, 15,
  49, 34,
  41, 24
 };
 for (int i : gameIndices)
  b.play(i);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_1);
}

TEST(HavannahGroup, winner_black_bridge) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 std::vector<int> gameIndices = {
  21, 44,
  40, 59,
  60, 51,
  68, 67,
  56, 52,
  34, 76
 };
 for (int i : gameIndices)
  b.play(i);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_1);
}

TEST(HavannahGroup, winner_black_ring) {
 Havannah::BoardTest<5, true> b;
 b.reset();
 std::vector<int> gameIndices = {
  29, 32,
  38, 42,
  64, 58,
  14, 48,
  25, 40,
  68, 41,
  34, 50,
  74, 57
 };
 for (int i : gameIndices)
  b.play(i);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_1);
}

// https://github.com/facebookexternal/polygames/commit/fdf094d1ab9a0bcc60422e03c44b59d1c2df9f3e#diff-9367f7a48f6ff9d5d578838ec92885baR406
// https://www.littlegolem.net/jsp/game/game.jsp?gid=2130971
TEST(HavannahGroup, littlegolem_lorentz_ttrttr) {
 Havannah::BoardTest<8, true> b;
 b.reset();
 std::vector<Havannah::Cell> gameCells = {
     {7,7}, {7,7},
     {3,6}, {3,11}, {2,8}, {7,2},
     {9,3}, {4,10}, {5,5}, {8,3},
     {8,5}, {7,5}, {6,6}, {7,6},
     {5,8}, {3,9}, {6,9}, {3,8},
     {5,11}, {8,4}, {3,7}, {9,2},
     {4,7}, {6,5}, {6,4}, {6,1},
     {9,4}, {10,2}, {12,3}, {5,6},
     {5,7}, {13,1}, {12,1}, {10,3},
     {13,4}, {10,4}, {10,5}, {9,5},
     {8,6}, {9,6}, {8,7}, {12,2},
     {14,1}, {11,5}, {11,4}, {14,0},
     {7,0}, {13,2}, {14,2}, {13,3},
     {14,3}, {9,1}, {12,4}, {8,1},
     {7,3}, {10,6}, {12,6}, {8,0},
     {10,7}, {12,5}, {13,5}, {7,8},
     {9,9}, {6,8}, {5,9}, {7,11},
     {7,10}, {7,9}, {5,10}, {6,10},
     {5,12}, {9,7}, {8,8}, {4,5},
     {8,9}, {6,11}, {2,6}, {1,7},
     {1,8}, {7,13}, {0,8}, {8,13},
     {1,6}, {7,12}, {0,7}
 };
 std::vector<int> gameIndices(gameCells.size());
 std::transform(gameCells.begin(), gameCells.end(), gameIndices.begin(),
         [&](const Havannah::Cell & c) { return b.convertCellToIndex(c); });
 for (int i : gameIndices)
  b.play(i);
 ASSERT_EQ(b.getWinnerPlayer(), PLAYER_0);
}

