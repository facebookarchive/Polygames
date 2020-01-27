/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit tests for the Hex game.

#pragma once

#include <hex.h>

///////////////////////////////////////////////////////////////////////////////
// helpers
///////////////////////////////////////////////////////////////////////////////

static void CheckHexPathInfo(const Hex::PathInfo & path, Hex::Color color, bool b1, bool b2, int m) {
 ASSERT_EQ(path._color, color);
 ASSERT_EQ(path._isConnectedBorder1, b1);
 ASSERT_EQ(path._isConnectedBorder2, b2);
 ASSERT_EQ(path._mainPathIndex, m);
}

namespace Hex {

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

    void computeBorderConnection(int index, Color color, 
      bool & isConnectedBorder1, bool & isConnectedBorder2) const {
     return Board<SIZE, PIE>::computeBorderConnection(index, color, 
       isConnectedBorder1, isConnectedBorder2);
    }

    void getPathIndexAndColorAtIndex(int index, int & pathIndex, 
      Color & color) const {
     return Board<SIZE, PIE>::getPathIndexAndColorAtIndex(index, pathIndex, color);
    }

    const std::array<std::array<int,7>,SIZE*SIZE> & getNeighboursBoard() const {
     return Board<SIZE, PIE>::_neighboursBoard;
    }

    std::array<PathInfo, SIZE*SIZE> & getPaths() {
     return Board<SIZE, PIE>::_paths;
    }

    int & getPathsEnd() {
     return Board<SIZE, PIE>::_pathsEnd;
    }

    std::array<int, SIZE*SIZE> & getPathBoard() {
     return Board<SIZE, PIE>::_pathBoard;
    }

    int findNthEmptyIndex(int n) const {
     assert(n < (Board<SIZE,PIE>::_nbEmptyIndices));
     assert(n >= 0);
   
     int nbEmpty = 0;
     int i = 0;
     while (true) {
       if (Board<SIZE, PIE>::_pathBoard[i] == 0) {
         if (nbEmpty == n)
           break;
         else
           nbEmpty++;
       }
       i++;
     }
     return i;
   }

   std::vector<int> findEmptyIndices() const {
     std::vector<int> emptyIndices;
     emptyIndices.reserve(Board<SIZE, PIE>::_nbEmptyIndices);
     for (int k = 0; k < Board<SIZE, PIE>::_nbFullIndices; k++)
       if (Board<SIZE, PIE>::_pathBoard[k] == 0)
         emptyIndices.push_back(k);
     if (Board<SIZE, PIE>::canPie() and
         Board<SIZE, PIE>::_nbEmptyIndices == Board<SIZE, PIE>::_nbIndices - 1)
       emptyIndices.push_back(*(Board<SIZE, PIE>::_lastIndex));
     return emptyIndices;
   }

  };

}  // namespace Hex

///////////////////////////////////////////////////////////////////////////////
// unit tests
///////////////////////////////////////////////////////////////////////////////

TEST(HexGroup, reset) {
 Hex::BoardTest<7,true> b;
 b.reset();
 ASSERT_EQ(b.getNbIndices(), 49);
 ASSERT_EQ(b.getCurrentColor(), Hex::COLOR_BLACK);
 ASSERT_EQ(b.getWinnerColor(), Hex::COLOR_NONE);
 ASSERT_FALSE(b.getLastIndex());
 ASSERT_EQ(b.getNbEmptyIndices(), 49);
}

TEST(HexGroup, copyConstructor) {
 Hex::BoardTest<7,true> b0;
 b0.reset();
 Hex::BoardTest<7,true> b(b0);
 ASSERT_EQ(b.getNbIndices(), 49);
 ASSERT_EQ(b.getCurrentColor(), Hex::COLOR_BLACK);
 ASSERT_EQ(b.getWinnerColor(), Hex::COLOR_NONE);
 ASSERT_FALSE(b.getLastIndex());
 ASSERT_EQ(b.getNbEmptyIndices(), 49);
}

TEST(HexGroup, resetNeighboursTopLeft) {
 Hex::BoardTest<7,true> b;
 b.reset();
 int index = b.convertCellToIndex(Hex::Cell(0, 0));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Hex::Cell(0,1)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Hex::Cell(1,0)));
 ASSERT_EQ(refNeighbourIndices[2], -1);
}

TEST(HexGroup, resetNeighboursTopCenter) {
 Hex::BoardTest<7,true> b;
 b.reset();
 int index = b.convertCellToIndex(Hex::Cell(0, 3));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Hex::Cell(0,2)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Hex::Cell(0,4)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Hex::Cell(1,2)));
 ASSERT_EQ(refNeighbourIndices[3], b.convertCellToIndex(Hex::Cell(1,3)));
 ASSERT_EQ(refNeighbourIndices[4], -1);
}

TEST(HexGroup, resetNeighboursTopRight) {
 Hex::BoardTest<7,true> b;
 b.reset();
 int index = b.convertCellToIndex(Hex::Cell(0, 6));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Hex::Cell(0,5)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Hex::Cell(1,5)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Hex::Cell(1,6)));
 ASSERT_EQ(refNeighbourIndices[3], -1);
}

TEST(HexGroup, resetNeighboursMiddleRight) {
 Hex::BoardTest<7,true> b;
 b.reset();
 int index = b.convertCellToIndex(Hex::Cell(3, 6));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Hex::Cell(2,6)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Hex::Cell(3,5)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Hex::Cell(4,5)));
 ASSERT_EQ(refNeighbourIndices[3], b.convertCellToIndex(Hex::Cell(4,6)));
 ASSERT_EQ(refNeighbourIndices[4], -1);
}

TEST(HexGroup, resetNeighboursBottomRight) {
 Hex::BoardTest<7,true> b;
 b.reset();
 int index = b.convertCellToIndex(Hex::Cell(6, 6));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Hex::Cell(5,6)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Hex::Cell(6,5)));
 ASSERT_EQ(refNeighbourIndices[2], -1);
}

TEST(HexGroup, resetNeighboursBottomCenter) {
 Hex::BoardTest<7,true> b;
 b.reset();
 int index = b.convertCellToIndex(Hex::Cell(6, 3));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Hex::Cell(5,3)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Hex::Cell(5,4)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Hex::Cell(6,2)));
 ASSERT_EQ(refNeighbourIndices[3], b.convertCellToIndex(Hex::Cell(6,4)));
 ASSERT_EQ(refNeighbourIndices[4], -1);
}

TEST(HexGroup, resetNeighboursBottomLeft) {
 Hex::BoardTest<7,true> b;
 b.reset();
 int index = b.convertCellToIndex(Hex::Cell(6, 0));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Hex::Cell(5,0)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Hex::Cell(5,1)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Hex::Cell(6,1)));
 ASSERT_EQ(refNeighbourIndices[3], -1);
}

TEST(HexGroup, resetNeighboursMiddleLeft) {
 Hex::BoardTest<7,true> b;
 b.reset();
 int index = b.convertCellToIndex(Hex::Cell(3, 0));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Hex::Cell(2,0)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Hex::Cell(2,1)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Hex::Cell(3,1)));
 ASSERT_EQ(refNeighbourIndices[3], b.convertCellToIndex(Hex::Cell(4,0)));
 ASSERT_EQ(refNeighbourIndices[4], -1);
}

TEST(HexGroup, resetNeighboursMiddleCenter) {
 Hex::BoardTest<7,true> b;
 b.reset();
 int index = b.convertCellToIndex(Hex::Cell(3, 3));
 const std::array<int, 7> refNeighbourIndices = b.getNeighboursBoard()[index];
 ASSERT_EQ(refNeighbourIndices[0], b.convertCellToIndex(Hex::Cell(2,3)));
 ASSERT_EQ(refNeighbourIndices[1], b.convertCellToIndex(Hex::Cell(2,4)));
 ASSERT_EQ(refNeighbourIndices[2], b.convertCellToIndex(Hex::Cell(3,2)));
 ASSERT_EQ(refNeighbourIndices[3], b.convertCellToIndex(Hex::Cell(3,4)));
 ASSERT_EQ(refNeighbourIndices[4], b.convertCellToIndex(Hex::Cell(4,2)));
 ASSERT_EQ(refNeighbourIndices[5], b.convertCellToIndex(Hex::Cell(4,3)));
 ASSERT_EQ(refNeighbourIndices[6], -1);
}

TEST(HexGroup, resetPaths) {
 Hex::BoardTest<5,true> b;
 b.reset();
 ASSERT_EQ(b.getPathsEnd(), 1);
 const Hex::PathInfo & p0 = b.getPaths()[0];
 ASSERT_EQ(p0._color, Hex::COLOR_NONE);
 ASSERT_EQ(p0._isConnectedBorder1, false);
 ASSERT_EQ(p0._isConnectedBorder2, false);
 ASSERT_EQ(p0._mainPathIndex, 0);
}

TEST(HexGroup, resetPathBoard) {
 Hex::BoardTest<5,true> b;
 b.reset();
 for (int i=0; i<25; i++) {
  ASSERT_EQ(b.getPathBoard()[i], 0); 
  int pathIndex;
  Hex::Color color;
  b.getPathIndexAndColorAtIndex(i, pathIndex, color);
  ASSERT_EQ(pathIndex, 0);
  ASSERT_EQ(color, Hex::COLOR_NONE);
 }
}

TEST(HexGroup, resetGetters1) {
 Hex::BoardTest<11,true> b;
 b.reset();
 ASSERT_EQ(b.getNbIndices(), 121);
 ASSERT_EQ(b.getNbEmptyIndices(), 121);
 ASSERT_FALSE(b.getLastIndex());
 ASSERT_EQ(b.getCurrentColor(), Hex::COLOR_BLACK);
 ASSERT_EQ(b.getWinnerColor(), Hex::COLOR_NONE);
}

TEST(HexGroup, isValid1) {
 Hex::BoardTest<8,true> b;
 b.reset();

 ASSERT_EQ(b.isValidIndex(12), true);
 ASSERT_EQ(b.isValidIndex(0), true);
 ASSERT_EQ(b.isValidIndex(-1), false);
 ASSERT_EQ(b.isValidIndex(63), true);
 ASSERT_EQ(b.isValidIndex(64), false);

 ASSERT_EQ(b.isValidCell(Hex::Cell(3, 4)), true);
 ASSERT_EQ(b.isValidCell(Hex::Cell(0, 2)), true);
 ASSERT_EQ(b.isValidCell(Hex::Cell(-1, 1)), false);
 ASSERT_EQ(b.isValidCell(Hex::Cell(2, 0)), true);
 ASSERT_EQ(b.isValidCell(Hex::Cell(2, -1)), false);
 ASSERT_EQ(b.isValidCell(Hex::Cell(3, 7)), true);
 ASSERT_EQ(b.isValidCell(Hex::Cell(3, 8)), false);
 ASSERT_EQ(b.isValidCell(Hex::Cell(7, 2)), true);
 ASSERT_EQ(b.isValidCell(Hex::Cell(8, 1)), false);
}


TEST(HexGroup, convertIndexToCell) {
 Hex::BoardTest<9,true> b;
 b.reset();
 ASSERT_EQ(b.convertIndexToCell(3), Hex::Cell(0, 3));
 ASSERT_EQ(b.convertIndexToCell(13), Hex::Cell(1, 4));
 ASSERT_EQ(b.convertIndexToCell(37), Hex::Cell(4, 1));
}

TEST(HexGroup, convertCellToIndex) {
 Hex::BoardTest<10,true> b;
 b.reset();
 ASSERT_EQ(b.convertCellToIndex(Hex::Cell(0, 2)), 2);
 ASSERT_EQ(b.convertCellToIndex(Hex::Cell(1, 0)), 10);
 ASSERT_EQ(b.convertCellToIndex(Hex::Cell(7, 5)), 75);
}

TEST(HexGroup, computeBorderConnection0) {
 Hex::BoardTest<12,true> b;
 b.reset();

 int index;
 bool isConnectedBorder1;
 bool isConnectedBorder2;

 // left
 index = b.convertCellToIndex(Hex::Cell(7, 0));
 b.computeBorderConnection(index, Hex::COLOR_WHITE, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, true);
 ASSERT_EQ(isConnectedBorder2, false);

 // right
 index = b.convertCellToIndex(Hex::Cell(7, 11));
 b.computeBorderConnection(index, Hex::COLOR_WHITE, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, false);
 ASSERT_EQ(isConnectedBorder2, true);

 // top
 index = b.convertCellToIndex(Hex::Cell(0, 7));
 b.computeBorderConnection(index, Hex::COLOR_WHITE, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, false);
 ASSERT_EQ(isConnectedBorder2, false);

 // bottom
 index = b.convertCellToIndex(Hex::Cell(11, 7));
 b.computeBorderConnection(index, Hex::COLOR_WHITE, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, false);
 ASSERT_EQ(isConnectedBorder2, false);
}

TEST(HexGroup, computeBorderConnection1) {
 Hex::BoardTest<12,true> b;
 b.reset();

 int index;
 bool isConnectedBorder1;
 bool isConnectedBorder2;

 // left
 index = b.convertCellToIndex(Hex::Cell(7, 0));
 b.computeBorderConnection(index, Hex::COLOR_BLACK, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, false);
 ASSERT_EQ(isConnectedBorder2, false);

 // right
 index = b.convertCellToIndex(Hex::Cell(7, 11));
 b.computeBorderConnection(index, Hex::COLOR_BLACK, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, false);
 ASSERT_EQ(isConnectedBorder2, false);

 // top
 index = b.convertCellToIndex(Hex::Cell(0, 7));
 b.computeBorderConnection(index, Hex::COLOR_BLACK, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, true);
 ASSERT_EQ(isConnectedBorder2, false);

 // bottom
 index = b.convertCellToIndex(Hex::Cell(11, 7));
 b.computeBorderConnection(index, Hex::COLOR_BLACK, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, false);
 ASSERT_EQ(isConnectedBorder2, true);
}

TEST(HexGroup, computeBorderConnectionNull) {
 Hex::BoardTest<12,true> b;
 b.reset();

 int index;
 bool isConnectedBorder1;
 bool isConnectedBorder2;

 // left
 index = b.convertCellToIndex(Hex::Cell(7, 0));
 b.computeBorderConnection(index, Hex::COLOR_NONE, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, false);
 ASSERT_EQ(isConnectedBorder2, false);

 // right
 index = b.convertCellToIndex(Hex::Cell(7, 11));
 b.computeBorderConnection(index, Hex::COLOR_NONE, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, false);
 ASSERT_EQ(isConnectedBorder2, false);

 // top
 index = b.convertCellToIndex(Hex::Cell(0, 7));
 b.computeBorderConnection(index, Hex::COLOR_NONE, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, false);
 ASSERT_EQ(isConnectedBorder2, false);

 // bottom
 index = b.convertCellToIndex(Hex::Cell(11, 7));
 b.computeBorderConnection(index, Hex::COLOR_NONE, 
   isConnectedBorder1, isConnectedBorder2);
 ASSERT_EQ(isConnectedBorder1, false);
 ASSERT_EQ(isConnectedBorder2, false);
}

TEST(HexGroup, getPathIndexAndColorAtIndex) {
 Hex::BoardTest<7,true> b;
 b.reset();
 b.getPathsEnd() = 3;
 b.getPaths()[1] = Hex::PathInfo(1, Hex::COLOR_BLACK, false, false);
 b.getPaths()[2] = Hex::PathInfo(2, Hex::COLOR_WHITE, false, false);

 int index_1_1 = b.convertCellToIndex(Hex::Cell(1, 1));
 int index_3_1 = b.convertCellToIndex(Hex::Cell(3, 1));
 int index_2_3 = b.convertCellToIndex(Hex::Cell(2, 3));

 b.getPathBoard()[index_3_1] = 1;
 b.getPathBoard()[index_2_3] = 2;

 int pathIndex;
 Hex::Color color;

 b.getPathIndexAndColorAtIndex(index_1_1, pathIndex, color);
 ASSERT_EQ(pathIndex, 0);
 ASSERT_EQ(color, Hex::COLOR_NONE);

 b.getPathIndexAndColorAtIndex(index_3_1, pathIndex, color);
 ASSERT_EQ(pathIndex, 1);
 ASSERT_EQ(color, Hex::COLOR_BLACK);

 b.getPathIndexAndColorAtIndex(index_2_3, pathIndex, color);
 ASSERT_EQ(pathIndex, 2);
 ASSERT_EQ(color, Hex::COLOR_WHITE);
}

TEST(HexGroup, play0) {
 Hex::BoardTest<5,true> b;
 b.reset();
 int index;

 index = b.convertCellToIndex(Hex::Cell(0, 0));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 2);
 ASSERT_EQ(b.getPathBoard()[index], 1);
 CheckHexPathInfo(b.getPaths()[1], Hex::COLOR_BLACK, true, false, 1);

 index = b.convertCellToIndex(Hex::Cell(0, 2));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 3);
 ASSERT_EQ(b.getPathBoard()[index], 2);
 CheckHexPathInfo(b.getPaths()[2], Hex::COLOR_WHITE, false, false, 2);

 index = b.convertCellToIndex(Hex::Cell(3, 0));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 4);
 ASSERT_EQ(b.getPathBoard()[index], 3);
 CheckHexPathInfo(b.getPaths()[3], Hex::COLOR_BLACK, false, false, 3);

 index = b.convertCellToIndex(Hex::Cell(2, 1));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 5);
 ASSERT_EQ(b.getPathBoard()[index], 4);
 CheckHexPathInfo(b.getPaths()[4], Hex::COLOR_WHITE, false, false, 4);

 index = b.convertCellToIndex(Hex::Cell(0, 1));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 5);
 ASSERT_EQ(b.getPathBoard()[index], 1);
 CheckHexPathInfo(b.getPaths()[1], Hex::COLOR_BLACK, true, false, 1);

 index = b.convertCellToIndex(Hex::Cell(2, 2));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 5);
 ASSERT_EQ(b.getPathBoard()[index], 4);
 CheckHexPathInfo(b.getPaths()[4], Hex::COLOR_WHITE, false, false, 4);

 index = b.convertCellToIndex(Hex::Cell(2, 4));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 6);
 ASSERT_EQ(b.getPathBoard()[index], 5);
 CheckHexPathInfo(b.getPaths()[5], Hex::COLOR_BLACK, false, false, 5);

 index = b.convertCellToIndex(Hex::Cell(4, 2));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 7);
 ASSERT_EQ(b.getPathBoard()[index], 6);
 CheckHexPathInfo(b.getPaths()[6], Hex::COLOR_WHITE, false, false, 6);

 index = b.convertCellToIndex(Hex::Cell(1, 0));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 7);
 ASSERT_EQ(b.getPathBoard()[index], 1);
 CheckHexPathInfo(b.getPaths()[1], Hex::COLOR_BLACK, true, false, 1);

 index = b.convertCellToIndex(Hex::Cell(4, 0));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 8);
 ASSERT_EQ(b.getPathBoard()[index], 7);
 CheckHexPathInfo(b.getPaths()[7], Hex::COLOR_WHITE, true, false, 7);

 index = b.convertCellToIndex(Hex::Cell(1, 3));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 9);
 ASSERT_EQ(b.getPathBoard()[index], 8);
 CheckHexPathInfo(b.getPaths()[8], Hex::COLOR_BLACK, false, false, 8);

 index = b.convertCellToIndex(Hex::Cell(2, 0));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 9);
 ASSERT_EQ(b.getPathBoard()[index], 4);
 CheckHexPathInfo(b.getPaths()[4], Hex::COLOR_WHITE, true, false, 4);

 index = b.convertCellToIndex(Hex::Cell(3, 2));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 10);
 ASSERT_EQ(b.getPathBoard()[index], 9);
 CheckHexPathInfo(b.getPaths()[9], Hex::COLOR_BLACK, false, false, 9);

 index = b.convertCellToIndex(Hex::Cell(4, 4));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 11);
 ASSERT_EQ(b.getPathBoard()[index], 10);
 CheckHexPathInfo(b.getPaths()[10], Hex::COLOR_WHITE, false, true, 10);

 index = b.convertCellToIndex(Hex::Cell(3, 3));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 11);
 ASSERT_EQ(b.getPathBoard()[index], 5);
 CheckHexPathInfo(b.getPaths()[5], Hex::COLOR_BLACK, false, false, 5);
 ASSERT_EQ(b.getPaths()[9]._mainPathIndex, 5);

 index = b.convertCellToIndex(Hex::Cell(1, 4));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 12);
 ASSERT_EQ(b.getPathBoard()[index], 11);
 CheckHexPathInfo(b.getPaths()[11], Hex::COLOR_WHITE, false, true, 11);

 index = b.convertCellToIndex(Hex::Cell(2, 3));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 12);
 ASSERT_EQ(b.getPathBoard()[index], 5);
 CheckHexPathInfo(b.getPaths()[5], Hex::COLOR_BLACK, false, false, 5);
 ASSERT_EQ(b.getPaths()[8]._mainPathIndex, 5);
 ASSERT_EQ(b.getPaths()[9]._mainPathIndex, 5);

 index = b.convertCellToIndex(Hex::Cell(0, 4));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 12);
 ASSERT_EQ(b.getPathBoard()[index], 11);
 CheckHexPathInfo(b.getPaths()[11], Hex::COLOR_WHITE, false, true, 11);

 index = b.convertCellToIndex(Hex::Cell(1, 1));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 12);
 ASSERT_EQ(b.getPathBoard()[index], 1);
 CheckHexPathInfo(b.getPaths()[1], Hex::COLOR_BLACK, true, false, 1);

 index = b.convertCellToIndex(Hex::Cell(3, 4));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 12);
 ASSERT_EQ(b.getPathBoard()[index], 10);
 CheckHexPathInfo(b.getPaths()[10], Hex::COLOR_WHITE, false, true, 10);

 index = b.convertCellToIndex(Hex::Cell(4, 3));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 12);
 ASSERT_EQ(b.getPathBoard()[index], 5);
 CheckHexPathInfo(b.getPaths()[5], Hex::COLOR_BLACK, false, true, 5);

 index = b.convertCellToIndex(Hex::Cell(0, 3));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 12);
 ASSERT_EQ(b.getPathBoard()[index], 2);
 CheckHexPathInfo(b.getPaths()[2], Hex::COLOR_WHITE, false, true, 2);
 ASSERT_EQ(b.getPaths()[11]._mainPathIndex, 2);

 index = b.convertCellToIndex(Hex::Cell(3, 1));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 12);
 ASSERT_EQ(b.getPathBoard()[index], 3);
 CheckHexPathInfo(b.getPaths()[3], Hex::COLOR_BLACK, false, true, 3);
 ASSERT_EQ(b.getPaths()[5]._mainPathIndex, 3);
 ASSERT_EQ(b.getPaths()[8]._mainPathIndex, 3);
 ASSERT_EQ(b.getPaths()[9]._mainPathIndex, 3);

 index = b.convertCellToIndex(Hex::Cell(4, 1));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 12);
 ASSERT_EQ(b.getPathBoard()[index], 6);
 CheckHexPathInfo(b.getPaths()[6], Hex::COLOR_WHITE, true, false, 6);
 ASSERT_EQ(b.getPaths()[7]._mainPathIndex, 6);

 index = b.convertCellToIndex(Hex::Cell(1, 2));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 12);
 ASSERT_EQ(b.getPathBoard()[index], 1);
 CheckHexPathInfo(b.getPaths()[1], Hex::COLOR_BLACK, true, true, 1);
 ASSERT_EQ(b.getPaths()[3]._mainPathIndex, 1);
 ASSERT_EQ(b.getPaths()[5]._mainPathIndex, 1);
 ASSERT_EQ(b.getPaths()[8]._mainPathIndex, 1);
 ASSERT_EQ(b.getPaths()[9]._mainPathIndex, 1);
 ASSERT_EQ(b.getWinnerColor(), Hex::COLOR_BLACK);
}

TEST(HexGroup, play1) {
 Hex::BoardTest<5,true> b;
 b.reset();
 int index;

 index = b.convertCellToIndex(Hex::Cell(2, 1));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 2);
 ASSERT_EQ(b.getPathBoard()[index], 1);
 ASSERT_EQ(b.getPaths()[1]._color, Hex::COLOR_BLACK);
 ASSERT_EQ(b.getPaths()[1]._isConnectedBorder1, false);
 ASSERT_EQ(b.getPaths()[1]._isConnectedBorder2, false);
 ASSERT_EQ(b.getPaths()[1]._mainPathIndex, 1);

 index = b.convertCellToIndex(Hex::Cell(3, 1));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 3);
 ASSERT_EQ(b.getPathBoard()[index], 2);
 ASSERT_EQ(b.getPaths()[2]._color, Hex::COLOR_WHITE);
 ASSERT_EQ(b.getPaths()[2]._isConnectedBorder1, false);
 ASSERT_EQ(b.getPaths()[2]._isConnectedBorder2, false);
 ASSERT_EQ(b.getPaths()[2]._mainPathIndex, 2);

 index = b.convertCellToIndex(Hex::Cell(2, 2));
 b.play(index);
 ASSERT_EQ(b.getPathsEnd(), 3);
 ASSERT_EQ(b.getPathBoard()[index], 1);
 ASSERT_EQ(b.getPaths()[1]._color, Hex::COLOR_BLACK);
 ASSERT_EQ(b.getPaths()[1]._isConnectedBorder1, false);
 ASSERT_EQ(b.getPaths()[1]._isConnectedBorder2, false);
 ASSERT_EQ(b.getPaths()[1]._mainPathIndex, 1);
}

TEST(HexGroup, play2) {
 Hex::BoardTest<5,true> b;
 b.reset();
 int index;

 index = b.convertCellToIndex(Hex::Cell(2, 1));
 b.play(index);
 index = b.convertCellToIndex(Hex::Cell(3, 1));
 b.play(index);
 index = b.convertCellToIndex(Hex::Cell(2, 2));
 b.play(index);

 ASSERT_EQ(b.getPathsEnd(), 3);
 ASSERT_EQ(b.getPathBoard()[index], 1);
 ASSERT_EQ(b.getPaths()[1]._color, Hex::COLOR_BLACK);
 ASSERT_EQ(b.getPaths()[1]._isConnectedBorder1, false);
 ASSERT_EQ(b.getPaths()[1]._isConnectedBorder2, false);
 ASSERT_EQ(b.getPaths()[1]._mainPathIndex, 1);
}

TEST(HexGroup, play3) {
 Hex::BoardTest<5,true> b;
 b.reset();
 int index;

 ASSERT_EQ(b.getNbIndices(), 25);

 ASSERT_EQ(b.getNbEmptyIndices(), 25);
 ASSERT_FALSE(b.getLastIndex());
 ASSERT_EQ(b.getCurrentColor(), Hex::COLOR_BLACK);
 ASSERT_EQ(b.getWinnerColor(), Hex::COLOR_NONE);

 index = b.convertCellToIndex(Hex::Cell(2, 1));
 b.play(index);
 ASSERT_EQ(b.getNbEmptyIndices(), 24);
 ASSERT_EQ(*b.getLastIndex(), index);
 ASSERT_EQ(b.getCurrentColor(), Hex::COLOR_WHITE);
 ASSERT_EQ(b.getWinnerColor(), Hex::COLOR_NONE);

 index = b.convertCellToIndex(Hex::Cell(3, 1));
 b.play(index);
 ASSERT_EQ(b.getNbEmptyIndices(), 23);
 ASSERT_EQ(*b.getLastIndex(), index);
 ASSERT_EQ(b.getCurrentColor(), Hex::COLOR_BLACK);
 ASSERT_EQ(b.getWinnerColor(), Hex::COLOR_NONE);

 index = b.convertCellToIndex(Hex::Cell(2, 2));
 b.play(index);
 ASSERT_EQ(b.getNbEmptyIndices(), 22);
 ASSERT_EQ(*b.getLastIndex(), index);
 ASSERT_EQ(b.getCurrentColor(), Hex::COLOR_WHITE);
 ASSERT_EQ(b.getWinnerColor(), Hex::COLOR_NONE);
}

TEST(HexGroup, findWinnerPath) {
 Hex::BoardTest<5,true> b;
 b.reset();

 // play cells
 std::vector<Hex::Cell> gameCells = {
  Hex::Cell(4, 4),
  Hex::Cell(1, 1), Hex::Cell(2, 0), Hex::Cell(2, 3), Hex::Cell(3, 2), Hex::Cell(4, 2), 
  Hex::Cell(1, 2), Hex::Cell(0, 4), Hex::Cell(3, 0), Hex::Cell(1, 0), Hex::Cell(3, 3), 
  Hex::Cell(1, 3), Hex::Cell(0, 0), Hex::Cell(2, 1), Hex::Cell(1, 4), Hex::Cell(2, 2)
 };
 for (const Hex::Cell & c : gameCells) {
  int index = b.convertCellToIndex(c);
  b.play(index);
 }
 ASSERT_EQ(b.getWinnerColor(), Hex::COLOR_WHITE);

 // test cells in winner path
 std::vector<int> winIndices = b.findWinnerPath();
 ASSERT_EQ(int(winIndices.size()), 7);
 std::vector<Hex::Cell> winCells { Hex::Cell(1, 1), Hex::Cell(2, 3), Hex::Cell(0, 4), 
  Hex::Cell(1, 0), Hex::Cell(1, 3), Hex::Cell(2, 1), Hex::Cell(2, 2) };
 for (const Hex::Cell & c : winCells) {
  int index = b.convertCellToIndex(c);
  ASSERT_NE(winIndices.end(), std::find(winIndices.begin(), 
     winIndices.end(), index));
 }
}

TEST(HexGroup, findEmptyIndices) {
 Hex::BoardTest<5,true> b;
 b.reset();

 // play cells
 std::vector<Hex::Cell> gameCells = {
  Hex::Cell(1, 1), Hex::Cell(2, 0), Hex::Cell(2, 3), Hex::Cell(3, 2), Hex::Cell(4, 2), 
  Hex::Cell(1, 2), Hex::Cell(0, 4), Hex::Cell(3, 0), Hex::Cell(1, 0), Hex::Cell(3, 3), 
  Hex::Cell(1, 3), Hex::Cell(0, 0), Hex::Cell(2, 1), Hex::Cell(1, 4), Hex::Cell(2, 2)
 };
 for (const Hex::Cell & c : gameCells) {
  int index = b.convertCellToIndex(c);
  b.play(index);
 }

 // test cells in winner path
 std::vector<Hex::Cell> expectedEmptyCells = { 
  Hex::Cell(0, 1), Hex::Cell(0, 2), Hex::Cell(0, 3), Hex::Cell(2, 4), Hex::Cell(3, 1), 
  Hex::Cell(3, 4), Hex::Cell(4, 0), Hex::Cell(4, 1), Hex::Cell(4, 3), Hex::Cell(4, 4) 
 };

 std::vector<int> emptyIndices = b.findEmptyIndices();
 ASSERT_EQ(emptyIndices.size(), expectedEmptyCells.size());
 ASSERT_EQ(unsigned(b.getNbEmptyIndices()), 
   expectedEmptyCells.size());

 for (unsigned i=0; i<emptyIndices.size(); i++) {
  int index = b.convertCellToIndex(expectedEmptyCells[i]);
  ASSERT_EQ(index, emptyIndices[i]);
 }
}

TEST(HexGroup, findNthEmptyIndex) {
 Hex::BoardTest<5,true> b;
 b.reset();

 // play cells
 std::vector<Hex::Cell> gameCells = {
  Hex::Cell(1, 1), Hex::Cell(2, 0), Hex::Cell(2, 3), Hex::Cell(3, 2), Hex::Cell(4, 2), 
  Hex::Cell(1, 2), Hex::Cell(0, 4), Hex::Cell(3, 0), Hex::Cell(1, 0), Hex::Cell(3, 3), 
  Hex::Cell(1, 3), Hex::Cell(0, 0), Hex::Cell(2, 1), Hex::Cell(1, 4), Hex::Cell(2, 2)
 };
 for (const Hex::Cell & c : gameCells) {
  int index = b.convertCellToIndex(c);
  b.play(index);
 }

 // test cells in winner path
 std::vector<Hex::Cell> expectedEmptyCells = { 
  Hex::Cell(0, 1), Hex::Cell(0, 2), Hex::Cell(0, 3), Hex::Cell(2, 4), Hex::Cell(3, 1), 
  Hex::Cell(3, 4), Hex::Cell(4, 0), Hex::Cell(4, 1), Hex::Cell(4, 3), Hex::Cell(4, 4) 
 };

 ASSERT_EQ(unsigned(b.getNbEmptyIndices()), 
   expectedEmptyCells.size());

 for (unsigned i=0; i<expectedEmptyCells.size(); i++) {
  int indexExpected = b.convertCellToIndex(expectedEmptyCells[i]);
  int indexFound = b.findNthEmptyIndex(i);
  ASSERT_EQ(indexFound, indexExpected);
 }
}

TEST(HexGroup, winner_white) {
    Hex::BoardTest<9,true> b;
    b.reset();
    std::vector<int> gameIndices = {
        0, 37, 21,
        47, 23,
        40, 24,
        50, 67,
        43, 69,
        35, 51,
        42, 41,
        49, 48,
        39, 38,
        46, 45,
        36
    };
    for (int i : gameIndices)
        b.play(i);
    ASSERT_EQ(b.getWinnerColor(), Hex::COLOR_WHITE);
    ASSERT_EQ(b.isGameFinished(), true);
}


TEST(HexGroup, playPie1) {
 Hex::BoardTest<5,true> b;
 b.reset();

 ASSERT_EQ(b.getNbEmptyIndices(), 25);

 int index;

 index = b.convertCellToIndex(Hex::Cell(2, 1));
 b.play(index);
 ASSERT_EQ(b.getNbEmptyIndices(), 24);
 ASSERT_EQ(b.canPie(), true);
 ASSERT_EQ(b.findEmptyIndices().size(), 25);

 index = b.convertCellToIndex(Hex::Cell(1, 1));
 b.play(index);
 ASSERT_EQ(b.getNbEmptyIndices(), 23);
 ASSERT_EQ(b.canPie(), false);
 ASSERT_EQ(b.findEmptyIndices().size(), 23);
}


TEST(HexGroup, playPie2) {
 Hex::BoardTest<5,true> b;
 b.reset();

 ASSERT_EQ(b.getNbEmptyIndices(), 25);

 int index;

 index = b.convertCellToIndex(Hex::Cell(2, 1));
 b.play(index);
 ASSERT_EQ(b.getNbEmptyIndices(), 24);
 ASSERT_EQ(b.canPie(), true);
 ASSERT_EQ(b.findEmptyIndices().size(), 25);

 index = b.convertCellToIndex(Hex::Cell(2, 1));
 b.play(index);
 ASSERT_EQ(b.getNbEmptyIndices(), 24);
 ASSERT_EQ(b.canPie(), false);
 ASSERT_EQ(b.findEmptyIndices().size(), 24);
}

TEST(HexGroup, winner_pie) {
    Hex::BoardTest<9,true> b;
    b.reset();
    std::vector<int> gameIndices = {
      13, 
      13, 29,
      23, 47,
      40, 56,
      50, 43,
      67, 61,
      75, 59,
      58, 49,
      41, 32,
      31, 22,
      14, 5,
      4
    };

    for (int i : gameIndices)
        b.play(i);
    ASSERT_EQ(b.getWinnerColor(), Hex::COLOR_BLACK);
    ASSERT_EQ(b.isGameFinished(), true);

    /*
    for (int i : gameIndices) {
        auto c = b.convertIndexToCell(i);
        std::swap(c.first, c.second);
        std::cout << b.convertCellToIndex(c) << std::endl;
    }
    */

}

