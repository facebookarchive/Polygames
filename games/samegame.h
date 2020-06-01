/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include <functional>
#include <random>
#include <vector>

// samegame board
// https://www.webgamesonline.com/samegame/index.php
//
// 3 |
// 2 |
// 1 |
// 0 |
// i  - - - - - - 
//  j 0 1 2 3 4 5
// 
// colors in [0, nbColors-1]
//
// -1 if no color
//

namespace Samegame {

 class Board {

  public:
   struct Move {
    int _i;
    int _j;
    int _color;
    int _eval;
   };

  protected:
   const int _nbI;
   const int _nbJ;
   const int _nbColors;

   std::vector<int> _dataColors;  // nbI*nbJ
   std::vector<int> _dataGroups;  // nbI*nbJ
   std::vector<int> _groupSizes;  // <= nbI*nbJ
   std::vector<int> _lastIs;      // nbJ
   std::vector<int> _lastJs;      // nbI
   std::vector<Move> _moves;      // <= nbI*nbJ
   int _score;

   static const std::vector<int> DATASETS;

  public:
   Board(int nbI, int nbJ, int nbColors, std::function<int()> fillFunc);
   Board();
   void reset(int iDataset);

   int getNbI() const;
   int getNbJ() const;
   int getNbColors() const;
   int isValid(int i, int j) const;
   int dataColors(int i, int j) const;

   bool isTerminated() const;
   int getScore() const;
   const std::vector<Move> & getMoves() const;
   void play(int n);
   bool play(int i, int j);

   void print() const;
   void printMore() const;

  protected:
   void findGroups();
   void buildGroup(int i0, int j0);
   void findMoves();
   void contractJ(int j);
   void contractI(int i, int j);
   int ind(int i, int j) const;
 };

}  // namespace Samegame

