/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "samegame_state.h"

///////////////////////////////////////////////////////////////////////////////
// Action
///////////////////////////////////////////////////////////////////////////////

Samegame::Action::Action(int i, int j, int indexInActions, int nj) {
 _loc[0] = 0;
 _loc[1] = i;
 _loc[2] = j;
 _hash = uint32_t(i * nj + j); 
 _i = indexInActions;  // (position in _legalActions)
}

///////////////////////////////////////////////////////////////////////////////
// State
///////////////////////////////////////////////////////////////////////////////

Samegame::State::State(int seed) :
 ::State(seed)
{
 Initialize();
}

Samegame::State::State(int seed, int history, bool turnFeatures) :
 ::State(seed)
{
 _history = history;
 _turnFeatures = turnFeatures;
 Initialize();
}

void Samegame::State::findFeatures() {
 std::fill(_features.begin(), _features.end(), 0.f);
 const int ni = _board.getNbI();
 const int nj = _board.getNbJ();
 const int nc = _board.getNbColors();
 for (int i=0; i<ni; i++) {
  for (int j=0; j<nj; j++) {
   int c = _board.dataColors(i, j);
   assert(c >= -1);
   assert(c < nc);
   _features[(i*nj+j)*(nc+1)+c+1] = 1.f;
  }
 }
}

void Samegame::State::findActions() {
 const auto moves = _board.getMoves();
 const int nj = _board.getNbJ();
 _legalActions.clear();
 _legalActions.reserve(moves.size());
 for (unsigned k=0; k<moves.size(); ++k) {
  const Samegame::Board::Move & move = moves[k];
  _legalActions.push_back(std::make_shared<Samegame::Action>(
     move._i, move._j, k, nj));
 }
}

void Samegame::State::Initialize() {
 _board.reset(0);  // TODO dataset
 _moves.clear();
 _hash = 0;
 _status = GameStatus::player0Turn;

 // features
 _featSize = {1+_board.getNbColors(), _board.getNbI(), _board.getNbJ()};
 _features = std::vector<float>(_featSize[0] * _featSize[1] * _featSize[2]);
 findFeatures();
 fillFullFeatures();

 // actions
 _actionSize = {1, _board.getNbI(), _board.getNbJ()};
 findActions();
}

void Samegame::State::ApplyAction(const _Action& action) {

 assert(not _board.isTerminated());

 // play move
 int i = action.GetY();
 int j = action.GetZ();
 _board.play(i, j);

 // update game status
 if (not _board.isTerminated()) {
  _status = GameStatus::player0Turn;
 }
 else {
  if (_board.getScore() > 500)  // TODO maximize score
   _status = GameStatus::player0Win;
  else
   _status = GameStatus::player1Win;
 }

 // update features and actions
 findFeatures();
 fillFullFeatures();
 findActions();

 // TODO update hash
}

void Samegame::State::DoGoodAction() {
 return DoRandomAction();
}

std::unique_ptr<mcts::State> Samegame::State::clone_() const {
 return std::make_unique<Samegame::State>(*this);
}

/*
   std::string Samegame::State::stateDescription() const {

   const auto& feats = _features;
   const auto& sizes = _featSize;

   int ni = sizes[1];
   int nj = sizes[2];

   auto ind = [ni,nj](int i, int j, int k) 
   { return (k*ni + i)*nj + j; };

   std::string str;

   str += "Samegame\n";

   str += "  ";
   for (int k=0; k<ni; k++) {
   str += " ";
   if (k<10) str += " ";
   str += std::to_string(k) + " ";
   }
   str += "\n";

   for (int i=0; i<ni; i++) {

   str += "   ";
   for (int k=0; k<i; k++) str += "  ";

   for (int k=0; k<SIZE-i-1; k++) str += "    ";
   for (int k=0; k<SIZE+i and k<3*SIZE-i-1; k++) str += "----";
   str += "\n";

   if (i<10) str += " ";
   str += std::to_string(i) + " ";
   for (int k=0; k<i; k++) str += "  ";
   for (int j=0; j<nj; j++) {

   if (_board.isValidCell({i,j}))
   str += "\\ ";
   else if (j<SIZE)
   str += "  ";

   if (feats[ind(i,j,0)] && feats[ind(i,j,1)])
   str += "! ";
   else if (feats[ind(i,j,0)])
   str += "X ";
   else if (feats[ind(i,j,1)])
   str += "O ";
   else if (_board.isValidCell({i,j}))
   str += ". ";
   else if (j<SIZE)
   str += "  ";
   else
   continue;
   }

   str += "\\ \n";
   }

   str += "  ";
   for (int k=0; k<ni; k++) str += "  ";
   for (int k=SIZE-2; _board.isValidCell({SIZE,k}); k++) str += "----";
   str += "\n";

   str += "    ";
   for (int k=0; k<SIZE-1; k++) str += "    ";
   for (int k=0; k<ni; k++) {
   str += " ";
   if (k<10) str += " ";
   str += std::to_string(k) + " ";
}
str += "\n";

return str;

}

std::string Samegame::State::actionDescription(const _Action & action) const {
 return std::to_string(action.GetY()) + "," + std::to_string(action.GetZ());
}

std::string Samegame::State::actionsDescription() {
 std::ostringstream oss;
 for (const auto & a : _legalActions) {
  oss << a->GetY() << "," << a->GetZ() << " ";
 }
 oss << std::endl;
 return oss.str();
}


int Samegame::State::parseAction(const std::string& str) {
 std::istringstream iss(str);
 try {
  std::string token;
  if (not std::getline(iss, token, ',')) throw -1;
  int i = std::stoi(token);
  if (not std::getline(iss, token)) throw -1;
  int j = std::stoi(token);
  for (unsigned k=0; k<_legalActions.size(); k++)
   if (_legalActions[k]->GetY() == i and _legalActions[k]->GetZ() == j)
    return k;
 }
 catch (...) {
  std::cout << "failed to parse action" << std::endl;
 }
 return -1;
}


int Samegame::State::getCurrentPlayerColor() const {
 return _board.getCurrentColor();
}
*/

