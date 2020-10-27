/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// testing push
#include "yinsh.h"
#include <signal.h>
#include <unistd.h>
// #include <iostream>

// PAKKA GALTI
// FIX LATER
// premature optimisation is the source of all evil
// relisting the types for reference
// std::vector<float> _features;  // neural network input
// std::vector<std::shared_ptr<_Action>> _NewlegalActions;
// DNU: shared pointer

// draw ka rule is different- no of rings pe aa jaati hai baat
// namespace Yinsh{

uint64_t StateForYinsh::hash_table[5][BOARD_X][BOARD_Y] = {0};
std::once_flag StateForYinsh::table_flag = once_flag();

string StateForYinsh::stateDescription(void) const {
  return "empty_state";
}
string StateForYinsh::actionsDescription(void) const {
  return "empty_act";
}

char StateForYinsh::map_piece_to_char(int p) const {
  // printf("enter:    map_piece_to_char\n");
  char ans;
  if (p == (int)(piece::invalid))
    ans = ' ';
  else if (p == (int)(piece::p0_marker))
    ans = 'a';
  else if (p == (int)(piece::p0_ring))
    ans = 'A';
  else if (p == (int)(piece::p1_marker))
    ans = 'b';
  else if (p == (int)(piece::p1_ring))
    ans = 'B';
  else if (p == (int)(piece::empty))
    ans = '.';
  else
    ans = '!';
  // printf("leave:    map_piece_to_char\n");
  return ans;
}

void StateForYinsh::set_vars() {
  // printf("enter:    set_vars\n");

  // my_rings starts with p0
  // cout<<"player is "<<player<<endl;
  if (_status == GameStatus::player0Turn)
    player = 0;
  else if (_status == GameStatus::player1Turn)
    player = 1;
  else {
    // means that either the state is someone's win or tie , so it shouldnt have
    // entered this func
    // cout<<"Game already over"<<endl;//PE
  }
  // 2,3,4,5, m, r, m, r
  my_marker = 2 * player + 2;
  my_ring = 2 * player + 3;
  opp_marker = 6 - my_marker;  // PE
  opp_ring = 8 - my_ring;

  // player = 1;
  // if (player==0){
  //   my_rings = &rings[0];
  //   opp_rings = &rings[1];
  // }
  // else{
  //   my_rings = &rings[1];
  //   opp_rings = &rings[0];
  // }
  // printf("leave:    set_vars\n");
}
void StateForYinsh::fill_hash_table() {
  std::random_device rd;
  std::default_random_engine generator(rd());
  generator.seed(0);
  std::uniform_int_distribution<long long unsigned> distribution(
      0, 0xFFFFFFFFFFFFFFFF);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < BOARD_X; j++) {
      for (int k = 0; k < BOARD_Y; k++) {
        hash_table[i][j][k] = (uint64_t)(distribution(generator));
      }
    }
  }
}
void StateForYinsh::Initialize() {
  // printf("enter:    Initialize\n");

  _moves.clear();  // DNU
  _status = GameStatus::player0Turn;
  _featSize[0] = NUM_PIECES;
  _featSize[1] = BOARD_X;
  _featSize[2] = BOARD_Y;

  _actionSize[0] = NUM_ACTIONS;
  _actionSize[1] = BOARD_X;
  _actionSize[2] = BOARD_Y;

  _hash = 0ULL;  // DNU
  _status = GameStatus::player0Turn;
  _features.clear();
  _features.resize(_featSize[0] * _featSize[1] * _featSize[2]);

  std::call_once(table_flag, fill_hash_table);
  // initGame();
  // hard code your init yahan
  // TODO
  // all empty pehle
  for (int i = 0; i < (BOARD_X + 2); i++) {
    for (int j = 0; j < (BOARD_Y + 2); j++) {
      board[i][j] = (int)(piece::empty);
      // mark invalid
      if (i == 0 || j == 0 || i == 12 || j == 12 || (i - j) >= 6 ||
          (j - i) >= 6) {
        board[i][j] = (int)(piece::invalid);
      }
    }
  }
  // corners of my hexagon
  board[1][1] = (int)(piece::invalid);
  board[1][6] = (int)(piece::invalid);
  board[6][11] = (int)(piece::invalid);
  board[11][11] = (int)(piece::invalid);
  board[6][1] = (int)(piece::invalid);
  board[11][6] = (int)(piece::invalid);

  // rings[0].clear();
  // // printf("LOOK HERE
  // _____________________________________%d\n",rings[0].size() );
  // rings[1].clear();
  vector<tuple<int, int>> temp1, temp2;
  // printf("num of rings are %d\n",rings.size() );
  // sleep(10);
  rings.clear();
  // printf("num of rings are %d\n",rings.size() );
  rings.push_back(temp1);
  rings.push_back(temp2);

  // printf("LOOK HERE
  // _____________________________________%d\n",rings[1].size() );
  initial_fill = 0;
  places_filled = 0;
  // rings[0].resize(5);
  // rings[1].resize(5);
  still_have_to_remove_ring = false;
  still_have_to_remove_marker = false;
  free_lunch = false;
  // ended = false;

  set_vars();
  findActions();
  findFeatures();
  fillFullFeatures();
  // printf("leave:    Initialize\n");
}
bool StateForYinsh::ended() {
  // printf("enter:    ended\n");

  if (_status == GameStatus::player0Win || _status == GameStatus::player1Win ||
      _status == GameStatus::tie) {
    // printf("leave:    ended\n");
    return true;
  }
  // printf("leave:    ended\n");
  return false;
}

void StateForYinsh::printCurrentBoard() const {
  // printf("enter:    printCurrentBoard\n");
  for (int j = 0; j < (BOARD_Y + 2); j++) {
    for (int i = 0; i < (BOARD_X + 2); i++) {
      cout << map_piece_to_char(board[i][BOARD_Y + 1 - j]) << " ";
    }
    cout << endl;
  }
  cout << endl;
  // printf("leave:    printCurrentBoard\n");
}
tuple<int, int> StateForYinsh::find_first_invalid(int x,
                                                  int y,
                                                  int d0,
                                                  int d1) {
  // printf("enter:    find_first_invalid\n");

  /// PENDING, shayad zaroorat nhi
  int ex_x, ex_y;
  int i = d0;
  int j = d1;
  // PE- assumes that x and y are non extremes
  for (int c = 0; c < 13; c++) {
    if (board[x + c * i][y + c * j] == (int)(piece::invalid)) {
      ex_x = x + c * i;
      ex_y = y + c * j;
      break;
    }
    if (c == 10) {
      cout << "Debug here" << endl;
    }
  }
  // printf("leave:    find_first_invalid\n");
  return make_tuple(ex_x, ex_y);
}
vector<int> StateForYinsh::find_first_5_for_specific_pt(int x, int y) {
  // printf("enter:    find_first_5_for_specific_pt\n");

  // returns the start point and direction
  int start_x, start_y;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if ((i + j) != 0) {
        // my three directions to check
        tuple<int, int> temp = find_first_invalid(x, y, -i, -j);
        // since x,y is included it cant be invlaid and hence my func should
        // work just fine
        int count = 0;
        int type;
        start_x = get<0>(temp);
        start_y = get<1>(temp);

        // PE kis extreme se start kar raha hai
        for (int k = 1; k < 13; k++) {
          start_x =
              start_x + i;  // one step in the direction with each iteration
          start_y = start_y + j;
          type = board[start_x][start_y];
          if (type == (int)(piece::invalid)) {
            break;
          } else if (type == my_marker) {
            count++;
          } else {
            count = 0;
          }
          if (count == 5) {
            vector<int> answer;
            answer.push_back(start_x - 4 * i);
            answer.push_back(start_y - 4 * j);
            answer.push_back(i);
            answer.push_back(j);
            // printf("leave:    find_first_5_for_specific_pt\n");
            return answer;
          }
        }
      }
    }
  }
  vector<int> wrong;
  printf("Wrong answer in find_first_5_for_specific_pt\n");
  printf("x: %d and y: %d\n", x, y);
  printCurrentBoard();
  raise(SIGSEGV);
  // printf("leave:    find_first_5_for_specific_pt\n");
  return wrong;
}
tuple<int, int> StateForYinsh::map_num_to_direction(int n) {
  // printf("enter:    map_num_to_direction\n");

  // if else
  // PENDING
  tuple<int, int> ans;
  if (n == 0)
    ans = make_tuple(0, 1);
  else if (n == 1)
    ans = make_tuple(1, 1);
  else if (n == 2)
    ans = make_tuple(1, 0);
  else if (n == 3)
    ans = make_tuple(0, -1);
  else if (n == 4)
    ans = make_tuple(-1, -1);
  else if (n == 5)
    ans = make_tuple(-1, 0);
  else {
    cout << "invalid number for direciton" << endl;
    // return NULL;
    ans = make_tuple(-7, -7);
  }
  // printf("leave:    map_num_to_direction\n");
  return ans;
}
int StateForYinsh::map_direction_to_num(int i, int j) {
  // printf("enter:    map_direction_to_num\n");
  int ans;
  if (i == 0 and j == 1)
    ans = 0;
  else if (i == 1 and j == 1)
    ans = 1;
  else if (i == 1 and j == 0)
    ans = 2;
  else if (i == 0 and j == -1)
    ans = 3;
  else if (i == -1 and j == -1)
    ans = 4;
  else if (i == -1 and j == 0)
    ans = 5;
  else {
    cout << "invalid direciton" << endl;
    ans = -8;
  }
  // printf("leave:    map_direction_to_num\n");
  return ans;
}
vector<vector<int>> StateForYinsh::find_all_5s(bool my) {
  // printf("enter:    find_all_5s\n");

  vector<vector<int>> all_5s;  // empty abhi
  // 3 baaar likhlo same cheez kya dikat hai
  // PENDING
  int local_marker;
  if (my) {
    local_marker = my_marker;
    // local_ring = my_ring;
  } else {
    local_marker = opp_marker;
    // local_ring = opp_ring;
  }
  for (int d_x = 0; d_x < 2; d_x++) {
    for (int d_y = 0; d_y < 2; d_y++) {
      if ((d_x + d_y) != 0) {
        // for all axes
        // all pts on x axis and y
        int x, y;
        // bool start = false;
        int count = 0;
        for (int types = 0; types < 2; types++) {
          for (int pt = 0; pt < 13; pt++) {
            if (types == 0) {
              x = 0;
              y = pt;
            } else {
              if (pt == 0)
                break;
              x = pt;
              y = 0;
            }
            // start = false;
            count = 0;
            int poi;
            for (int jump = 0; jump < 13; jump++) {
              if (x + jump * d_x > 12 || y + jump * d_y > 12)
                break;
              // this will definitely go out of bounds
              // %13 lagaden yahan?
              // PAKKA GALTI //PE
              poi = board[(x + jump * d_x)][(y + jump * d_y)];
              if (poi == local_marker)
                count++;
              else
                count = 0;
              if (count >= 5) {
                // store
                vector<int> temporary_vec;
                temporary_vec.push_back((x + (jump - 4) * d_x));  // start_x
                temporary_vec.push_back((y + (jump - 4) * d_y));  // start_y
                temporary_vec.push_back(d_x);                     // d_x
                temporary_vec.push_back(d_y);                     // d_y
                all_5s.push_back(temporary_vec);
              }
            }
          }
        }
      }
    }
  }
  // printf("leave:    find_all_5s\n");
  return all_5s;
}
// vector<vector<int>> StateForYinsh::find_all_5s(bool my){
//   printf("enter:    find_all_5s\n");

//   vector<vector<int>> all_5s; //empty abhi
//   //3 baaar likhlo same cheez kya dikat hai
//   //PENDING
//   int local_marker;
//   if(my){
//     local_marker = my_marker;
//     // local_ring = my_ring;
//   }
//   else{
//     local_marker = opp_marker;
//     // local_ring = opp_ring;
//   }
//   for(int d_x = 0; d_x < 2; d_x ++){
//     for(int d_y = 0; d_y <2; d_y ++){
//       if((d_x + d_y) != 0){
//         //for all axes
//         //all pts on x axis and y
//         int x, y;
//         // bool start = false;
//         int count = 0;
//         for(int types = 0 ; types<2; types ++){
//           for(int pt = 0; pt<13; pt++){
//             if(types == 0){
//               x = 0;
//               y = pt;
//             }
//             else{
//               x = pt;
//               y = 0;
//             }
//             // start = false;
//             count = 0;
//             int poi;
//             for(int jump = 0; jump<13;
// jump++){
//               //this will definitely go out of
// bounds
//               // %13 lagaden yahan?
//               //PAKKA GALTI //PE
//               poi = board[(x + jump*d_x)%13][(y +
// jump*d_y)%13];               if(poi == local_marker) count++; else count = 0;
// if (count
// >=5){
//                 //store
//                 vector<int>
// temporary_vec;                 temporary_vec.push_back((x +
// (jump-4)*d_x)%13);//start_x
//                 temporary_vec.push_back((y +
// (jump-4)*d_y)%13);//start_y temporary_vec.push_back(d_x);//d_x
//                 temporary_vec.push_back(d_y);//d_y
//                 all_5s.push_back(temporary_vec);
//               }
//             }
//           }
//         }
//       }
//     }
//   }
//   printf("leave:    find_all_5s\n");
//   return all_5s;
// }
void StateForYinsh::ApplyAction(const _Action& action) {
  // printf("enter:    ApplyAction\n");

  // fetch args
  // for every update manage hash
  set_vars();
  // printCurrentBoard();
  // printf("following move was played by player %d\n", player);
  int action_num = action.GetX();
  int x = action.GetY() + 1;  // PE
  int y = action.GetZ() + 1;  // PE
  // printf("no of rings left %d\n", ((rings[player]).size()));
  // decide player_number
  // set_vars();
  // since _status is a protected variable StateForYinsh being a subclass should
  // be able to access

  // depending on the action num do different things
  // till 2:30
  // cout<<"for p0";
  // for(int i=0; i<(rings[0]).size(); i++){
  // printf("(%d, %d)  ", get<0>((rings[0])[i]),get<1>((rings[0])[i]));
  // }
  // cout<<endl;
  // cout<<"for p1";
  // for(int i=0; i<(rings[1]).size(); i++){
  // printf("(%d, %d)  ", get<0>((rings[1])[i]),get<1>((rings[1])[i]));
  // }
  // cout<<endl;

  // remember to maintain all state variables
  // cout<<"on enter"<<endl;
  // printCurrentBoard();
  if (action_num == 56) {
    // if(initial_fill==0){
    //   // rings[0].clear();
    //   // rings[1].clear();
    // }
    // simply place the ring on the board
    //
    // cout<<"debug"<<endl;
    _hash ^= hash_table[board[x][y] - 1][x - 1][y - 1];
    board[x][y] = my_ring;
    _hash ^= hash_table[board[x][y] - 1][x - 1][y - 1];

    // assignment
    // _hash ^= hash_table[][][];
    tuple<int, int> placed_ring(x, y);
    // if(player == 0) rings[0].push_back(placed_ring);
    // else rings[1].push_back(placed_ring);
    // cout<<"placed at "<<x<<" "<<y<<endl;
    // printf("%p\n",my_rings );

    if (rings[player].size() >= 5) {
      printf("WRONG NUM OF RINGS\n");
      raise(SIGSEGV);
    }
    (rings[player]).push_back(placed_ring);
    // cout<<"debug"<<endl;

    initial_fill++;
    places_filled++;

    // flip turn
    if (player == 0)
      _status = GameStatus::player1Turn;
    else
      _status = GameStatus::player0Turn;
  } else if (action_num == 57) {
    // ring selection for removal
    if (board[x][y] != my_ring) {
      cout << "trying to remove something which is not my ring" << endl;
      raise(SIGSEGV);
    }
    _hash ^= hash_table[board[x][y] - 1][x - 1][y - 1];
    board[x][y] = (int)(piece::empty);
    _hash ^= hash_table[board[x][y] - 1][x - 1][y - 1];

    // remove it from my vector_list too
    // cout<<"ring removed from "<<x<<" "<<y<<endl;
    for (int i = 0; i < (int)rings[player].size(); i++) {
      int a, b;
      a = get<0>((rings[player])[i]);
      b = get<1>((rings[player])[i]);
      if (a == x && b == y)
        (rings[player]).erase((rings[player]).begin() + i);
    }

    still_have_to_remove_ring = false;
    still_have_to_remove_marker = true;
    places_filled--;
    // dont change turn

  } else if (action_num == 58) {
    // sequence selection
    // find the 5 and then delete
    // for all directions '
    // cout<<"LOOK HERE"<<endl;
    vector<int> temp;
    // temp = find_first_5_for_specific_pt(x,y);
    temp = find_first_5_for_specific_pt(x, y);
    // cout<<"hey"<<endl;
    if (temp.size() == 1) {
      cout << "couldnt find 5, shouldnt have happened" << endl;
    }
    int start_x = temp[0];
    int start_y = temp[1];
    int dir_x = temp[2];
    int dir_y = temp[3];
    // vector<vector<int>> temp;
    // // temp = find_first_5_for_specific_pt(x,y);
    // temp = find_all_5s(true);
    // if (temp.size() == 0 ){
    //   cout<<"couldnt find 5, shouldnt have happened"<<endl;
    // }
    // int start_x = temp[0][0];
    // int start_y = temp[0][1];
    // int dir_x = temp[0][2];
    // int dir_y = temp[0][3];

    // cout<<"sequence_removed, starting and ending were-";
    // printf("(%d ,%d) and (%d, %d) ",start_x,start_y, start_x + 4*dir_x,
    // start_y + 4*dir_y);

    for (int i = 0; i < 5; i++) {
      int x1, y1;
      x1 = start_x + i * dir_x;
      y1 = start_y + i * dir_y;
      _hash ^= hash_table[board[x1][y1] - 1][x1 - 1][y1 - 1];
      board[x1][y1] = (int)(piece::empty);
      _hash ^= hash_table[board[x1][y1] - 1][x1 - 1][y1 - 1];
    }
    places_filled -= 5;

    still_have_to_remove_marker = false;
    if ((rings[player]).size() == 2) {
      // GAME OVER
      // cout<<"GAME OVER"<<endl;
      // cout<<"player number "<<player<<" wins"<<endl;
      if (player == 0)
        _status = GameStatus::player0Win;
      else
        _status = GameStatus::player1Win;
      // DNU_ASK_SOMEONE
      // rings[0].clear();
      // rings[1].clear();
      // raise(SIGSEGV);
      // PROB in this route
      fillFullFeatures();
      // printf("leave:    ApplyAction\n");
      return;
      // NO DRAWS IN THIS GAME UNLESS AND UNTIL you fill all the board
    }
    // check for more sequences.
    vector<vector<int>> temp2 = find_all_5s(true);
    if (temp2.size() == 0) {
      // change turn
      if (!free_lunch) {
        if (player == 0)
          _status = GameStatus::player1Turn;
        else
          _status = GameStatus::player0Turn;
      } else {
        // continue you turn now
        // cout<<"FREE"<<endl;
        free_lunch = false;
      }

    } else {
      still_have_to_remove_ring = true;
      still_have_to_remove_marker = true;
    }
  } else if (action_num >= 0 && action_num < 56) {
    // normal movement
    // dont check for legality

    int direction = action_num / 9;
    int jump = action_num % 9 + 1;

    tuple<int, int> dir = map_num_to_direction(direction);
    int d_x = get<0>(dir);
    int d_y = get<1>(dir);
    // ismei invert kardo bas
    // ring at x,y
    // cout<<"ring pos, direction and jump were-";
    // printf("(%d ,%d), (%d, %d) and %d respectively\n",x,y, d_x, d_y, jump);
    if (board[x][y] == my_ring) {
      _hash ^= hash_table[board[x][y] - 1][x - 1][y - 1];
      board[x][y] = my_marker;
      _hash ^= hash_table[board[x][y] - 1][x - 1][y - 1];
    } else {
      cout << "trying to move non_ring " << x << " " << y << ", actually found "
           << board[x][y] << endl;
      raise(SIGSEGV);
    }

    int x1, y1;
    for (int i = 1; i < jump; i++) {
      int found = board[x + (i)*d_x][y + (i)*d_y];

      if (found == (int)(piece::empty)) {
        // do nothing
      } else if (found == my_marker || found == opp_marker) {
        // flip it

        x1 = x + (i)*d_x;
        y1 = y + (i)*d_y;
        _hash ^= hash_table[board[x1][y1] - 1][x1 - 1][y1 - 1];
        board[x1][y1] = 6 - found;
        _hash ^= hash_table[board[x1][y1] - 1][x1 - 1][y1 - 1];
      } else {
        cout << "invalid move, dude" << endl;
      }
    }

    x1 = x + jump * d_x;
    y1 = y + jump * d_y;
    _hash ^= hash_table[board[x1][y1] - 1][x1 - 1][y1 - 1];
    board[x1][y1] = my_ring;
    _hash ^= hash_table[board[x1][y1] - 1][x1 - 1][y1 - 1];
    // update my_rings
    for (int i = 0; i < (int)rings[player].size(); i++) {
      int a, b;
      a = get<0>((rings[player])[i]);
      b = get<1>((rings[player])[i]);
      if (a == x && b == y)
        (rings[player]).erase((rings[player]).begin() + i);
    }
    (rings[player]).push_back(make_tuple(x + jump * d_x, y + jump * d_y));

    // check for 5
    places_filled++;
    vector<vector<int>> all_5s = find_all_5s(true);
    vector<vector<int>> all_5s_other = find_all_5s(false);
    // cout<<"places filled are "<<places_filled<<endl;
    if (all_5s.size() == 0) {
      // cout<<"hi1"<<endl;
      if (all_5s_other.size() == 0) {
        // continue
        // cout<<"hi3"<<endl;
        if (places_filled == 85) {
          if (rings[0].size() < rings[1].size())
            _status = GameStatus::player0Win;
          else if (rings[0].size() > rings[1].size())
            _status = GameStatus::player1Win;
          else
            _status = GameStatus::tie;
          // rings[0].clear();
          // rings[1].clear();
          // break;
          // raise(SIGSEGV);
          fillFullFeatures();
          // printf("leave:    ApplyAction\n");
          return;
        } else {
          // CHANGE TURN
          if (player == 0)
            _status = GameStatus::player1Turn;
          else
            _status = GameStatus::player0Turn;
        }
      } else {
        // cout<<"hi4"<<endl;
        // made a mistake
        // cout<<"supposed free seqs for other"<<endl;
        vector<int> t = all_5s_other[0];

        // cout<<"sequence_removed, starting and ending were-";
        // printf("(%d ,%d) and (%d, %d) ",t[0],t[1], t[0] + 4*t[2], t[1] +
        // 4*t[3]);
        free_lunch = true;
        still_have_to_remove_ring = true;
        still_have_to_remove_marker = true;
        if (player == 0)
          _status = GameStatus::player1Turn;
        else
          _status = GameStatus::player0Turn;
      }

    } else {
      // cout<<"hi2"<<endl;
      // cout<<"found my sequences"<<endl;
      vector<int> t = all_5s[0];

      // cout<<"sequence_removed, starting and ending were-";
      // printf("(%d ,%d)/ and (%d, %d) ",t[0],t[1], t[0] + 4*t[2], t[1] +
      // 4*t[3]);

      still_have_to_remove_ring = true;
      still_have_to_remove_marker = true;
      // dont change turn
    }
    // you have to set the flags in appply to
  } else {
    cout << "You fucked up something" << endl;
  }

  // cout<<"before calling others"<<endl;
  // printCurrentBoard();
  // cout<<"debug3"<<endl;

  set_vars();
  // cout<<"debug4"<<endl;
  // cout<<"after set vars"<<endl;
  // printCurrentBoard();

  findActions();
  // cout<<"debug5"<<endl;
  // cout<<"after finding actions"<<endl;
  // printCurrentBoard();

  findFeatures();
  // cout<<"debug6"<<endl;
  // cout<<"after finding features"<<endl;
  // printCurrentBoard();
  fillFullFeatures();
  // check_termination
  // update turns
  // printf("leave:    ApplyAction\n");
}
// void StateForYinsh::findActions(){

// }
void StateForYinsh::findActions() {
  //
  // printf("enter:    findActions\n");

  // cout<<"frequency_check"<<endl;
  clearActions();

  if (initial_fill < 10) {
    // you can only try to fill the empty spaces

    // main line
    // cout<<"is it here"<<endl;
    for (int i = 0; i < 13; i++) {
      for (int j = 0; j < 13; j++) {
        if (board[i][j] == (int)(piece::empty)) {
          addAction(56, i - 1, j - 1);
        }
      }
    }
  } else if (still_have_to_remove_ring) {
    for (int i = 0; i < (int)rings[player].size(); i++) {
      addAction(
          57, get<0>((rings[player])[i]) - 1, get<1>((rings[player])[i]) - 1);
    }
  } else if (still_have_to_remove_marker) {
    vector<vector<int>> all_5s = find_all_5s(true);

    vector<vector<int>> till_now;
    int m_x, m_y, d_x, d_y, k, j;
    bool matched = false;
    //  printf("num of 5s are %d\n", all_5s.size());

    for (int i = 0; i < (int)all_5s.size(); i++) {
      // FIX LATER
      // i just put the first point to make this quick
      // _NewlegalActions.push_back(make_shared<ActionForYinsh>(
      //     58, (all_5s[i][0]) - 1, (all_5s[i][1]) - 1,
      //     _NewlegalActions.size()));
      m_x = all_5s[i][0];
      m_y = all_5s[i][1];
      d_x = all_5s[i][2];
      d_y = all_5s[i][3];
      //  printf("mx, my, dx, dy are %d %d %d %d \n",m_x, m_y, d_x, d_y);

      for (j = 0; j < 5; j++) {
        matched = false;
        // for all 5 markers
        for (k = 0; k < (int)till_now.size(); k++) {
          if (((till_now[k][0] == m_x + d_x * j) &&
               (till_now[k][1] == m_y + d_y * j)))
            matched = true;
        }
        if (!matched)
          break;
      }
      // fill the chosen marker
      //  printf("chose %d, %d \n", (m_x + d_x*j), (m_y + d_y*j));

      addAction(58, (m_x + d_x * j) - 1, (m_y + d_y * j) - 1);
      vector<int> temp_pt;
      temp_pt.push_back(m_x + d_x * j);
      temp_pt.push_back(m_y + d_y * j);
      till_now.push_back(temp_pt);
    }

  } else {
    // normal move
    // for all rings
    // go in all directions
    // cout<<"enter"<<endl;
    // cout<<rings[0].size()<<endl;
    // cout<<rings[1].size()<<endl;
    int x, y, start_x, start_y;
    bool start = false;
    for (int k = 0; k < (int)rings[player].size(); k++) {

      // cout<<"before access"<<endl;
      start_x = get<0>((rings[player])[k]);
      start_y = get<1>((rings[player])[k]);
      // cout<<"ring "<<k<<" at "<<start_x<<" "<<start_y<<endl;
      for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
          if ((i + j) != 0) {
            start = false;
            for (int jump = 1; jump < 13; jump++) {
              x = start_x + jump * i;
              y = start_y + jump * j;
              if (!start) {
                if (board[x][y] == (int)(piece::empty)) {
                  // possible move
                  addAction((jump - 1) + 9 * map_direction_to_num(i, j),
                            start_x - 1, start_y - 1);
                } else {
                  start = true;
                }
              }
              // yahan else nhi ayega
              if (start) {
                if (board[x][y] == my_marker || board[x][y] == opp_marker) {
                  // do nothing
                  // continur
                } else if (board[x][y] == (int)(piece::empty)) {
                  addAction((jump - 1) + 9 * map_direction_to_num(i, j),
                            start_x - 1, start_y - 1);
                  break;
                } else {
                  // must be invalid or someones ring

                  break;
                }
              }
            }
          }
        }
      }
    }
  }
  if (_legalActions.size() == 0) {
    // cout<<"ZERO MOVES TO PLAY"<<endl;
    // printf("player is %d" , player);
    // printCurrentBoard();
    if (rings[0].size() < rings[1].size())
      _status = GameStatus::player0Win;
    else if (rings[0].size() > rings[1].size())
      _status = GameStatus::player1Win;
    else
      _status = GameStatus::tie;
    fillFullFeatures();
    // printf("leave:    findActions\n");
    return;
  }
  // printf("leave:    findActions\n");
}

unique_ptr<core::State> StateForYinsh::clone_() const {
  // printf("enter:    clone_\n");
  unique_ptr<core::State> temp = std::make_unique<StateForYinsh>(*this);
  // printf("leave:    clone_\n");
  return temp;
}

void StateForYinsh::DoGoodAction() {
  // printf("enter:    DoGoodAction\n");

  // return DoRandomAction();
  int i = rand() % _legalActions.size();
  if (_legalActions.size() == 0)
    cout << "NO MOVES" << endl;
  // printf("No of legal moves are %d\n",_NewlegalActions.size() );
  _Action a = _legalActions[i];
  ApplyAction(a);
  // printf("leave:    DoGoodAction\n");
}
void StateForYinsh::findFeatures() {
  // printf("enter:    findFeatures\n");

  // yeh toh chill function hai
  // simply just map the board to the 3 dimensions discussed
  // NUM_PIECES, BOARD_X, BOARD_Y;
  // cout<<"enter"  <<endl;
  for (int p = 0; p < NUM_PIECES; p++) {
    for (int j = 0; j < BOARD_Y; j++) {
      for (int i = 0; i < BOARD_X; i++) {
        if (p == 0 && board[i + 1][j + 1] == (int)(piece::empty))
          _features[p * (BOARD_X * BOARD_Y) + j * BOARD_X + i] = 1;
        else if (p == 1 && board[i + 1][j + 1] == my_ring)
          _features[p * (BOARD_X * BOARD_Y) + j * BOARD_X + i] = 1;
        else if (p == 2 && board[i + 1][j + 1] == my_marker)
          _features[p * (BOARD_X * BOARD_Y) + j * BOARD_X + i] = 1;
        else if (p == 3 && board[i + 1][j + 1] == opp_ring)
          _features[p * (BOARD_X * BOARD_Y) + j * BOARD_X + i] = 1;
        else if (p == 4 && board[i + 1][j + 1] == opp_marker)
          _features[p * (BOARD_X * BOARD_Y) + j * BOARD_X + i] = 1;
        else
          _features[p * (BOARD_X * BOARD_Y) + j * BOARD_X + i] = 0;
      }
    }
  }
  // cout<<"leave"  <<endl;
  // printf("leave:    findFeatures\n");
}
// string vala and good action
// void StateForYinsh:

// }
// int main(){
//   cout<<"hello world"<<endl;
//   double a = time(NULL);
//   srand(a);
//   StateForYinsh g1(0);
//   // g1.printCurrentBoard();
//   g1.Initialize();
//   g1.printCurrentBoard();

//   //to debug- print num of legal moves at every turn and count
//   for(int n = 1; n<150; n++){
//     cout<<"move number "<<n<<endl;
//     g1.DoGoodAction();
//     g1.printCurrentBoard();
//     if(g1.ended()){
//       break;
//     }

//     // cout<<g1.initial_fill<<endl;
//     // if(g1._status == GameStatus::player0Win || g1._status ==
// GameStatus::player1Win ||g1._status == GameStatus::playerTie ){
//     //   break;
//     // }
//     //floating point error because of division by zero legal moves
//   }
//   // printf("Seed was %lf\n",a );

// }
