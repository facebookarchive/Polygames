/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author1: Lin Hsin-I
// - Github: https://github.com/free00000000000
// - Email:  410521233@gms.ndhu.edu.tw
// Facilitator: 邱顯棟 (Xiǎn-Dòng Qiū)
// - Github: https://github.com/YumJelly
// - Email:  yumjelly@gmail.com

#pragma once
#include "../core/state.h"
#include "shogi.h"
#include <vector>
#include <queue>
#include <sstream>

class ActionForDiceshogi : public _Action {
  public:
    ActionForDiceshogi(int x, int y, int p, size_t index) : _Action() {
        _loc[0] = p;
        _loc[1] = x;
        _loc[2] = y;
        _hash = (x + y * 5) * 19 + p;
        _i = (int)index;
    }
};

class StateForDiceshogi : public State, public Shogi {
  public:
    unsigned long long HashArray[2][10][Dx][Dy];
    unsigned long long HashArrayJail[20];
    unsigned long long HashTurn;
    unsigned long long hash;
    int length;
    short dice; // 0-5
    int repeat;
    std::queue<unsigned long long> situation;

    StateForDiceshogi(int seed) : State(seed), Shogi() {
        Initialize();
        _stochasticReset = true;
    }

    virtual void Initialize() override {
        _stochastic = true;
        _moves.clear();
        _hash = 0;
        _status = GameStatus::player0Turn;
        _featSize[0] = 225;
        _featSize[1] = Dy;
        _featSize[2] = Dx;
        _actionSize[0] = 19;  // 11 pieces + 8 promoted
        _actionSize[1] = Dy;
        _actionSize[2] = Dx;
        _features.clear();
        _features.resize(_featSize[0] * _featSize[1] * _featSize[2]);
        // setFeatures(false, false, false, 0, 0, false);

        gameInit();
        initHash();
        // printCurrentBoard();

        findFeature();
        findActions();
        // fixxx
        fillFullFeatures();
    }

    void gameInit() {
        chess.clear();
        chess.resize(2);
        for(int i=0; i<Dx; ++i)
            for (int j = 0; j < Dy; ++j)
                board[i][j] = Piece();

        for(int i=0; i<Dx; ++i) {
            board[i][0] = Piece(White, PieceType(i+1), false, Position(i, 0));
            chess[White].push_back(board[i][0]);
        }
        board[0][1] = Piece(White, PieceType::Pawn, false, Position(0, 1));
        chess[White].push_back(board[0][1]);

        for(int i=1; i<=Dx; ++i) {
            int x = Dx - i;
            board[x][4] = Piece(Black, PieceType(i), false, Position(x, 4));
            chess[Black].push_back(board[x][4]);
        }
        board[4][3] = Piece(Black, PieceType::Pawn, false, Position(4, 3));
        chess[Black].push_back(board[4][3]);

        if (forcedDice > 0) {
          assert(forcedDice > 0);
          assert(forcedDice < 7);
          dice = forcedDice -1 ;
          forcedDice = -1;
        } else {
          dice = _rng() % 6;
        }
        
        _hash = dice + 1; 
        hash = 0;
        length = 0;
        repeat = 0;
        situation.push(hash);

    }

    void initHash() {
        for(int a=0; a<2; ++a) 
            for(int b=0; b<10; ++b)
                for(int c=0; c<5; ++c)
                    for(int d=0; d<5; ++d) {
                        HashArray[a][b][c][d] = 0;
                        for(int k=0; k<64; ++k)
                            if ((_rng() / (RAND_MAX + 1.0)) > 0.5)
                                HashArray[a][b][c][d] |= (1ULL << k);
                    }
        for(int a=0; a<20; ++a) {
            for(int k=0; k<64; ++k)
                if ((_rng() / (RAND_MAX + 1.0)) > 0.5)
                    HashArrayJail[a] |= (1ULL << k);
        }

        HashTurn = 0;
        for (int k = 0; k < 64; k++)
            if ((_rng() / (RAND_MAX + 1.0)) > 0.5)
                HashTurn |= (1ULL << k);
    }

    void findFeature() {
        std::vector<float> old(_features);
        for (int i = 0; i < 5425; ++i)
            _features[i] = 0;
        // 0 ~ 500
        for (int i = 0; i < 25; ++i) {
        Piece p = board[i % 5][i / 5];
        if (p.color == White) {
            switch (p.type) {
            case PieceType::King:
            _features[i] = 1;
            break;

            case PieceType::Gold:
            case PieceType::Gold2:
            _features[25 + i] = 1;
            break;

            case PieceType::Silver:
            case PieceType::Silver2:
            if (p.promoted)
                _features[50 + i] = 1;
            else
                _features[75 + i] = 1;
            break;

            case PieceType::Bishop:
            case PieceType::Bishop2:
            if (p.promoted)
                _features[100 + i] = 1;
            else
                _features[125 + i] = 1;
            break;

            case PieceType::Rook:
            case PieceType::Rook2:
            if (p.promoted)
                _features[150 + i] = 1;
            else
                _features[175 + i] = 1;
            break;

            case PieceType::Pawn:
            case PieceType::Pawn2:
            if (p.promoted)
                _features[200 + i] = 1;
            else
                _features[225 + i] = 1;
            break;

            default:
            break;
            }
        } else {
            switch (p.type) {
            case PieceType::King:
            _features[250 + i] = 1;
            break;

            case PieceType::Gold:
            case PieceType::Gold2:
            _features[275 + i] = 1;
            break;

            case PieceType::Silver:
            case PieceType::Silver2:
            if (p.promoted)
                _features[300 + i] = 1;
            else
                _features[325 + i] = 1;
            break;

            case PieceType::Bishop:
            case PieceType::Bishop2:
            if (p.promoted)
                _features[350 + i] = 1;
            else
                _features[375 + i] = 1;
            break;

            case PieceType::Rook:
            case PieceType::Rook2:
            if (p.promoted)
                _features[400 + i] = 1;
            else
                _features[425 + i] = 1;
            break;

            case PieceType::Pawn:
            case PieceType::Pawn2:
            if (p.promoted)
                _features[450 + i] = 1;
            else
                _features[475 + i] = 1;
            break;

            default:
            break;
            }
        }
        }

        // 500 ~ 575
        switch (repeat) {
        case 1:
        std::fill(_features.begin() + 500, _features.begin() + 525, 1);
        break;
        case 5:
        std::fill(_features.begin() + 525, _features.begin() + 550, 1);
        break;
        case 9:
        std::fill(_features.begin() + 550, _features.begin() + 575, 1);
        break;
        default:
        break;
        }

        // prison w 575 ~ 625
        // prison b 625 ~ 675
        int tmp = 575;
        for (int i = 0; i < 2; ++i) {
        std::vector<Piece>::iterator it;
        for (it = chess[i].begin(); it != chess[i].end(); ++it) {
            if (!(*it).pos.on_board()) {
            switch ((*it).type) {
            case PieceType::Gold:
                std::fill(_features.begin() + tmp, _features.begin() + tmp + 5, 1);
                break;
            case PieceType::Silver:
                std::fill(
                    _features.begin() + tmp + 5, _features.begin() + tmp + 10, 1);
                break;
            case PieceType::Bishop:
                std::fill(
                    _features.begin() + tmp + 10, _features.begin() + tmp + 15, 1);
                break;
            case PieceType::Rook:
                std::fill(
                    _features.begin() + tmp + 15, _features.begin() + tmp + 20, 1);
                break;
            case PieceType::Pawn:
                std::fill(
                    _features.begin() + tmp + 20, _features.begin() + tmp + 25, 1);
                break;
            case PieceType::Gold2:
                std::fill(
                    _features.begin() + tmp + 25, _features.begin() + tmp + 30, 1);
                break;
            case PieceType::Silver2:
                std::fill(
                    _features.begin() + tmp + 30, _features.begin() + tmp + 35, 1);
                break;
            case PieceType::Bishop2:
                std::fill(
                    _features.begin() + tmp + 35, _features.begin() + tmp + 40, 1);
                break;
            case PieceType::Rook2:
                std::fill(
                    _features.begin() + tmp + 40, _features.begin() + tmp + 45, 1);
                break;
            case PieceType::Pawn2:
                std::fill(
                    _features.begin() + tmp + 45, _features.begin() + tmp + 50, 1);
                break;
            default:
                break;
            }
            }
        }
        tmp += 50;
        }

        // dice 675 ~ 700
        if (dice == 5)
          std::fill(_features.begin() + 675, _features.begin() + 700, 1);
        else
          std::fill(_features.begin() + dice * 5, _features.begin() + dice * 5 + 5, 1);

        // history 700 ~ 4900+700
        std::copy(old.begin(), old.begin() + 4900, _features.begin() + 700);

        // 5600 ~ 5625
        std::fill(_features.begin() + 5600, _features.end(), (int)_status);
    }

    void findActions() {
        std::vector<Move> moves;
        std::vector<Move> dice_moves;
        for(auto i : chess[(int)_status]) {
            legalMoves(i, moves);
        }

        // dice limit
        if (dice != 5) {
          for(auto m : moves)
            if (m.next.x == dice)
              dice_moves.push_back(m);
        }
        if (dice_moves.empty())
          dice_moves = moves;

        int i = 0;
        _legalActions.clear();
        for(auto m : dice_moves) {
            m.piece.promoted = m.promote;

            int x = m.next.x;
            int y = m.next.y;
            int z = type_to_z(m.piece);

            _legalActions.push_back(
                std::make_shared<ActionForDiceshogi>(x, y, z, _legalActions.size())
            );
            i++;
        }

    }

    virtual void printCurrentBoard() const override {
        std::cerr << stateDescription();
        // for(int i=0; i<2; ++i) {
        //     for(auto j : chess[i]) {
        //         fprintf(stderr, "(%c,%d) ", j.pos.x+'A', j.pos.y);
        //     }
        //     std::cerr << std::endl;
        // }
    }

    std::string print_chess(const int color) const {
        std::string str;
        if(color == White) str += "DiceWhite: ";
        else str += "DiceBlack: ";
        for(auto i : chess[color]) {
            if(!i.pos.on_board()) {
                str += '(';
                str += i.print();
                str += ')';
            }
            else str += i.print();
            str += ' ';
        }
        str += '\n';
        return str;
    }

    virtual std::string stateDescription() const override {
        std::string str;
        str += "   A| B| C| D| E\n";
        for(int i=Dy-1; i>=0; --i) {
            str += std::to_string(i+1) + ' ';
            for(int j=0; j<Dx; ++j) {
                if(j > 0) str += '|';
                str += board[j][i].print();
            }
            str += '\n';
        }
        str += print_chess(White);
        str += print_chess(Black);

        return str;
    }

    virtual std::string actionsDescription() override {
        std::stringstream ss;
        int i=0;
        for(auto action : _legalActions) {
            int z = (*action).GetX();
            int x = (*action).GetY();
            int y = (*action).GetZ();

            Piece p;
            p.type = z_to_type(z);
            p.promoted = z_promoted(z);
            p.color = (int)_status;

            for(auto i : chess[p.color])
                if(i.type == p.type) p.pos = i.pos;
           
            ss << p.print();
            char buff[53];
            sprintf(buff, " (%c, %c) to (%c, %c) ---%d\n", p.pos.x+'A', p.pos.y+'1', x+'A', y+'1', i++);
            ss << buff;
        }
        ss << "\nInput format: action index e.g. 0\n";
        return ss.str();
    }

    virtual std::string actionDescription(const _Action & action) const {
        std::stringstream ss;
        int z = action.GetX();
        int x = action.GetY();
        int y = action.GetZ();

        Piece p;
        p.type =  z_to_type(z);
        p.promoted = z_promoted(z);
        p.color = opponent((int)_status);
        
        for(auto i : chess[p.color])
          if(i.type == p.type) p.pos = i.pos;
    
        ss << p.print(); 
        char buff[21];
        sprintf(buff, " to (%c, %c)\n", x+'A', y+'1');
        ss << buff;

        return ss.str();
    }

    virtual std::unique_ptr<mcts::State> clone_() const override {
        return std::make_unique<StateForDiceshogi>(*this);
    }

    int getHashNum(Piece p) {
        int num = (int)p.type;
        if (num >= 7)
        num -= 5;
        if (p.promoted)
        num += 5;
        //   7 8 9 10 11
        // 1 2 3 4  5  6 | 7 8 9 10
        num -= 1;
        return num;
    }

    int getHashNumjail(Piece p) {
        // 0~19
        return (int)p.type - 2 + 10 * p.color;
    }

    void play(Move m) {
        // std::cerr << m.piece.print();
        // fprintf(stderr, " play (%c, %d) to (%c, %d)\n\n", m.piece.pos.x+'A', m.piece.pos.y, m.next.x+'A', m.next.y);
        m.piece.promoted |= m.promote;

        if (m.piece.pos.on_board()) {
            hash ^= HashArray[m.piece.color][getHashNum(m.piece)][m.piece.pos.x][m.piece.pos.y];
            // eat
            if (board[m.next.x][m.next.y].color != Empty) {
                int opp = opponent(m.piece.color);
                hash ^= HashArray[opp][getHashNum(board[m.next.x][m.next.y])][m.next.x][m.next.y];
                hash ^= HashArrayJail[getHashNumjail(m.piece)];

                Piece tmp(m.piece.color, new_type(board[m.next.x][m.next.y].type), false);
                chess[m.piece.color].push_back(tmp);
                
                std::vector<Piece>::iterator it;
                for (it = chess[opp].begin(); it != chess[opp].end(); ++it) {
                    if ((*it).type == board[m.next.x][m.next.y].type) {
                        chess[opp].erase(it);
                        break;
                    }
                }
    
            }

            std::vector<Piece>::iterator it;
            for(it = chess[m.piece.color].begin(); it != chess[m.piece.color].end(); ++it) {
                if((*it).type == m.piece.type) {
                    (*it).pos = m.next;
                    // decide promoted
                    if ((m.piece.color == White && m.next.y == Dy - 1) || (m.piece.color == Black && m.next.y == 0)) {
                        if (m.piece.promoted || (*it).type == PieceType::Pawn || (*it).type == PieceType::Pawn2)
                            (*it).promoted = true;
                    }
                    board[m.next.x][m.next.y] = (*it);
                    board[m.piece.pos.x][m.piece.pos.y] = Piece();
                    break;
                }
            }
        } else {  // Drop move 
            hash ^= HashArrayJail[getHashNumjail(m.piece)];
            std::vector<Piece>::iterator it;
            for(it = chess[m.piece.color].begin(); it != chess[m.piece.color].end(); ++it) {
                if((*it).type == m.piece.type) {
                    (*it).pos = m.next;
                    board[m.next.x][m.next.y] = (*it);
                    break;
                }
            }
        }
        hash ^= HashArray[m.piece.color][getHashNum(board[m.next.x][m.next.y])][m.next.x][m.next.y];
        hash ^= HashTurn;

        if (length < MaxPlayoutLength) {
            //rollout[length] = m;
            length++;
        } else {
            // set draw when the moves bigger than 1000
            _status = GameStatus::tie;
        }

        // find repeat
        if (hash != situation.front()) repeat = 0;
        else repeat += 1;
        // fprintf(stderr, "end play\n");
    }

    bool fourfold() {
        if (repeat < 9) return false;
        return true;
    }

    bool won(int color) {
        // fprintf(stderr, "won: ");
        // for(auto i : chess[opponent(color)]) {
        //     if(i.type == PieceType::King) {
        //         if(checkmate(i)) return true;
        //         break;
        //     }
        // }
        if(checkmate(opponent(color))) return true;
 
        if (fourfold() && opponent((int)_status) == opponent(color))
            return true;
        return false;
    }
    
    virtual void ApplyAction(const _Action& action) override {
        // fprintf(stderr, "\nApply Action %d\n", (int)_status);
        
        Move m;
        int z = action.GetX();
        int x = action.GetY();
        int y = action.GetZ();

        m.next.x = x;
        m.next.y = y;
        m.piece.type = z_to_type(z);
        m.promote = z_promoted(z);

        if(_status == GameStatus::player0Turn) {  // White to move
            m.piece.color = White;
            // find original position
            for(auto i : chess[White]) {
                if(i.type == m.piece.type)
                    m.piece.pos = i.pos;
            }

            play(m);
            // printCurrentBoard();
            // fprintf(stderr, "hash: %llu\n", hash);
            // fprintf(stderr, "repeat: %d\n", repeat);
            if ((GameStatus)_status == GameStatus::tie) {}
            else if (!won(White))
                _status = GameStatus::player1Turn;  // Black turn
            else
                _status = GameStatus::player0Win;  // White won
        } else {  // Black to move
            m.piece.color = Black;
            // find original position
            for(auto i : chess[Black]) {
                if(i.type == m.piece.type)
                    m.piece.pos = i.pos;
            }

            play(m);
            // printCurrentBoard();
            // fprintf(stderr, "hash: %llu\n", hash);
            // fprintf(stderr, "repeat: %d\n", repeat);

            if ((GameStatus)_status == GameStatus::tie)
                {}
            else if (!won(Black))
                _status = GameStatus::player0Turn;  // White turn
            else
                _status = GameStatus::player1Win;  // Black won
        }
        if(_status == GameStatus::player0Turn ||
           _status == GameStatus::player1Turn) {
            if (forcedDice >= 0) {
              assert(forcedDice > 0);
              assert(forcedDice < 7);
              dice = forcedDice - 1;
              forcedDice = -1;
            } else {
              dice = _rng() % 6;
            }
            _hash = dice + 1;
            findFeature();
            findActions();
            fillFullFeatures();

            if(situation.size() == 4) {
                situation.pop();
                situation.push(hash);
            } else 
                situation.push(hash);
        } else {
            _legalActions.clear();
            // if(_status == GameStatus::player0Win)
            //     fprintf(stderr, "white win\n");
            // else if(_status == GameStatus::player1Win)
            //     fprintf(stderr, "black win\n");
            // else fprintf(stderr, "tie\n");
        }
        // fprintf(stderr, "end apply action\n");
    }

    virtual void DoGoodAction() override {
        // int i;
        // printCurrentBoard();
        // std::cout << actionsDescription();
        // std::cin >> i;
        // _Action a = *(_legalActions[i].get());
        // ApplyAction(a);
        // std::cout << actionDescription(a);

        return DoRandomAction();
    }

};
