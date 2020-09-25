#pragma once
#include <iostream>
#include <vector>

namespace Golomb {

class Board {
    private:
        int _order;
        int _length;
        int _max;
        int _score;                 // score that represents optimal golomb numbers

        std::vector<int> _solution; // representation of a solution
        std::vector<int> _distanceList; // list of all distances
        std::vector<int> _legalMoves; // list of all legal moves
    
        void updateInternalState(int number);   // update of _distanceList
                                                // update of _order
                                                // update of _length
                                                // update of legalMoves

    public:
        int getMax() { return _max; }

        Board(int max);
        Board();
        void reset();
        std::vector<int> legalMovesToVector() const;
        void applyAction(int number);
        void print() const;
        void printAll() const;
        bool isTerminated() const;
        int getScore() const;
        std::vector<int> getSolution() const;
        std::vector<int> getDistanceList() const;
        std::vector<int> getLegalMoves() const;
};

}
