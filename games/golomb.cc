
#include "golomb.h"
#include <cassert>
#include <cmath>

Board::Board(int max) :
    _max(max) {
        _order=0;
        _length = 0;
        _solution.resize(_max, 0);
        _distanceList.resize(_max, 0);
        _legalMoves.resize(_max, 1);
        _solution[0] = 1;
        _legalMoves[0] = 0;
        _score = 0;
    }

void Board::print() const{
    std::cout << "Solution (" << _order << ", " << _length << ")\n";
    for (unsigned i=0; i<_solution.size(); i++)
        if (_solution[i])
            std::cout << i << " ";
    std::cout << "\nscore: " << _score << std::endl;
}

void Board::printAll() const{
    std::cout << "Solution (" << _order << ", " << _length << ")\n";
    for (unsigned i=0; i<_solution.size(); i++)
        if (_solution[i])
            std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "Distances: " << std::endl;
    for (int i=1; i<_max; i++)
        if (_distanceList[i])
            std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "Legals: " << std::endl;
    for (int i=1; i<_max; i++)
        if (_legalMoves[i] == 1)
            std::cout << i << " ";
    std::cout << "\nscore: " << _score << std::endl;
}

void Board::applyAction(int number) {
    assert(number < _max);
    assert(_solution[number] == 0);
    assert(_legalMoves[number]);
    _solution[number] = 1;
    updateInternalState(number);
}

void Board::updateInternalState(int number) {
    bool scoreOk = true;
    for (int i=0; i<_max; i++) {
        if (_solution[i]) {
            // update of all distances
            _distanceList[std::abs(number-i)] = 1;
            // update of legalMoves
            for (int j=0; j<_max; j++) {
                if (_distanceList[j]) {
                    if (i+j < _max)
                        _legalMoves[i+j] = 0;
                    if (i-j >= 0)
                        _legalMoves[i-j] = 0;
                    if (j-i >= 0)
                        _legalMoves[j-i] = 0;
		    if (not ((i+j) % 2)) {
                        _legalMoves[(i+j)/2] = 0;

                }
            }
        }
        if (scoreOk && !_distanceList[i]) {
            scoreOk = false;
            _score = i-1;
        }
    }
    // update of _order
    _order++;

    // update of _length
    for (unsigned i=0; i<_solution.size(); i++)
        if (_solution[i])
            _length = i;    
}

std::vector<int> Board::legalMovesToVector() const {
    std::vector<int> legals;
    for (unsigned i=0; i<_legalMoves.size(); i++)
        if (_legalMoves[i])
            legals.push_back(i);    
    return legals;
}

bool Board::isTerminated() const {
    for (unsigned i=0; i<_legalMoves.size(); i++)
        if (_legalMoves[i])
            return false;
    return true;
}
