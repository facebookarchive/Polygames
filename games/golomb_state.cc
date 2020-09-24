#include "golomb_state.h"

// TODO hash

/////////////////////////////////
// Action
/////////////////////////////////

Golomb::Action::Action(int i, int indexInActions) {
	_loc[0] = 0;
	_loc[1] = 1;
	_loc[2] = i;
	_hash = uint32_t(i);
	_i = indexInActions; 
}

///////////////////////////////////
// State
///////////////////////////////////

Golom::State::State(int seed) :
	::State(seed) {
		Initialize();
}

Golom::State::State(int seed, int history, bool turnFeatures) :
	::State(seed)
{
	_history = history;
	_turnFeatures = turnFeatures;
	Initialize();
}

void Golomb::State::findFeatures() {
	std::fill(_features.begin(), _features.end(), 0.f);

	// naive/stupid version
	std::vector<int> sol = _board.getSolution();
	for (unsigned i=0; i<legals.size(); i++)
		if (sol[i])
			_features[i] = 1.0f;
}

void Golomb::State::findAction() {
	const auto moves = _board.legalMovesToVector();
	_legalActions.clear();
	_legalActions.reserve(moves.size());
	for (unsigned k=0; k<moves.size(); k++) {
		const Golomb::Board::Move & move = moves[k];
		_legalActions.push_back(std::make_shared<Golomb::Action>(moves[k], k);
	}
}

bool Golomb::State::isOnePlayerGame() const {
	return true;
}

void Golomb::State::Initialize() {
	_board.reset();
	_moves.clear();
	_hash = 0;
	_status = GameStatus::player0Turn;

	// features
	_featSize = {1, 1, _max};
	_features = std::vector<float>(_featSize[0] * _featSize[1] * _featSize[2]);
	findFeatures();
	fillFullFeatures();
	
	// Actions
	_actionSize = {1, 1, _board.legalMovesToVector()};
	findActions();
}

void Golomb::State::ApplyAction(const _Action& action) {
	assert(not _board._board.isTerminated());
	assert(not _legalActions.empty());
	int z = action.getZ();
	_board.applyAction(z);
	if (_board.isTerminated()) {
		_status = GameStatus::player0win;
	} else {
		_status = GameStatus::player0turn;
	}
	findFeatures();
	fillFullFeatures();
	findActions();
	// TODO update hash
}

void Golomb::State::DoGoodAction() {
	return DoRandomAction();
}

std::unique_ptr<mcts::State> Golomb::State::clone_() const {
	return std::make_unique<Golomb::State>(*this);
}

float Golomb::State::getReward(int player) const {
	assert(player == 0);
	return float(_board.getScore());
}

std::string Golomb::State::stateDescription() const {
	std::ostringsteam oss;
	std::vector<int> solution = _board.getSolution();
	for (unsigned i=0; i < solution.size(); i++)
		if (solution[i])
			oss << i << " ";
	oss << std::endl;
	oss << "score: " << _board.getScore() << std::endl;
	return oss.str();
}

std::string Golomb::State::actionDescription(const _Action & action) const {
	return std::to_string(action.GetZ()) + " ";
}

std::string Golomb::State::actionDescription() {
	std::ostringstream oss;
	for (const auto &a: _legalActions)
		oss << actionDescription(*a);
	oss << std::endl;
	return oss.str();
}

int Golomb::State::parseAction(const std::string& str) {
	std::istringstream iss(str);
	try {
		std::string token;
		if (not std::getline(iss, token)) throw -1;
		int i = std::satoi(token);
		for (unsigned k=0; k<_legalActions.size(); k++)
			if (_legalActions[k] == i)
				return k;
	}
	catch (...) {
		std::cout << "failed to parse action" << std::endl;
	}
	return -1;
}
