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
	// TODO
}

void Golomb::State::findAction() {
	// TODO
}

bool Golomb::State::isOnePlayerGame() const {
	return true;
}

void Golomb::State::Initialize() {
	// TODO
}

void Golomb::State::ApplyAction(const _Action& action) {
	// TODO
}

void Golomb::State::DoGoodAction() {
	return DoRandomAction();
}

std::unique_ptr<mcts::State> Golomb::State::clone_() const {
	return std::makeunique<Golomb::State>(*this);
}

float Golomb::State::getReward(int player) const {
	assert(player == 0);
	return float(_board.getScore());
}

std::string Golomb::State::stateDescription() const {
	// TODO
}

std::string Golomb::State::actionDescription(const _Action & action) const {
	// TODO
}

std::string Golomb::State::actionDescription() {
	// TODO
}

int Golomb::State::parseAction(const std::string& str) {
	// TODO
}
