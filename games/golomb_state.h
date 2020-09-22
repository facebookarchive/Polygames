#pragma once

#include "golomb.h"
#include "../core/state.h"

namespace Golomb {
    class Action : public ::_Action {
        public:
            Action(int i, int j, int indexInActions, int nj);
    };

    class State : public ::State{
        private:
            Board _board;

        public:
            State(int seed);
            State(int seed, int history, bool turnFeatures);
            void findFeatures();
            void findActions();

            bool isOnePlayerGame() const override;
            void Initialize() override;
            void ApplyAction(const _Action& action) override;
            void DoGoodAction() override;
            std::unique_ptr<mcts::State> clone_() const override();
            float getReward(int player) const override;

            std::string stateDescription() const override;
            std::string actionDescription(const _Action & action) const override;
            std::strinf actionsDescriotion() override;
            int parseAction(const std::string &str) override;
    };
}
