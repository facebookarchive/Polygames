# Contributing to Polygames
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Our Development Process

Any pull request will trigger continuous integration. Its configuration is
available [here](../.circleci/config.yml).
In particular it defines tests that you should try to run locally as well:
- testing mcts and state C++ code
``` 
./build/test_state
./build/torchRL/mcts/test_mcts 1 100
./build/torchRL/mcts/test_mcts 4 50
```
- testing the python tools:
```
pytest internal pypolygames --durations=10 --verbose
```

- trying a short training:
```
python -m pypolygames traineval --act_batchsize=2 \
  --batchsize=2 --replay_capacity=16  --replay_warmup=2 \
  --num_epoch=1 --num_game=12 --model_name=NanoFCLogitModel \
  --epoch_len=1 --device=cpu --game_name=TicTacToe --sync_period=1  --device_eval=cpu \
  --num_actor_eval=2 --num_rollouts_opponent=50 --num_game_eval=4
```

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Coding Style  

The root contains a ```.clang-format``` file that define the coding style of
this repo, run the following command before submitting PR or push
```
clang-format -i path_to_your_cc_files
clang-format -i path_to_your_h_files
```


## License
By contributing to `Polygames`, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
