[![CircleCI](https://circleci.com/gh/facebookincubator/Polygames.svg?style=svg)](https://circleci.com/gh/facebookincubator/Polygames)

# Polygames

This README is a work in progress, please feel very free to post issues - we are happy to help.
Save up computational power: you can find checkpoints here: http://dl.fbaipublicfiles.com/polygames/checkpoints/list.txt (feel free to open an issue for discussing which checkpoint you should use for which game/problem!).

For Nix users: see [this doc](./nix/README.md).

## Requirement:
```
C++17 compatible compiler
miniconda3
```

## Compilation Guide:

### First install conda and pytorch

Create a fresh conda environment with python3.7, install pytorch and dependencies.

```
# create a fresh conda environment with python3
# you will need to have miniconda3 set up
conda create --name [your env name] python=3.7 pip

conda activate [your env name] # Or source activate [your env name], depending on conda version.

conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install pytorch cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorboardx
conda install -c conda-forge openjdk  # optional
conda install -c conda-forge graphviz

pip install visdom
pip install torchviz

```

### Clone the repo and build


```
git clone --recursive https://github.com/facebookincubator/polygames
cd polygames

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=relwithdebinfo -DPYTORCH15=ON
make -j

```

LUDII support can be disabled by appending `-DWITH_LUDII=OFF` to the cmake command (required if you don't have jdk)

## Content

The repo contains mostly the following folders:

- the `pypolygames` python package, which serves as an entry point for the application
- the `src` folder, containing all C++ source code and third party libraries
  - the `src/games` folder, containing the games coded in C++

## How to use the application

The application is launched from the `pypolygames` python package, in either of the following modes:ar

- `pypolygames train` (training mode): a game and a model (as well as several other options, see below) are chosen and the model is iteratively trained with MCTS
- `pypolygames eval` (evaluation mode): the model confronts either a pure MCTS or another neural network powered MCTS. The evaluation of a training can be done either offline (from checkpoints periodically saved) or in real time; in that case, the evaluation considers only the most recent checkpoint in order to follow closely the training, skipping some checkpoints in case the eval computation takes longer than the time becween consecutive checkpoints. It is displayed through visdom.
- `pypolygames traineval` (training + evaluation mode): it mixes the two previous modes and allow to launch one command instead of two. With the `real_time` option the modes can be launched in parallel instead of sequentially.
- `pypolygames human` (human mode): a human player plays against the machine

When a training is launched, it creates a `game_GAMENAME_model_MODELNAME_feat_FEATURIZATION_GMT_YYYYMMDDHHMMSS` within the `save_dir` where it will log relevant files:
- `model.pt`
- `train.log`
- `stat.tb`
- `checkpoints_EPOCH.pt` for for checkpoints saved each `saving_period` epoch (e.g., if `saving_period == 10`, `checkpoints_0.pt`, `checkpoints_10.pt`, `checkpoints_20.pt`, `checkpoints_30.pt`)

This directory will be the `checkpoint_save_dir` directory used by evaluation to retrieve the checkpoints to perform eval computation.

### Parameters

The list of parameters for each mode is available with

```
python -m pypolygames {train,eval,traineval,human} --help
```

#### Threads

In train (resp. eval) mode, `num_game * num_actor` (resp. `num_game * num_actor_eval * num_actor_opponent`) is the total number of threads. The more `num_actor` (and `num_actor_eval`, `num_actor_opponent`), the larger the MCTS is for a given player.

In human mode, since `num_game` is set to one, for leveraging the computing power available on the platform, a rule-of-thumb is to set `num_actor` to 5 times the number of CPUs available (it is platform-dependent though, and performance tests should be done).

### Model zoo

All models can be found in `pypolygames/model_zoo`. They come with a set of sensible parameters that can be customized as well as default games.

Usually models come in pair: `MODELNAMEFCLogitModel` and `MODELNAMEConvLogitModel`:

- `FCLogit` models use a fully-connected layer for logit inference and are compatible with all games
- `ConvLogit` models use a convolutional layer for logit inference and are only compatible with games whose action space if of same dimensions than their input space (an exception will be raised in case of an attempt to use an incompatible game)

So far the models being implemented are the folling:

- `GenericModel`: generic model compatible with all games, default when no `model_name` is specified
- `NanoFCLogitModel`: a simple model with a logit-inference fully-connected layer
- `NanoConvLogitModel`: a simple model with a logit-inference convolutional layer
- `ResConvFCLogitModel`: resnets with a logit-inference fully-connected layer
- `ResConvConvLogitModel`: resnets with a logit-inference convolutional layer
- `UConvFCLogitModel`: unets (direct paths between first and last layers) with a logit-inference fully-connected layer
- `UConvConvLogitModel`: unets (direct paths between first and last layers) with a logit-inference convolutional layer
- `AmazonsModel`: only for the Amazons game

Depending on the actual model chosen, some parameters might not have any use.

### Featurization

```
--out_features=True: the input to the NN includes a channel with 1 on the frontier.
--turn_features=True: the input to the NN includes a channel with the player index broadcasted.
--geometric_features=True: the input to the NN includes 4 geometric channels representing the position on the board.
--random_features=4: the input to the NN includes 4 random features.
--one_feature=True: the input to the NN includes a channel with 1 everywhere.
--history=3: the representation from the last 3 steps is added in the featurization.
```
### Examples

Run the following command before running the code
```
export OMP_NUM_THREADS=1
```

#### Examples for the training mode

- Launch the game `Connect4` with the `GenericModel`
```
python -m pypolygames train --game_name="Connect4"
```

- Launch a game with a specific model and specific parameters
```
python -m pypolygames train --game_name="Connect4" --out_features=True \
    --model_name="UConvFCLogitModel" \
    --nnsize=16 \
    --nnks=3 \
    --pooling
```

- Save checkpoints every 20 epochs in a specific folder
```
python -m pypolygames train --game_name="Connect4" --model_name="UConvFCLogitModel" \
    --saving_period=20 \
    --save_dir="/checkpoints"
```

- Run training on GPU for a max time
```
python -m pypolygames train --game_name="Connect4" --model_name="UConvFCLogitModel" \
    --device="cuda:0" \
    --max_time=3600
```

- Resume training from a given epoch
```
python -m pypolygames train \
    --save_dir="/checkpoints/game_Connect4_model_GenericModel_feat..._GMT_20190717103728" \
    --init_epoch=42
```

- Initiate from a pretrained model
```
python -m pypolygames train --init_checkpoint="path/to/pretrained_model.pt" \
    --lr=0.001
```

Note that any checkpoint can serve as a pretrained model

- Train on multiple GPUs
```
python -m pypolygames train --init_checkpoint "path/to/pretrained_model.pt" \
    --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4
```

In this case `cuda:0` will be used for training the model while `cuda:1`, `cuda:2` and `cuda:3` will be used for generating games. If there is only one device specified, it will be used for both purposes.

Notes:

- By default, the number of threads used for processing and batch sizes for inference are set automatically. These can be overriden with `num_thread` and `per_thread_batchsize` respectively.
- `num_game` specifies the number of "master" threads scheduling games, and the total number of games being run in parallel will be `num_game * per_thread_batchsize`. Since `per_thread_batchsize` is automatically determined by default, this could be a large number in some instances.

#### Examples for the evaluation mode

- Run offline evaluation
```
python -m pypolygames eval \
    --checkpoint_dir="/checkpoints/game_Connect4_model_GenericModel_feat..._GMT_20190717103728"
```

- Plot evaluation on `http://localhost:10000` as the same time as training happens (training needs to be run from another process)
```
python -m pypolygames eval \
    --checkpoint_dir="/checkpoints/game_Connect4_model_GenericModel_feat..._GMT_20190717103728" \
    --real_time \
    --plot_enabled \
    --plot_port=10000
```

- Run evaluation on cpu with 100 games per evaluation, the pure-MCTS opponent playing 1000 rollouts while the model plays 400 rollouts
```
python -m pypolygames eval \
    --checkpoint_dir="/checkpoints/game_Connect4_model_GenericModel_feat..._GMT_20190717103728" \
    --device_eval="cpu" \
    --num_game_eval=100 \
    --num_rollouts_eval=400 \
    --num_actor_eval=8 \
    --num_rollouts_opponent=1000 \
    --num_actor_opponent=8
```

- A specific checkpoint plays against another neural-network-powered MCTS
```
python -m pypolygames eval \
    --checkpoint="/checkpoints/checkpoint_600.zip" \
    --num_rollouts_eval=400 \
    --num_actor_eval=8 \
    --checkpoint_opponent="/checkpoints/checkpoint_200.zip" \
    --num_rollouts_opponent=1000 \
    --num_actor_opponent=8
```

- Four GPUs are used for evaluating the model, all for inference
```
python -m pypolygames eval \
    --checkpoint="/checkpoints/checkpoint_600.zip" \
    --device_eval cuda:0 cuda:1 cuda:2 cuda:3 \
    --num_rollouts_eval=400 \
    --num_actor_eval=8 \
    --num_rollouts_opponent=1000 \
    --num_actor_opponent=8
```

Notes:

- `num_actor_eval`, `num_rollouts_eval`, `num_actor_opponent` and `num_rollouts_opponent` are independent from the values used during training; in particular for proper benchmarking `num_actor_eval` and `num_rollouts_eval` should be set to the values used in human mode
- `num_game_eval * num_actor_eval` (resp. `num_game_eval * num_actor_opponent`) is the number of threads used by the model to be evaluated (resp. the opponent)
- there is no `per_thread_batchsize` in this mode
- the higher `num_actor_eval` (resp. `num_actor_opponent`), the larger MCTS for a move in a given game will be, up to a limit where overheads between threads lead to decreasing returns. Empiracally this limit seems to be around 8. This limit may be game/model/platform dependent and should be tuned for a given instance.
- against a pure MCTS opponent, `num_rollouts_opponent` should be set significantly higher than `num_rollouts_eval`

#### Examples for the training+evaluation mode

- Run first training then evaluation on the last checkpoint
```
python -m pypolygames traineval --game_name="Connect4" \
    --save_dir="/checkpoints" \
    --num_epoch=1000
```

- Plot evaluation on `http://localhost:10000` as the same time as training happens
```
python -m pypolygames traineval --game_name="Connect4" \
    --save_dir="/checkpoints" \
    --real_time \
    --plot_enabled \
    --plot_port=10000
```

#### Examples for the human mode

- Play to Connect4 against a pure MCTS as the second player with 8 threads
```
python -m pypolygames human --game_name="Connect4" \
    --pure_mcts \
    --num_actor 8
```

- Play to Connect4 against a pretrained model as the second player
```
python -m pypolygames human \
    --init_checkpoint="/checkpoints/checkpoint_600.zip" \
    --human_first
```

- Play with a timer, each side having 1800s in total, and the model playing each move with 0.07 of the remaining time
```
python -m pypolygames human \
    --init_checkpoint="/checkpoints/checkpoint_600.zip" \
    --total_time=1800 \
    --time_ratio=0.07
```

- The model uses four GPUs, all for inference
```
python -m pypolygames human \
    --init_checkpoint "/checkpoints/checkpoint_600.zip" \
    --device cuda:0 cuda:1 cuda:2 cuda:3
```

- The model uses four GPUs, all for inference, and uses the text protocol (actions are represented by x y z, each on one line):
```
python -m pypolygames tp \
    --init_checkpoint "/checkpoints/checkpoint_600.zip" \
    --device cuda:0 cuda:1 cuda:2 cuda:3
```

Notes:

- in human mode, the model being fixed, the goal is to maximize performance given the platform running the model
- the most effective way to improve model performance is to increase the MCTS size
- as for training and evaluation, but given that there is only one game played, `num_actor` is the total number of threads
- the higher `num_actor`, the larger the MCTS, up to a limit where overheads between threads lead to decreasing returns. Empiracally this limit seems to be around 8. This limit may be game/model/platform dependent and should be tuned for a given instance.
- in a time-limited game `num_rollouts` should not be specified as it is maximized within each `time_ratio` * remaining time period

### Examples for converting models

Saved checkpoints of models also store details about the game for which they were trained, and can only be used directly for the
game in which they were trained. This is why `eval` runs do not require the `--game_name` to be specified; this is inferred from
the model. The `pypolygames convert` command can be used to convert models to different games.

- Fully automated convert between games:

```
python -m pypolygames convert \
    --init_checkpoint "/checkpoints/checkpoint_600.pt.gz" \
	--game_name="LudiiGomoku.lud" \
	--out="/checkpoints/converted/XToGomoku.pt.gz"
```

This takes the previously-trained model stored in `"/checkpoints/checkpoint_600.pt.gz"`,
modifies it such that it can be used to play the Ludii implementation of Gomoku, and stores
this modified version of the model in the new file `"/checkpoints/converted/XToGomoku.pt.gz"`.

This works best when using neural network architectures that are compatible with arbitrary
board shapes (such as `ResConvConvLogitPoolModel`), and source and target games that have
identical numbers of channels for state and move tensors, as well as identical semantics for
those channels. For instance, the Ludii implementation of Yavalath has the same number of
channels with identical semantics (in the same order) as Gomoku. Therefore, if the source model
in `"/checkpoints/checkpoint_600.pt.gz"` was trained using `--model_name=ResConvConvLogitPoolModel`
and `--game_name="LudiiYavalath.lud"`, this conversion can be performed directly without having
to delete any parameters or add any new parameters.

- Fully automated convert between game options:

```
python -m pypolygames convert \
    --init_checkpoint "/checkpoints/checkpoint_600.pt.gz" \
	--game_options="Board Size/19x19" \
	--out="/checkpoints/converted/Gomoku/15x15_to_19x19.pt.gz"
```

This example will convert the source checkpoint `"/checkpoints/checkpoint_600.pt.gz"`
into a model that can be used in a game loaded with the additional 
`--game_options="Board Size/19x19"` argument. For example, `--game_name=LudiiGomoku.lud`
is by default played on a 15x15 board, but can be played on a larger 19x19 board with
the `--game_options="Board Size/19x19"` argument.

Note that the convert command only takes game options into account if some form of
`--game_options` is explicitly provided among the command line arguments. This means that, if
a model was first trained for `--game_options=Board Size/19x19`, and the goal is to convert
it into one for the default board size of 15x15, it is still necessary to provide either 
`--game_options` (without any values after it) or `--game_options=Board Size/15x15`
to the convert script. This tells it that the goal is indeed to revert to default options,
rather than just leaving whichever options were baked into the source model.

## Contributing

We welcome contributions! Please check basic instructions [here](.github/CONTRIBUTING.md)

## Initial contributors

Contributors to the early version of Polygames (before open source release) include:

Tristan Cazenave, Univ. Dauphine; Yen-Chi Chen, National Taiwan Normal University; Guan-Wei Chen, National Dong Hwa University; Shi-Yu Chen, National Dong Hwa University; Xian-Dong Chiu, National Dong Hwa University; Julien Dehos, Univ. Littoral Cote d’Opale; Maria Elsa, National Dong Hwa University; Qucheng Gong, Facebook AI Research; Hengyuan Hu, Facebook AI Research; Vasil Khalidov, Facebook AI Research; Chen-Ling Li, National Dong Hwa University; Hsin-I Lin, National Dong Hwa University; Yu-Jin Lin, National Dong Hwa University; Xavier Martinet, Facebook AI Research; Vegard Mella, Facebook AI Research; Jeremy Rapin, Facebook AI Research; Baptiste Roziere, Facebook AI Research; Gabriel Synnaeve, Facebook AI Research; Fabien Teytaud, Univ. Littoral Cote d’Opale; Olivier Teytaud, Facebook AI Research; Shi-Cheng Ye, National Dong Hwa University; Yi-Jun Ye, National Dong Hwa University; Shi-Jim Yen, National Dong Hwa University; Sergey Zagoruyko, Facebook AI Research

## License

`polygames` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
Third-party libraries are also included under their own license.
