# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import tempfile
from contextlib import contextmanager
import pypolygames as ppg
import polygames
import tube

class TrainingEnvironment(object):
    model_fpath = None
    assembler = None
    context = None
    optim = None
    game = None

DEFAULT_CAPACITY = 1000

@contextmanager
def create_training_env(game_name, act_batchsize=50,
    n_act_channels=1, n_games=100, n_actors=1,
    model_device="cuda:0", act_devices=["cuda:0"],
    lr=6.25e-5, eps=1.5e-4,
    replay_capacity=DEFAULT_CAPACITY, seed=1):
    model_fname = game_name + "_model_latest.pt"
    training_env = TrainingEnvironment()
    with tempfile.TemporaryDirectory() as save_dir:
        train_option, eval_option = ppg.workflow.set_up_options(
            seed=seed, save_dir=save_dir, game_name=game_name)

        model_fpath = os.path.join(save_dir, model_fname)
        model = ppg.workflow.create_model(game_name).to(model_device)
        model.save(model_fpath)
        optim = ppg.workflow.create_optimizer(model=model, lr=lr, eps=eps)

        context, assembler, get_train_reward = ppg.workflow.create_train_envs(
            act_batchsize=act_batchsize,
            replay_capacity=replay_capacity,
            game_name=game_name,
            num_game=n_games,
            num_actor=n_actors,
            model_path=model_fpath,
            seed=seed,
            train_option=train_option,
        )

        training_env.train_option = train_option
        training_env.eval_option = eval_option
        training_env.model_fpath = model_fpath
        training_env.assembler = assembler
        training_env.context = context
        training_env.optim = optim
        training_env.get_train_reward = get_train_reward
        try:
            yield training_env
        finally:
            pass

class TestReplayBuffer(unittest.TestCase):

    def test_init(self):
        game_name = "Connect4"
        with create_training_env(game_name) as training_env:
            replay_buffer = training_env.assembler.buffer
            self.assertTrue(hasattr(replay_buffer, 'size'))
            self.assertEqual(replay_buffer.size, 0)
            self.assertTrue(hasattr(replay_buffer, 'capacity'))
            self.assertEqual(replay_buffer.capacity, DEFAULT_CAPACITY)
            self.assertTrue(hasattr(replay_buffer, 'is_full'))
            self.assertFalse(replay_buffer.is_full)

    def test_init_one_game(self):
        game_name = "Connect4"
        with create_training_env(game_name, n_games=1) as training_env:
            training_env.context.start()
            training_env.assembler.start()

            evaluate_before_training(
                game_name=game_name,
                num_game=1,
                seed=1,
                model=training_env.model,
                device="cuda:0",
                eval_option=training_env.eval_option,
                num_actor=1,
            )

            time.sleep(2)
            print("replay buffer size: ", assembler.buffer_size())
            time.sleep(2)
            print("replay buffer size: ", assembler.buffer_size())

if __name__ == '__main__':
    unittest.main()
