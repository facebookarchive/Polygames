# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root, 'build'))

from collections import defaultdict
import torch
import xrl

from . import data_channel_manager


def create_env(batchsize, num_env, timeout_ms):
    dc_fast = xrl.DataChannel('fast', batchsize, timeout_ms)
    dc_slow = xrl.DataChannel('slow', batchsize, timeout_ms)
    dc_manager = data_channel_manager.DataChannelManager([dc_fast, dc_slow])

    context = xrl.Context()
    for i in range(num_env):
        p = xrl.DualDispatchThread(i, 10, dc_fast, dc_slow)
        context.push_env_thread(p)

    return context, dc_manager


if __name__ == '__main__':
    context, dc_manager = create_env(5, 8, 10)
    context.start()

    count = defaultdict(int)
    bcount = defaultdict(int)

    while not context.terminated():
        print('get input')
        batches = dc_manager.get_input(max_timeout_s=1)
        for key, batch in batches.items():
            batchsize = batch['s'].size(0)
            print('@@@ receive:', key, ', batchsize:', batchsize)
            count[key] += 1
            bcount[batchsize] += 1
            reply = {'a': batch['s']}
            dc_manager.set_reply(key, reply)

        print(count)
        print(bcount)

    print('end of the story')
    dc_manager.terminate()
