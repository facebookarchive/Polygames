# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
from . import utils


class DataChannelManager:
    def __init__(self, channels, *, num_thread=None):
        self.channels = {}
        for c in channels:
            assert c.name not in self.channels
            self.channels[c.name] = c

        self.num_thread = num_thread if num_thread is not None else len(channels)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_thread)

        self.channels_waiting_reply = set()
        self.futures = []
        for _, c in self.channels.items():
            self.futures.append(self.executor.submit(self._channel_get_input, c))

    def __contains__(self, name):
        return name in self.channels

    def _channel_get_input(self, channel):
        """a helper wrapper function for channel.get_input."""
        data = channel.get_input()
        return channel.name, data

    def get_input(self, *, max_timeout_s=None):
        """
        max_timeout_s: the max amount of time (in second) before this
            function returns. if no data arrives within this
            max_timeout, this function will return empty dict {}
        """
        # print('@@@ remaining futures:', len(self.futures))
        utils.assert_eq(len(self.futures), len(self.channels))
        done, pending = concurrent.futures.wait(
            self.futures,
            timeout=max_timeout_s,
            return_when=concurrent.futures.FIRST_COMPLETED)
        done = list(done)
        pending = list(pending)
        # utils.assert_eq(len(done) + len(pending), len(self.futures))
        self.futures = pending

        ready = {}
        for f in done:
            name, data = f.result()
            # assert name not in self.channels_waiting_reply
            # self.channels_waiting_reply.add(name)
            ready[name] = data
        return ready

    def set_reply(self, name, reply):
        reply = {key: reply[key].detach().cpu() for key in reply}
        # breakpoint()
        # assert name in self.channels_waiting_reply
        # self.channels_waiting_reply.remove(name)

        channel = self.channels[name]
        channel.set_reply(reply)
        self.futures.append(self.executor.submit(self._channel_get_input, channel))

    def terminate(self):
        for name, c in self.channels.items():
            c.terminate()

        self.executor.shutdown(wait=True)
