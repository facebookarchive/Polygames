# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import numpy as np

from .result import Result


class Plotter:
    def __init__(self, plot_enabled: bool, env: str, server: str, port: int):
        self.plot_enabled = plot_enabled
        self.env = env
        if plot_enabled:
            import visdom

            self.vis = visdom.Visdom(env=env, server=server, port=port)

    def plot_results(self, results: List[Tuple[int, Result]]):
        if self.plot_enabled:
            epochs, results = list(map(np.array, list(zip(*results))))
            nb_wins, nb_ties, nb_losses = list(
                map(
                    np.array,
                    zip(
                        *[
                            (
                                result.result["win"],
                                result.result["tie"],
                                result.result["loss"],
                            )
                            for result in results
                        ]
                    ),
                )
            )
            nb_totals = np.array(list(map(sum, zip(nb_wins, nb_ties, nb_losses))))
            win_percents, tie_percents, loss_percents = list(
                map(
                    np.array,
                    zip(
                        *[
                            (100 * w / total, 100 * t / total, 100 * l / total)
                            for w, t, l, total in zip(
                                nb_wins, nb_ties, nb_losses, nb_totals
                            )
                        ]
                    ),
                )
            )
            # lines
            self.vis.line(
                win="eval win-tie-loss",
                X=epochs,
                Y=np.array([nb_wins, nb_ties, nb_losses]).T,
                opts={
                    "title": "eval win-tie-loss",
                    "xlabel": "#epochs",
                    "ylabel": "#games",
                    "xtickmin": 0,
                    "ytickmin": 0,
                    "ytickmax": 100,
                    "legend": ["wins", "ties", "losses"],
                },
            )
            # stacked area
            self.vis.line(
                win="eval stacked percentages",
                X=epochs,
                Y=np.array(
                    [
                        win_percents,
                        list(map(sum, zip(win_percents, tie_percents))),
                        [100] * len(epochs),
                    ]
                ).T,
                opts={
                    "title": "eval stacked percentages",
                    "xlabel": "#epochs",
                    "ylabel": "%",
                    "xtickmin": 0,
                    "ytickmin": 0,
                    "ytickmax": 100,
                    "fillarea": True,
                    "legend": ["wins", "ties", "losses"],
                },
            )

    def save(self):
        if self.plot_enabled:
            self.vis.save([self.env])
