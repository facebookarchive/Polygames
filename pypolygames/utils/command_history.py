# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import itertools
from dataclasses import fields
from typing import Optional, List

from ..params import GenericParams


class CommandHistory:
    def __init__(self):
        # remove '='
        command = [
            x
            for x in itertools.chain.from_iterable(
                map(lambda x: x.split("="), sys.argv)
            )
        ]
        self._commands = [command]

    def build_history(self, former_command_history: "CommandHistory"):
        self._commands = former_command_history._commands + self._commands

    def former_commands_contain(self, option: str) -> bool:
        if option[:2] != "--":
            option = f"--{option}"
        for command in self._commands[:-1]:
            if option in command:
                return True
        return False

    def last_command_contains(self, option: str) -> bool:
        if option[:2] != "--":
            option = f"--{option}"
        if self._commands:
            if option in self._commands[-1]:
                return True
        return False

    def last_command_contains_params(
        self, DataclassParams: GenericParams, exclude: Optional[List[str]] = None
    ) -> bool:
        if exclude is None:
            exclude = []
        exclude = [
            f"--{option}" if option[:2] != "--" else option for option in exclude
        ]
        if self._commands:
            for _, arg_field in DataclassParams.arg_fields():
                if arg_field.name not in exclude and self.last_command_contains(
                    arg_field.name
                ):
                    return True
        return False

    def update_params_from_checkpoint(
        self, checkpoint_params: GenericParams, resume_params: GenericParams
    ) -> GenericParams:
        Dataclass = type(checkpoint_params)
        params = {}
        for field in fields(Dataclass):
            formerly_set = self.former_commands_contain(field.name)
            newly_set = self.last_command_contains(field.name)
            if not formerly_set:
                if not newly_set:
                    params.update({field.name: getattr(resume_params, field.name)})
                else:
                    params.update({field.name: getattr(resume_params, field.name)})
            else:
                if not newly_set:
                    params.update({field.name: getattr(checkpoint_params, field.name)})
                else:
                    params.update({field.name: getattr(resume_params, field.name)})
        return Dataclass(**params)
