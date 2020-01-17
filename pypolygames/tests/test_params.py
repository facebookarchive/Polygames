# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from pathlib import Path
from dataclasses import asdict
from .. import params


@pytest.mark.parametrize(
    "cls",
    [
        params.GameParams,
        params.ModelParams,
        params.OptimParams,
        params.SimulationParams,
        params.EvalParams,
        params.ExecutionParams,
    ],
)
def test_dataclass(cls):
    argfields = dict(cls.arg_fields())
    for key, field in argfields.items():
        if key != field.name.strip("-"):
            raise AssertionError(
                f"ArgField key ('{key}') and "
                f"name ('{field.name.strip('-')}') must match!"
            )
        if "default" in field.opts:
            assert field.opts["default"] == getattr(
                cls, key
            ), f"Field default do not match class one for {key}"
    field_keys = set(argfields)
    cls_attrs = set(
        asdict(cls() if cls != params.EvalParams else cls(checkpoint_dir=Path("blublu")))
    )
    additional = field_keys - cls_attrs
    missing = cls_attrs - field_keys
    errors = []
    if additional:
        errors.append(
            f"Found additional fields {additional} in "
            f"{cls.__name__}.arg_fields() compared to its attributes."
        )
    if missing:
        errors.append(
            f"Found missing fields {missing} in "
            f"{cls.__name__}.arg_fields() compared to its attributes."
        )
    if errors:
        raise AssertionError("\n".join(errors))
