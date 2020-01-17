# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""utils"""


def assert_eq(real, expected, msg='assert_eq fails'):
    assert real == expected, '%s: %s (real) vs %s (expected)' % (msg, real, expected)
