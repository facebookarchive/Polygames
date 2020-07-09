/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "state.h"

std::ostream& operator<<(std::ostream& os, const _Action& action) {
  os << action.GetX() << ", " << action.GetY() << ", " << action.GetZ();
  return os;
}

void State::fillFullFeatures() {
  size_t offset = 0;
  auto expand = [&](size_t n) {
    size_t newOffset = offset + n;
    if (newOffset > _fullFeatures.size()) {
      throw std::runtime_error("internal error: _fullFeatures is too small");
    }
    return _fullFeatures.data() + std::exchange(offset, newOffset);
  };
  const size_t planeSize = _featSize[1] * _featSize[2];
  auto add_constant_plane = [&](float value) {
    auto* at = expand(planeSize);
    std::fill(at, at + planeSize, value);
  };
  if (_fullFeatures.empty()) {
    _outFeatSize = _featSize;
    _outFeatSize[0] *= (1 + _featopts->history);
    _outFeatSize[0] +=
        (_featopts->outFeatures ? 1 : 0) + (_featopts->turnFeatures ? 1 : 0) +
        (_featopts->geometricFeatures ? 4 : 0) +
        (_featopts->oneFeature ? 1 : 0) + _featopts->randomFeatures;
    _fullFeatures.resize(_outFeatSize[0] * _outFeatSize[1] * _outFeatSize[2]);

    if (_featopts->history > 0) {
      expand(_features.size() * (_featopts->history + 1));
    } else {
      expand(_features.size());
    }

    if (_featopts->randomFeatures > 0) {
      float* dst =
          expand(_featopts->randomFeatures * _featSize[1] * _featSize[2]);
      for (int k = 1; k <= _featopts->randomFeatures; k++) {
        for (int i = 1; i <= _featSize[1]; i++) {
          for (int j = 1; j <= _featSize[2]; j++) {
            float x = k * 0.754421f + i * 0.147731f + j * 0.242551f;
            x +=
                0.145531f * (i * k) + 0.741431f * (i * j) + 0.134134f * (j * k);
            x += 0.423423f * (i * j * k);
            *dst++ = x - std::floor(x);
          }
        }
      }
    }
    if (_featopts->geometricFeatures) {
      float* dst = expand(4 * _featSize[1] * _featSize[2]);
      for (int k = 0; k < 4; k++) {
        for (int i = 0; i < _featSize[1]; i++) {
          for (int j = 0; j < _featSize[2]; j++) {
            if (k == 0) {
              *dst++ = float(i) / float(_featSize[1] - 1);
            } else if (k == 1) {
              *dst++ = float(j) / float(_featSize[2] - 1);
            } else if (k == 2) {
              float x = float(i) / float(_featSize[1] - 1) - 0.5f;
              float y = float(j) / float(_featSize[2] - 1) - 0.5f;
              *dst++ = x * x + y * y;
            } else if (k == 3) {
              float x1 = float(i) / float(_featSize[1] - 1);
              float x2 = 1.f - float(i) / float(_featSize[1] - 1);
              x2 = x2 < x1 ? x2 : x1;
              float x3 = float(j) / float(_featSize[2] - 1);
              x2 = x2 < x3 ? x2 : x3;
              float x4 = 1.f - float(j) / float(_featSize[2] - 1);
              x2 = x2 < x4 ? x2 : x4;
              *dst++ = x2;
            }
          }
        }
      }
    }
    if (_featopts->oneFeature) {
      add_constant_plane(1);
    }
    if (_featopts->turnFeatures) {
      _turnFeaturesOffset = offset;
      expand(planeSize);
    }
    if (_featopts->outFeatures) {
      float* dst = expand(_featSize[1] * _featSize[2]);
      for (int i = 0; i < _featSize[1]; i++) {
        for (int j = 0; j < _featSize[2]; j++) {
          if ((i == 0) || (i == _featSize[1] - 1) || (j == 0) ||
              (j == _featSize[2] - 1)) {
            *dst++ = (1.);  // 1 for the frontier
          } else {
            *dst++ = (0.);  // 0 for the rest
          }
        }
      }
    }

    _previousFeatures.resize(_features.size() * (_featopts->history + 1));
    for (int i = 0; i != _featopts->history + 1; ++i) {
      std::memcpy(_previousFeatures.data() + _features.size() * i,
                  _features.data(), sizeof(float) * _features.size());
    }
  }
  if (_featopts->history > 0) {
    offset = 0;
    // we check the expected size of features.
    unsigned int expected_size =
        (_featopts->history + 1) * _featSize[0] * _featSize[1] * _featSize[2];
    if (_previousFeatures.size() != expected_size) {
      throw std::runtime_error(
          "internal error: previousFeatures is of incorrect size!");
    }
    std::memcpy(_previousFeatures.data() + _previousFeaturesOffset,
                _features.data(), sizeof(float) * _features.size());

    _previousFeaturesOffset += _features.size();
    if (_previousFeaturesOffset == expected_size) {
      _previousFeaturesOffset = 0;
    }

    // we add the previous features in the full features.
    auto* dst = expand(expected_size);
    std::memcpy(dst, _previousFeatures.data() + _previousFeaturesOffset,
                sizeof(float) * (expected_size - _previousFeaturesOffset));
    std::memcpy(dst + expected_size - _previousFeaturesOffset,
                _previousFeatures.data(),
                sizeof(float) * _previousFeaturesOffset);
  } else {
    offset = 0;
    std::memcpy(expand(_features.size()), _features.data(),
                sizeof(float) * _features.size());
  }
  if (_featopts->turnFeatures) {
    offset = _turnFeaturesOffset;
    add_constant_plane(getCurrentPlayerColor());
  }
}
