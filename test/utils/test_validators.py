"""validators 单元测试。"""

import numpy as np
import pytest

from ahl.utils.validators import (
    validate_point_3d, validate_positive,
    validate_non_negative, validate_array_shape,
)


class TestValidatePoint3D:
    def test_tuple(self):
        assert validate_point_3d((1, 2, 3)) == (1, 2, 3)

    def test_list(self):
        assert validate_point_3d([4, 5, 6]) == (4, 5, 6)

    def test_float_truncated(self):
        assert validate_point_3d((1.9, 2.1, 3.7)) == (1, 2, 3)

    def test_wrong_length(self):
        with pytest.raises(ValueError):
            validate_point_3d((1, 2))

    def test_none(self):
        with pytest.raises(ValueError):
            validate_point_3d(None)


class TestValidatePositive:
    def test_valid(self):
        assert validate_positive(1.0) == 1.0

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            validate_positive(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_positive(-1)


class TestValidateNonNegative:
    def test_zero_ok(self):
        assert validate_non_negative(0) == 0.0

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_non_negative(-0.1)


class TestValidateArrayShape:
    def test_exact_match(self):
        arr = np.zeros((3, 4, 5))
        result = validate_array_shape(arr, (3, 4, 5))
        assert result is arr

    def test_wildcard(self):
        arr = np.zeros((10, 3))
        validate_array_shape(arr, (-1, 3))  # 不报错

    def test_wrong_ndim(self):
        arr = np.zeros((3, 4))
        with pytest.raises(ValueError, match="维数"):
            validate_array_shape(arr, (3, 4, 5))

    def test_wrong_size(self):
        arr = np.zeros((3, 4, 5))
        with pytest.raises(ValueError, match="大小"):
            validate_array_shape(arr, (3, 4, 6))
