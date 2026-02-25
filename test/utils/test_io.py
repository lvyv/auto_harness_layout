"""IO 工具单元测试。"""

import numpy as np
import pytest

from ahl.utils.io import save_npz, load_npz, save_json, load_json


class TestNpzIO:
    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "test.npz")
        a = np.array([1, 2, 3])
        b = np.zeros((3, 4))

        save_npz(path, arr_a=a, arr_b=b)
        loaded = load_npz(path)

        np.testing.assert_array_equal(loaded['arr_a'], a)
        np.testing.assert_array_equal(loaded['arr_b'], b)

    def test_uncompressed(self, tmp_path):
        path = str(tmp_path / "test.npz")
        a = np.ones(100)
        save_npz(path, compress=False, data=a)
        loaded = load_npz(path)
        np.testing.assert_array_equal(loaded['data'], a)


class TestJsonIO:
    def test_save_load_dict(self, tmp_path):
        path = str(tmp_path / "test.json")
        data = {"a": 1, "b": [2, 3], "c": "hello"}
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data

    def test_numpy_serialization(self, tmp_path):
        path = str(tmp_path / "test.json")
        data = {
            "int_val": np.int32(42),
            "float_val": np.float64(3.14),
            "array": np.array([1, 2, 3]),
        }
        save_json(path, data)
        loaded = load_json(path)
        assert loaded["int_val"] == 42
        assert abs(loaded["float_val"] - 3.14) < 1e-10
        assert loaded["array"] == [1, 2, 3]

    def test_unicode(self, tmp_path):
        path = str(tmp_path / "test.json")
        data = {"名称": "测试", "值": 123}
        save_json(path, data)
        loaded = load_json(path)
        assert loaded["名称"] == "测试"
