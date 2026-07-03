import numpy as np
import pytest

from g1_primitives.perception.segment import (
    NullSegmenter, Sam3Segmenter, _prompt_kwargs)

H, W = 8, 10


class FakeClient:
    """Stands in for Sam3Client (context manager) returning canned masks."""

    def __init__(self, masks):
        self.masks = masks
        self.kw = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def segment(self, rgb, **kw):
        self.kw = kw
        m = self.masks
        scores = np.arange(len(m), 0, -1, dtype=np.float32)
        return m, scores, [""] * len(m)


def _rgb():
    return np.zeros((H, W, 3), np.uint8)


def test_null_segmenter_returns_none():
    assert NullSegmenter().mask(_rgb()) is None


def test_sam3_segmenter_returns_top_mask():
    m = np.zeros((2, H, W), np.uint8)
    m[0, 1:3, 1:3] = 255
    m[1] = 255
    client = FakeClient(m)
    out = Sam3Segmenter(lambda: client, {"text": "red block"}).mask(_rgb())
    assert out.dtype == bool and out.shape == (H, W)
    np.testing.assert_array_equal(out, m[0] > 0)           # the top (best) mask
    assert client.kw["text"] == "red block" and client.kw["top_k"] == 1


def test_sam3_segmenter_empty_when_not_found():
    out = Sam3Segmenter(lambda: FakeClient(np.zeros((0, H, W), np.uint8)),
                        {"box": [1, 1, 4, 4]}).mask(_rgb())
    assert out is not None and out.dtype == bool and out.shape == (H, W) and out.sum() == 0


def test_prompt_kwargs():
    assert _prompt_kwargs({"text": "x"}) == {"text": "x"}
    assert _prompt_kwargs({"box": [0, 0, 1, 1]}) == {"box": [0, 0, 1, 1]}
    assert _prompt_kwargs({"points": [[1, 2]], "point_labels": [1]}) == \
        {"points": [[1, 2]], "point_labels": [1]}
    assert _prompt_kwargs({"points": [[1, 2]]}) == {"points": [[1, 2]]}
    with pytest.raises(ValueError):
        _prompt_kwargs({})
