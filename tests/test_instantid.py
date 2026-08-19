"""CPU tests for the pure InstantID helpers: face selection. The GPU bodies
(image-projection, attn-processor wiring, the per-render call) defer torch/diffusers/insightface
and are validated on a pod, not here."""
from dataclasses import dataclass

from vivijure_backend.instantid import crop_from_bbox, crop_face, largest_face


@dataclass
class _Face:
    bbox: tuple


def test_largest_face_picks_the_biggest_bbox():
    small = _Face((0, 0, 10, 10))      # area 100
    big = _Face((0, 0, 100, 120))      # area 12000
    mid = _Face((0, 0, 50, 50))        # area 2500
    assert largest_face([small, big, mid]) is big


def test_largest_face_empty_is_none():
    assert largest_face([]) is None
    assert largest_face(None) is None


def test_crop_from_bbox_returns_padded_crop():
    from PIL import Image
    img = Image.new("RGB", (100, 80), (10, 20, 30))
    # paint a distinct face rect so we can assert the crop is not the full frame
    for x in range(20, 40):
        for y in range(10, 30):
            img.putpixel((x, y), (200, 0, 0))
    crop = crop_from_bbox(img, (20, 10, 40, 30), pad_ratio=0.0)
    assert crop.size == (20, 20)
    assert crop.getpixel((0, 0)) == (200, 0, 0)
    padded = crop_from_bbox(img, (20, 10, 40, 30), pad_ratio=0.5)
    assert padded.size[0] > crop.size[0]
    assert padded.size[1] > crop.size[1]
    assert padded.size[0] < img.size[0]  # still a crop, not the full studio frame


def test_crop_face_no_face_returns_none():
    from PIL import Image

    class _Analyzer:
        def get(self, _arr):
            return []

    img = Image.new("RGB", (64, 64), (128, 128, 128))
    assert crop_face(_Analyzer(), img) is None


def test_crop_face_uses_largest_bbox():
    from PIL import Image

    class _Face:
        def __init__(self, bbox):
            self.bbox = bbox

    class _Analyzer:
        def get(self, _arr):
            return [_Face((0, 0, 4, 4)), _Face((10, 10, 50, 50))]

    img = Image.new("RGB", (80, 80), (1, 2, 3))
    crop = crop_face(_Analyzer(), img, pad_ratio=0.0)
    assert crop is not None
    assert crop.size == (40, 40)
