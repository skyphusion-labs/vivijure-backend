"""CPU tests for the pure InstantID helpers: face selection. The GPU bodies
(image-projection, attn-processor wiring, the per-render call) defer torch/diffusers/insightface
and are validated on a pod, not here."""
from dataclasses import dataclass

from vivijure_backend.instantid import largest_face


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
