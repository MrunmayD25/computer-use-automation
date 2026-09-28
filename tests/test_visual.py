"""Checks for the pixel resolver that re-finds a painted control."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from computeruse.visual import MAX_REGIONS, Box, MatchResult, cut_out, locate, segment

PAPER = 255
INK = 40
WIDE = 72
TALL = 28

type Button = tuple[int, int, int]
"""A button to paint: left, top, and how many bars are drawn inside it."""


def _scene(buttons: list[Button], size: tuple[int, int] = (320, 160)) -> bytes:
    """Paint dark buttons on a pale canvas and return PNG bytes.

    Each button carries a different number of pale bars, so two buttons differ
    at quarter scale as well as at full scale. Text would not survive the
    coarse pass, and the resolver would call two labelled buttons the same.
    """
    picture = Image.new("L", size, PAPER)
    draw = ImageDraw.Draw(picture)
    for left, top, bars in buttons:
        draw.rectangle((left, top, left + WIDE - 1, top + TALL - 1), fill=INK)
        for bar in range(bars):
            x = left + 6 + bar * 12
            draw.rectangle((x, top + 6, x + 6, top + TALL - 7), fill=PAPER)
    return _png(picture)


def _png(picture: Image.Image) -> bytes:
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def _crop(scene: bytes, box: tuple[int, int, int, int]) -> bytes:
    with Image.open(io.BytesIO(scene)) as picture:
        return _png(picture.crop(box))


def test_segment_finds_each_button_in_reading_order() -> None:
    scene = _scene([(200, 20, 3), (20, 20, 1), (20, 100, 2)])

    boxes = [candidate.box for candidate in segment(scene)]

    assert [(box.left, box.top) for box in boxes] == [(20, 20), (200, 20), (20, 100)]
    assert all((box.width, box.height) == (WIDE, TALL) for box in boxes)


def test_segment_is_deterministic() -> None:
    scene = _scene([(20, 20, 1), (200, 100, 3)])

    assert segment(scene) == segment(scene)


def test_segment_ignores_specks_smaller_than_a_control() -> None:
    picture = Image.new("L", (120, 60), PAPER)
    ImageDraw.Draw(picture).rectangle((10, 10, 14, 14), fill=INK)

    assert segment(_png(picture)) == ()


def test_segment_caps_the_number_of_candidates() -> None:
    buttons: list[Button] = [
        (10 + 100 * (index % 4), 10 + 60 * (index // 4), 1 + index % 3)
        for index in range(12)
    ]

    assert len(segment(_scene(buttons, size=(640, 400)))) == MAX_REGIONS


def test_locate_refinds_a_control_after_it_moves() -> None:
    before = _scene([(20, 20, 1), (200, 20, 3)])
    crop = segment(before)[0].crop
    after = _scene([(140, 110, 1), (200, 20, 3)])

    match = locate(crop, after)

    assert match.result is MatchResult.FOUND
    assert match.box is not None
    assert (match.box.left, match.box.top) == (140, 110)


def test_locate_reports_identical_controls_as_ambiguous() -> None:
    scene = _scene([(20, 20, 2), (200, 100, 2)])
    crop = _crop(scene, (20, 20, 20 + WIDE, 20 + TALL))

    assert locate(crop, scene).result is MatchResult.AMBIGUOUS


def test_locate_reports_a_control_that_is_gone() -> None:
    crop = segment(_scene([(20, 20, 1)]))[0].crop
    blank = _png(Image.new("L", (320, 160), PAPER))

    match = locate(crop, blank)

    assert match.result is MatchResult.NOT_FOUND
    assert match.box is None


def test_locate_rejects_a_crop_larger_than_the_scene() -> None:
    crop = _scene([(20, 20, 1)])
    scene = _png(Image.new("L", (40, 20), PAPER))

    assert locate(crop, scene).result is MatchResult.NOT_FOUND


def test_cutting_out_a_box_returns_that_part_of_the_picture() -> None:
    scene = _scene([(20, 20, 2)])

    piece = cut_out(scene, Box(20, 20, WIDE, TALL))

    assert piece is not None
    with Image.open(io.BytesIO(piece)) as picture:
        assert picture.size == (WIDE, TALL)


def test_a_box_reaching_past_the_picture_is_refused() -> None:
    scene = _scene([(20, 20, 2)], size=(320, 160))

    assert cut_out(scene, Box(280, 20, WIDE, TALL)) is None
    assert cut_out(scene, Box(-4, 20, WIDE, TALL)) is None
    assert cut_out(scene, Box(20, 120, WIDE, 80)) is None
