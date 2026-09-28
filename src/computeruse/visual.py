"""Resolve a painted control without asking a model where it is.

A canvas control has no element to address. The only thing that persists
between seeing it and clicking it is what it looks like, so this module keeps
a crop of it and matches that crop against the live pixels again immediately
before the click.

The search follows two rules. First, the adapter finds the canvas itself
structurally through the DOM and searches only inside it. Pixels identify only
the control within that canvas. Second, a match must be unique by a margin. A
row of identical painted buttons is ambiguous and requires a person to choose.

Nothing here calls a model. Replay uses this deterministic resolver.
"""

from __future__ import annotations

import dataclasses
import io
from collections import deque
from enum import StrEnum

from PIL import Image, ImageChops, ImageStat

BACKGROUND_TOLERANCE = 24
"""How far a pixel must sit from the page background to count as ink."""

MIN_REGION_WIDTH = 16
MIN_REGION_HEIGHT = 10
"""Smaller than this is a glyph or an artifact, not a control."""

MAX_REGIONS = 8
"""More candidates than an operator would read means the segmentation is wrong."""

MAX_SEGMENT_PIXELS = 4_000_000
"""A canvas larger than this is reported unavailable rather than scanned."""

COARSE_SCALE = 4
"""The first pass runs at a quarter scale; the refinement runs at full scale."""

REFINE_RADIUS = COARSE_SCALE * 2
"""How far the full-scale pass looks around the coarse winner."""

COARSE_TOLERANCE = 60.0
"""Mean difference a coarse offset may carry and still be refined.

Shrinking the crop and the scene separately puts them on different pixel
grids unless the control sits on a multiple of ``COARSE_SCALE``. An exact copy
off that grid scores about 20 at coarse scale, so the coarse pass only
nominates offsets and the full-scale pass decides.
"""

COARSE_CANDIDATES = 3
"""How many separate coarse offsets are refined at full scale."""

MATCH_TOLERANCE = 20.0
"""Mean per-pixel difference a match may carry. Anti-aliasing moves a few units."""

UNIQUE_MARGIN = 1.6
"""A runner-up elsewhere must be this much worse, or the match is ambiguous."""

MIN_SEPARATION = 8.0
"""A runner-up must also be this many units worse, so two perfect copies tie."""

MISS_SCORE = 255.0
"""The score of an offset that could not be refined, the whole shade range."""


class MatchResult(StrEnum):
    """The result of resolving a visual target."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


@dataclasses.dataclass(frozen=True, slots=True)
class Box:
    """A rectangle in the coordinate space of the image it was found in."""

    left: int
    top: int
    width: int
    height: int

    @property
    def centre(self) -> tuple[float, float]:
        """Return the middle of the box, for a click that happens right now.

        This is computed during execution and thrown away. It is never part of
        how a target is named, because a coordinate names nothing a later run
        could find.
        """
        return (self.left + self.width / 2, self.top + self.height / 2)


@dataclasses.dataclass(frozen=True, slots=True)
class Candidate:
    """One painted region, with the crop used to find it again."""

    box: Box
    crop: bytes


@dataclasses.dataclass(frozen=True, slots=True)
class Match:
    """A crop's location in a later image and whether the match was unique."""

    result: MatchResult
    box: Box | None = None
    score: float = 0.0


def segment(image: bytes) -> tuple[Candidate, ...]:
    """Find candidate painted controls in one image, deterministically.

    The background is the most common shade in the image. Everything else is
    ink, and each connected run of ink is one candidate. Ordering is by
    position, top row first, so the same canvas always yields the same list
    in the same order and an anchor name means the same thing twice.

    Parameters
    ----------
    image
        PNG bytes of one canvas element.

    Returns
    -------
    tuple of Candidate
        At most ``MAX_REGIONS`` candidates, each carrying its own crop.
    """
    picture = _load(image)
    width, height = picture.size
    if width * height > MAX_SEGMENT_PIXELS or width == 0 or height == 0:
        return ()
    pixels = picture.tobytes()
    background = _background(picture)
    boxes = _components(pixels, width, height, background)
    candidates = []
    for box in boxes[:MAX_REGIONS]:
        crop = picture.crop(
            (box.left, box.top, box.left + box.width, box.top + box.height)
        )
        candidates.append(Candidate(box=box, crop=_dump(crop)))
    return tuple(candidates)


def cut_out(image: bytes, box: Box) -> bytes | None:
    """Return ``box`` of ``image`` as PNG bytes, or None if it does not fit.

    The caller passes a box it read from the live page, so the box can sit
    partly or wholly outside the picture. Returning None rather than a
    clamped crop keeps a half-captured control from being offered as a whole
    one.
    """
    picture = _load(image)
    width, height = picture.size
    right, bottom = box.left + box.width, box.top + box.height
    if box.left < 0 or box.top < 0 or right > width or bottom > height:
        return None
    if box.width < 1 or box.height < 1:
        return None
    return _dump(picture.crop((box.left, box.top, right, bottom)))


def locate(crop: bytes, scene: bytes) -> Match:
    """Find ``crop`` inside ``scene`` and report whether it was found once.

    A coarse pass at a quarter scale nominates up to ``COARSE_CANDIDATES``
    offsets that do not overlap each other. A full-scale pass refines each
    one. A match requires an error score no higher than the threshold and at
    least the required margin below the runner-up. Two identical painted
    buttons therefore produce ``AMBIGUOUS``.
    """
    needle = _load(crop)
    haystack = _load(scene)
    if needle.size[0] > haystack.size[0] or needle.size[1] > haystack.size[1]:
        return Match(MatchResult.NOT_FOUND)
    coarse_needle = _shrink(needle, COARSE_SCALE)
    coarse_scene = _shrink(haystack, COARSE_SCALE)
    scores = _scan(coarse_needle, coarse_scene, step=1)
    nominees = _nominees(scores, coarse_needle.size)
    if not nominees:
        return Match(MatchResult.NOT_FOUND)

    refined = sorted(
        (_refine(needle, haystack, nominee) for nominee in nominees),
        key=lambda item: item[2],
    )
    left, top, best = refined[0]
    if best > MATCH_TOLERANCE:
        return Match(MatchResult.NOT_FOUND, score=best)
    if len(refined) > 1:
        rival = refined[1][2]
        if rival < max(best * UNIQUE_MARGIN, best + MIN_SEPARATION):
            return Match(MatchResult.AMBIGUOUS, score=best)
    box = Box(left=left, top=top, width=needle.size[0], height=needle.size[1])
    return Match(MatchResult.FOUND, box=box, score=best)


def _nominees(
    scores: list[tuple[int, int, float]], size: tuple[int, int]
) -> list[tuple[int, int, float]]:
    """Return the best coarse offsets, none overlapping another, best first."""
    wide, tall = size
    chosen: list[tuple[int, int, float]] = []
    for item in sorted(scores, key=lambda entry: entry[2]):
        if item[2] > COARSE_TOLERANCE or len(chosen) == COARSE_CANDIDATES:
            break
        if all(
            abs(item[0] - other[0]) >= wide or abs(item[1] - other[1]) >= tall
            for other in chosen
        ):
            chosen.append(item)
    return chosen


def _refine(
    needle: Image.Image, haystack: Image.Image, coarse: tuple[int, int, float]
) -> tuple[int, int, float]:
    """Return the best full-scale offset near one coarse offset."""
    origin = (coarse[0] * COARSE_SCALE, coarse[1] * COARSE_SCALE)
    window = _scan(needle, haystack, step=1, origin=origin, radius=REFINE_RADIUS)
    if not window:
        return (coarse[0] * COARSE_SCALE, coarse[1] * COARSE_SCALE, MISS_SCORE)
    return min(window, key=lambda item: item[2])


def _scan(
    needle: Image.Image,
    haystack: Image.Image,
    *,
    step: int,
    origin: tuple[int, int] | None = None,
    radius: int = 0,
) -> list[tuple[int, int, float]]:
    """Return the mean absolute difference at every offset considered.

    The difference itself is computed by the imaging library, so each offset
    costs one library call rather than a pixel loop in Python.
    """
    wide, tall = needle.size
    limit_x = haystack.size[0] - wide
    limit_y = haystack.size[1] - tall
    if limit_x < 0 or limit_y < 0:
        return []
    if origin is None:
        columns = range(0, limit_x + 1, step)
        rows = range(0, limit_y + 1, step)
    else:
        columns = range(
            max(0, origin[0] - radius), min(limit_x, origin[0] + radius) + 1, step
        )
        rows = range(
            max(0, origin[1] - radius), min(limit_y, origin[1] + radius) + 1, step
        )
    area = float(wide * tall)
    scores: list[tuple[int, int, float]] = []
    for top in rows:
        for left in columns:
            patch = haystack.crop((left, top, left + wide, top + tall))
            total = ImageStat.Stat(ImageChops.difference(patch, needle)).sum[0]
            scores.append((left, top, total / area))
    return scores


def _components(pixels: bytes, width: int, height: int, background: int) -> list[Box]:
    """Group ink pixels into bounding boxes, four-connected.

    Examples
    --------
    >>> width, height = 24, 14
    >>> ink = bytes(
    ...     255 if 2 <= index % width < 20 and 2 <= index // width < 12 else 0
    ...     for index in range(width * height)
    ... )
    >>> [(box.left, box.top, box.width, box.height) for box in
    ...  _components(ink, width, height, 0)]
    [(2, 2, 18, 10)]
    """
    seen = bytearray(width * height)
    boxes: list[Box] = []
    for index in range(width * height):
        if seen[index] or abs(pixels[index] - background) <= BACKGROUND_TOLERANCE:
            continue
        left = right = index % width
        top = bottom = index // width
        seen[index] = 1
        queue = deque([index])
        while queue:
            here = queue.popleft()
            x, y = here % width, here // width
            left, right = min(left, x), max(right, x)
            top, bottom = min(top, y), max(bottom, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                neighbour = ny * width + nx
                if seen[neighbour]:
                    continue
                if abs(pixels[neighbour] - background) <= BACKGROUND_TOLERANCE:
                    continue
                seen[neighbour] = 1
                queue.append(neighbour)
        box = Box(left, top, right - left + 1, bottom - top + 1)
        if box.width >= MIN_REGION_WIDTH and box.height >= MIN_REGION_HEIGHT:
            boxes.append(box)
    boxes.sort(key=lambda box: (box.top, box.left))
    return boxes


def _background(picture: Image.Image) -> int:
    counts = picture.histogram()
    return max(range(len(counts)), key=lambda shade: counts[shade])


def _load(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as picture:
        return picture.convert("L")


def _shrink(picture: Image.Image, factor: int) -> Image.Image:
    width = max(1, picture.size[0] // factor)
    height = max(1, picture.size[1] // factor)
    return picture.resize((width, height), Image.Resampling.BILINEAR)


def _dump(picture: Image.Image) -> bytes:
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()
