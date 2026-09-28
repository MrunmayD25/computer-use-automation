"""Reading the text a screenshot shows, with a local recognition engine.

Some applications draw their whole interface on a canvas, so the page holds
no text, form, or button that a structured observation could name. The image
is then the only source for the displayed text. This module recognizes that
text locally and does not save the image. Other components may send masked
screenshots or permitted image regions to the configured model API.

The engine sometimes drops the space between two words, so every comparison
here ignores spacing and case. That is exact for identifiers, statuses, and
account numbers, which hold no spaces; a value with spaces may come back
joined, which the caller's typed contract decides whether to accept.

A reading is deterministic for one picture. It can still misread a
character, so a caller that acts on one checks it against what it can: an
identifier against the input it names, a value against its declared type,
and two readings of two fresh pictures against each other.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

MIN_SCORE = 0.8
"""The confidence below which a line is not read at all."""

_MODEL = Path(__file__).with_name("models") / "en_PP-OCRv4_rec_mobile.onnx"
_MODEL_SHA256 = "e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318"


@dataclasses.dataclass(frozen=True, slots=True)
class Line:
    """One recognized line with its location and confidence.

    ``left``, ``top``, ``width``, and ``height`` are in the picture's pixels,
    which are the coordinates a screen target in the same capture uses.
    """

    text: str
    left: float
    top: float
    width: float
    height: float
    score: float

    @property
    def centre(self) -> tuple[float, float]:
        """Return the middle of the line's box."""
        return self.left + self.width / 2, self.top + self.height / 2

    def holds(self, x: float, y: float) -> bool:
        """Report whether the point lies inside the line's box."""
        return (
            self.left <= x <= self.left + self.width
            and self.top <= y <= self.top + self.height
        )


class Reader(Protocol):
    """Reads a picture into lines. The browser adapter holds one."""

    def lines(self, image: bytes) -> tuple[Line, ...]:
        """Return every line read with at least ``MIN_SCORE`` confidence."""
        ...


class LocalReader:
    """Read upright English UI text with the packaged PP-OCRv4 recognizer.

    The model loads lazily, is verified before use, and needs no network.
    Detector margin preserves narrow glyphs; individual recognition crops
    avoid batch padding changing punctuation. Screenshot text stays upright.
    """

    def __init__(self) -> None:
        self._engine: Any = None

    def lines(self, image: bytes) -> tuple[Line, ...]:
        """Return the lines of ``image`` read with enough confidence."""
        import numpy
        from PIL import Image

        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            if hashlib.sha256(_MODEL.read_bytes()).hexdigest() != _MODEL_SHA256:
                raise RecognitionError("the packaged English OCR model is corrupt")
            self._engine = RapidOCR(
                rec_model_path=str(_MODEL),
                use_cls=False,
                rec_batch_num=1,
                det_unclip_ratio=2.0,
            )
        picture = numpy.array(Image.open(io.BytesIO(image)).convert("RGB"))
        found, _ = self._engine(picture)
        read: list[Line] = []
        for box, text, score in found or ():
            text = str(text).strip()
            if float(score) < MIN_SCORE or not text:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            read.append(
                Line(
                    text,
                    min(xs),
                    min(ys),
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                    float(score),
                )
            )
        return tuple(read)


def squash(text: str) -> str:
    """Drop spacing and case, as every comparison of read text does.

    Examples
    --------
    >>> squash("Order number: O-2001") == squash("Ordernumber:O-2001")
    True
    """
    return "".join(text.split()).casefold()


def at(lines: tuple[Line, ...], x: float, y: float) -> Line | None:
    """Return the one line whose box holds the point, or None."""
    found = [line for line in lines if line.holds(x, y)]
    return found[0] if len(found) == 1 else None


def anchored(lines: tuple[Line, ...], anchor: str) -> list[Line]:
    """Return the lines that start with ``anchor`` or are exactly it.

    Examples
    --------
    >>> read = (Line("Status:active", 0, 0, 90, 20, 0.99),
    ...         Line("Find member", 0, 40, 90, 20, 0.99))
    >>> [line.text for line in anchored(read, "Status:")]
    ['Status:active']
    """
    wanted = squash(anchor)
    return [line for line in lines if wanted and squash(line.text).startswith(wanted)]


def after(line: Line, anchor: str) -> str:
    """Return what a line says after ``anchor``, with the line's own spacing.

    The anchor is matched without spacing or case, and the rest of the line
    is returned as the engine read it.

    Examples
    --------
    >>> after(Line("Status: active", 0, 0, 90, 20, 0.99), "Status:")
    'active'
    >>> after(Line("Ordernumber:O-2001", 0, 0, 90, 20, 0.99), "Order number:")
    'O-2001'
    """
    wanted = squash(anchor)
    taken = 0
    for index, character in enumerate(line.text):
        if not character.isspace():
            taken += 1
        if taken == len(wanted):
            return line.text[index + 1 :].strip()
    return ""


@dataclasses.dataclass(frozen=True, slots=True)
class RadioLabel:
    """A circular control marker and the text recognized beside it.

    The marker proves only a target location. Selection and persisted values
    need their own evidence. All coordinates are ephemeral capture pixels.
    """

    marker: Line
    label: Line

    def holds(self, x: float, y: float) -> bool:
        """Report whether a click landed on this marker or its label."""
        return self.marker.holds(x, y) or self.label.holds(x, y)


class RecognitionError(ValueError):
    """The bounded control scan could not inspect the whole capture."""


def radio_labels(
    image: bytes, reader: Reader, lines: tuple[Line, ...] | None = None
) -> tuple[RadioLabel, ...]:
    """Recognize labels beside isolated radio rings without editing factual OCR.

    Scan only bounded neighborhoods of observed text. A candidate needs a
    round ring larger than adjacent glyphs, an empty separating gap, and an
    independently readable label crop. Excessive scenes raise
    ``RecognitionError``. Both selected and empty centers are allowed.
    """
    import numpy
    from PIL import Image, UnidentifiedImageError

    try:
        picture = Image.open(io.BytesIO(image)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise RecognitionError("control capture is not a readable image") from error
    if picture.width * picture.height > 4_000_000:
        raise RecognitionError("control capture exceeds the pixel limit")
    observed = reader.lines(image) if lines is None else lines
    if len(observed) > 200:
        raise RecognitionError("control capture exceeds the line limit")
    found: list[RadioLabel] = []
    for line in observed:
        left = max(0, int(line.left - 2 * line.height))
        top = max(0, int(line.top - line.height / 3))
        right = min(picture.width, int(line.left + line.width + 3))
        bottom = min(picture.height, int(line.top + line.height * 4 / 3))
        pixels = numpy.asarray(picture.crop((left, top, right, bottom)))
        marker = _radio_ring(pixels, line.left - left, line.height)
        if marker is None:
            continue
        mx, my, wide, tall, label_left = marker
        crop = picture.crop((left + label_left, top, right, bottom))
        encoded = io.BytesIO()
        crop.save(encoded, format="PNG")
        words = _control_label(reader.lines(encoded.getvalue()))
        label = dataclasses.replace(
            words, left=words.left + left + label_left, top=words.top + top
        )
        candidate = RadioLabel(Line("", left + mx, top + my, wide, tall, 1), label)
        if candidate not in found:
            found.append(candidate)
        if len(found) > 40:
            raise RecognitionError("control capture exceeds the candidate limit")
    return tuple(found)


def _control_label(words: tuple[Line, ...]) -> Line:
    if not words:
        raise RecognitionError("a control label could not be read")
    ordered = sorted(words, key=lambda word: word.left)
    top = min(word.top for word in ordered)
    bottom = max(word.top + word.height for word in ordered)
    shared_height = min(word.top + word.height for word in ordered) - max(
        word.top for word in ordered
    )
    if shared_height < min(word.height for word in ordered) * 0.6:
        raise RecognitionError("a control label spans multiple text rows")
    for first, second in pairwise(ordered):
        gap = second.left - first.left - first.width
        if (
            second.left <= first.left
            or gap < -0.4 * min(first.width, second.width)
            or gap > max(first.height, second.height)
        ):
            raise RecognitionError("a control label has ambiguous text pieces")
    left = ordered[0].left
    right = max(word.left + word.width for word in ordered)
    return Line(
        " ".join(word.text for word in ordered),
        left,
        top,
        right - left,
        bottom - top,
        min(word.score for word in ordered),
    )


def _radio_ring(
    pixels: NDArray[np.uint8], text_left: float, text_height: float
) -> tuple[int, int, int, int, int] | None:
    import numpy as np

    if not pixels.size:
        return None
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    background = np.median(border, axis=0)
    ink = np.max(np.abs(pixels.astype(float) - background), axis=2) > 40
    columns = np.flatnonzero(ink.any(axis=0))
    groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
    for index, group in enumerate(groups[:-1]):
        if not len(group) or group[0] > text_left + text_height:
            continue
        left, right = int(group[0]), int(group[-1]) + 1
        rows = np.flatnonzero(ink[:, left:right].any(axis=1))
        if not len(rows):
            continue
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        wide, tall = right - left, bottom - top
        following = int(groups[index + 1][0])
        if not (
            12 <= tall <= text_height * 1.5
            and 0.90 <= wide / tall <= 1.10
            and max(3, wide * 0.2) <= following - right <= wide
        ):
            continue
        glyph_rows = [
            np.flatnonzero(ink[:, part].any(axis=1)) for part in groups[index + 1 :]
        ]
        glyph_height = max(rows[-1] - rows[0] + 1 for rows in glyph_rows)
        if tall < 1.12 * glyph_height:
            continue
        ring = ink[top:bottom, left:right]
        yy, xx = np.indices(ring.shape)
        distance = np.sqrt(
            ((xx + 0.5 - wide / 2) / (wide / 2)) ** 2
            + ((yy + 0.5 - tall / 2) / (tall / 2)) ** 2
        )
        outline = (distance >= 0.80) & (distance <= 1.0)
        moat = (distance >= 0.55) & (distance <= 0.70)
        outside = distance > 1.10
        if (
            ring[outline].mean() >= 0.70
            and ring[moat].mean() <= 0.15
            and not ring[outside].any()
        ):
            return left, top, wide, tall, right + (following - right) // 2
    return None
