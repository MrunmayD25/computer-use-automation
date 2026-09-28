"""Read unchanged synthetic canvas captures with the installed OCR model."""

from pathlib import Path

from computeruse.reading import LocalReader


def test_english_canvas_preserves_words_upright_ids_and_label_punctuation():
    reader = LocalReader()
    fixtures = Path(__file__).parent / "fixtures/ocr"
    cases = {
        "member-not-found.png": {
            "NM999998|",
            "No member with number NM999998.",
            "Find member",
        },
        "member-details.png": {"Member number: NM000015", "Status: active"},
    }
    for name, expected in cases.items():
        recognized = {
            line.text for line in reader.lines((fixtures / name).read_bytes())
        }
        assert expected <= recognized, (name, expected - recognized, recognized)
