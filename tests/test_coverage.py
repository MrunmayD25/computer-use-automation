"""Complete coverage across bounded pages, including changing documents."""

import dataclasses

from computeruse.actions import (
    AxNode,
    Observation,
    ObservationMode,
    ObservationRequest,
    ObservationStatus,
    PageState,
    Window,
)
from computeruse.coverage import gather
from computeruse.retarget import complete


def pages(nodes, *, change=False):
    calls = []

    def read(request):
        calls.append(request.start)
        shown = nodes[request.start : request.start + 200]
        if change and len(calls) > 3 and shown:
            shown = (dataclasses.replace(shown[0], name="Changed"), *shown[1:])
        return Observation(
            str(len(calls)),
            request.mode,
            ObservationStatus.PARTIAL
            if request.start + len(shown) < len(nodes)
            else ObservationStatus.COMPLETE,
            PageState("https://example.test/desk", page="one"),
            nodes=shown,
            window=Window(request.start, len(shown), len(nodes)),
        )

    return read, calls


def test_all_pages_are_collected_and_checked_twice():
    nodes = tuple(AxNode("button", str(i)) for i in range(571))
    read, calls = pages(nodes)
    request = ObservationRequest(ObservationMode.STRUCTURED)
    result = gather(read(request), request, read)
    assert complete(result)
    assert result.nodes == nodes
    assert calls == [0, 200, 400, 0, 200, 400]


def test_changed_or_refused_pages_cannot_claim_complete_coverage():
    nodes = tuple(AxNode("button", str(i)) for i in range(450))
    request = ObservationRequest(ObservationMode.STRUCTURED)
    read, _ = pages(nodes, change=True)
    assert not complete(gather(read(request), request, read))
    read, _ = pages(nodes)
    assert not complete(gather(read(request), request, lambda _: None))


def test_a_scoped_read_does_not_prove_global_uniqueness():
    read, _ = pages((AxNode("button", "Save"),))
    request = ObservationRequest(ObservationMode.STRUCTURED)
    observation = read(request)
    scoped = dataclasses.replace(observation, window=Window(0, 1, 1, frame=("child",)))
    assert not complete(scoped)


def test_discovery_checks_a_record_past_the_first_page():
    from pathlib import Path

    from fakes import Screen, ScriptedDecider, ScriptedEscalator, ScriptedSurface

    from computeruse.actions import AxLocator
    from computeruse.decider import (
        CheckKind,
        Finish,
        Observe,
        ResultCheck,
    )
    from computeruse.journal import MemoryJournal
    from computeruse.loop import Ending, discover
    from computeruse.profile import load_profile

    controls = (*(("text", f"Notice {i}") for i in range(3)), ("text", "10001"))
    surface = ScriptedSurface(
        [Screen("http://127.0.0.1:8787/members", controls=controls)],
        extracts=["10001"],
        page_size=2,
    )
    check = ResultCheck(CheckKind.RECORD, AxLocator("text", "10001"), "10001")
    result = discover(
        "Find member 10001.",
        load_profile(Path("evaluation/profile.yaml")),
        surface=surface,
        decider=ScriptedDecider(
            [
                Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                Finish({}, checks=(check,)),
            ],
        ),
        journal=MemoryJournal(),
        clock=lambda: 0,
        escalator=ScriptedEscalator([]),
    )
    # The identifier is on the second page of the only screen.
    assert result.ending is Ending.COMPLETED
    assert [look.start for look in surface.looks][:3] == [0, 2, 0]


def test_a_failed_frame_on_a_middle_page_is_not_complete_coverage():
    nodes = tuple(AxNode("button", str(i)) for i in range(450))
    read, _ = pages(nodes)

    def failing(request):
        part = read(request)
        if request.start == 200:
            # This page lost a frame, although its count still looks right.
            window = dataclasses.replace(part.window, collected=False)
            return dataclasses.replace(part, window=window)
        return part

    request = ObservationRequest(ObservationMode.STRUCTURED)
    assert not complete(gather(failing(request), request, failing))
