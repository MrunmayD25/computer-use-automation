"""Cover each reported defect with a test that failed before its fix.

The browser cases run against real Chromium and the pages under
``evaluation/``. The loop cases use scripted collaborators because they cover
control flow rather than page behavior. No test calls a model. Two payload
checks use ``httpx.MockTransport`` to assert what this project sends.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
from collections.abc import Callable, Iterator

import httpx
import pytest
from fakes import (
    DONE,
    DONE_TARGET,
    FakeClock,
    Screen,
    ScriptedDecider,
    ScriptedEscalator,
    ScriptedSurface,
    grants,
)
from PIL import Image

from computeruse.actions import (
    Action,
    ActionResult,
    AxLocator,
    AxNode,
    DomAttribute,
    DomLocator,
    Expectation,
    Observation,
    ObservationMode,
    ObservationRequest,
    ObservationStatus,
    Outcome,
    PageState,
    Point,
    RecordEvidence,
    Relation,
    Scope,
    ScopeKind,
    ScreenTarget,
    SecretRef,
    VisualAnchor,
    VisualMeta,
    VisualRegion,
    names_node,
    unbound,
)
from computeruse.browser import BrowserLimits, BrowserSurface, open_session
from computeruse.decider import (
    AskHuman,
    CheckKind,
    Decision,
    FactRef,
    Finish,
    FlagRisk,
    Match,
    ModelError,
    Observe,
    Propose,
    Remember,
    ResultCheck,
    RiskFinding,
    Task,
    TaskOutput,
    TaskRecord,
    Transcript,
)
from computeruse.escalation import (
    Escalator,
    Handoff,
    HandoffOutcome,
    InterventionRequest,
    Trigger,
)
from computeruse.journal import (
    Acted,
    Escalated,
    Flagged,
    Interpreted,
    MemoryJournal,
    Observed,
    Refused,
    Verified,
    as_record,
)
from computeruse.loop import Ending, RunResult, Verification, discover
from computeruse.model import LunaDecider
from computeruse.policy import Denial, Source
from computeruse.profile import ActionKind, Profile, Risk

STRUCTURED = ObservationRequest(ObservationMode.STRUCTURED)
VISUAL = ObservationRequest(ObservationMode.VISUAL)
MASK = (16, 16, 16)

PREFILLED = "synthetic-prefilled-passcode"
ONE_TIME_CODE = "synthetic-one-time-code"

APPROVE = AxLocator("button", "Approve")
PREFILL = DomLocator("button", DomAttribute.ID, "prefill-passcode")
NEXT_RECORD = DomLocator("button", DomAttribute.ID, "next-record")
MOVE_APPROVE = DomLocator("button", DomAttribute.ID, "move-approve")
DETACH_LEDGER = DomLocator("button", DomAttribute.ID, "detach-ledger")
SWAP_PADS = DomLocator("button", DomAttribute.ID, "swap-pads")
COVER_PADS = DomLocator("button", DomAttribute.ID, "cover-pads")
POST_ENTRY = DomLocator("button", DomAttribute.ID, "post-entry")
APPROVED_RECORD = DomLocator("output", DomAttribute.ID, "approved-record")
APPROVAL_STATE = DomLocator("output", DomAttribute.ID, "approval-state")
DIALOG_STATE = DomLocator("output", DomAttribute.ID, "dialog-state")
HOLD_STATE = DomLocator("output", DomAttribute.ID, "hold-state")

DIALOGS = {
    "observe": "safe",
    "read": "safe",
    "click": "safe",
    "navigate": "safe",
    "accept_dialog": "safe",
    "dismiss_dialog": "safe",
}


@pytest.fixture
def surface(site: str, site_profile) -> Callable[..., Iterator[BrowserSurface]]:
    """Return a factory that opens a session at one of the evaluation pages."""

    def _open(path: str = "/members/12345", **edits: object):
        profile: Profile = site_profile(**edits)
        return open_session(profile, f"{site}{path}")

    return _open


def click(page: BrowserSurface, target: DomLocator) -> Outcome:
    return page.act(Action(ActionKind.CLICK, target)).outcome


def read(page: BrowserSurface, target: DomLocator) -> str | None:
    return page.act(Action(ActionKind.READ, target)).extracted


def node_for(observation: Observation, element_id: str):
    for node in observation.nodes:
        if ("id", element_id) in node.attributes:
            return node
    raise AssertionError(f"no node carried the id {element_id}")


def mask_pixels(image: bytes | None) -> int:
    """Count the pixels the driver painted over while capturing."""
    assert image is not None
    with Image.open(io.BytesIO(image)) as picture:
        pixels = picture.convert("RGB").tobytes()
    return sum(
        1
        for index in range(0, len(pixels), 3)
        if tuple(pixels[index : index + 3]) == MASK
    )


def expectation(observation: Observation, target: AxLocator) -> Expectation:
    """Build what the loop would send with an action against this observation."""
    found = [node for node in observation.nodes if names_node(target, node)]
    assert len(found) == 1
    return Expectation(page_state=observation.page_state, context=found[0].context)


# 1. A credential value must not reach an observation or the model payload.


def test_a_rendered_password_value_attribute_is_never_collected(surface) -> None:
    with surface() as page:
        click(page, PREFILL)
        observation = page.observe(STRUCTURED)

    field = node_for(observation, "approver-pin")
    assert field.secret is True
    assert field.value is None
    assert [name for name, _ in field.attributes if name == "value"] == []
    assert PREFILLED not in repr(observation)


def test_a_one_time_code_value_is_never_collected(surface) -> None:
    with surface() as page:
        observation = page.observe(STRUCTURED)

    field = node_for(observation, "otp-code")
    assert field.secret is True
    assert field.value is None
    assert ONE_TIME_CODE not in repr(observation)


def test_credential_fields_are_masked_in_a_screenshot(surface) -> None:
    with surface() as page:
        click(page, PREFILL)
        picture = page.observe(VISUAL)

    assert picture.visual is not None
    assert "credential fields" in picture.visual.masked
    assert mask_pixels(picture.image) > 0


def test_the_one_time_code_field_is_masked_as_well_as_the_password(
    surface,
) -> None:
    """Both fields are covered. One field's worth of mask is not enough."""
    with surface() as page:
        for _ in range(4):
            page.act(Action(ActionKind.SCROLL, value="down"))
        picture = page.observe(VISUAL)

    assert picture.visual is not None
    assert "credential fields" in picture.visual.masked
    # A default text input covers about 3,200 pixels in this viewport, so a
    # count this high cannot come from the password field alone.
    assert mask_pixels(picture.image) > 5_000


def test_no_credential_value_reaches_the_model_payload(surface, profile) -> None:
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "name": "ask_human",
                        "arguments": json.dumps(
                            {"reason": "no_progress", "detail": ""}
                        ),
                    }
                ],
            },
        )

    with surface() as page:
        click(page, PREFILL)
        page.act(
            Action(ActionKind.TYPE, AxLocator("textbox", "One time code"), "typed")
        )
        observations = (page.observe(STRUCTURED), page.observe(VISUAL))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        LunaDecider(api_key="not-a-real-key", client=client).decide(
            Transcript(
                goal="verify the approver",
                location=observations[0].location,
                allowed=profile.actions,
                permitted_modes=(ObservationMode.STRUCTURED, ObservationMode.VISUAL),
                observations=observations,
            )
        )

    payload = sent[0].content.decode()
    assert PREFILLED not in payload
    assert ONE_TIME_CODE not in payload


# 2. A decision must not land on a record that replaced the one it was made for.


def test_a_moved_button_for_the_same_record_is_still_acted_on(surface) -> None:
    with surface() as page:
        observation = page.observe(STRUCTURED)
        expect = expectation(observation, APPROVE)
        click(page, MOVE_APPROVE)
        result = page.act(Action(ActionKind.CLICK, APPROVE), expect=expect)
        approved = read(page, APPROVED_RECORD)

    assert expect.context != ()
    assert result.outcome is Outcome.OK
    assert approved == "Record A"


def test_a_replacement_record_at_the_same_url_receives_no_stale_action(
    surface,
) -> None:
    with surface() as page:
        observation = page.observe(STRUCTURED)
        expect = expectation(observation, APPROVE)
        click(page, NEXT_RECORD)
        result = page.act(Action(ActionKind.CLICK, APPROVE), expect=expect)
        approved = read(page, APPROVED_RECORD)
        heading = page.act(Action(ActionKind.READ, AxLocator("heading", "Record B")))

    assert heading.outcome is Outcome.OK
    assert result.outcome is Outcome.STALE
    assert approved == "none"


def test_unrelated_content_changing_does_not_refuse_the_action(surface) -> None:
    with surface() as page:
        observation = page.observe(STRUCTURED)
        expect = expectation(observation, APPROVE)
        click(page, DETACH_LEDGER)
        click(page, COVER_PADS)
        result = page.act(Action(ActionKind.CLICK, APPROVE), expect=expect)
        approved = read(page, APPROVED_RECORD)

    assert result.outcome is Outcome.OK
    assert approved == "Record A"


def test_a_record_swapped_during_approval_is_not_acted_on(
    site: str, site_profile
) -> None:
    """The approval was given for Record A. What arrives is Record B."""
    profile: Profile = site_profile(
        actions=grants(
            {"observe": "safe", "read": "safe", "click": "risky", "navigate": "safe"}
        )
    )
    journal = MemoryJournal()

    with open_session(profile, f"{site}/members/12345") as page:

        class SwapDuringApproval:
            """An operator who walks the queue on before answering."""

            def __init__(self) -> None:
                self.requests: list[InterventionRequest] = []

            def request(self, intervention: InterventionRequest) -> Handoff:
                self.requests.append(intervention)
                page.act(Action(ActionKind.CLICK, NEXT_RECORD))
                return Handoff(HandoffOutcome.APPROVED, "looks fine")

        escalator = SwapDuringApproval()
        decider = ScriptedDecider(
            decisions=[
                Observe(STRUCTURED, "read the queue"),
                Propose(Action(ActionKind.CLICK, APPROVE), "approve the record"),
            ]
        )
        result = discover(
            "approve the queued record",
            profile,
            surface=page,
            decider=decider,
            escalator=escalator,
            journal=journal,
            clock=FakeClock(),
        )
        approved = read(page, APPROVED_RECORD)

    assert escalator.requests[0].trigger is Trigger.RISKY_ACTION
    assert approved == "none"
    assert result.ending is not None


# 3. Screenshot permission follows the frame's current state.


def test_a_frame_that_replaced_its_contents_is_excluded_and_masked(surface) -> None:
    with surface() as page:
        before = page.observe(VISUAL)
        click(page, DETACH_LEDGER)
        after = page.observe(VISUAL)
        structure = page.observe(STRUCTURED)

    assert [node for node in structure.nodes if node.frame == ("ledger",)] == []
    assert after.visual is not None
    assert "a frame outside the profile" in after.visual.masked
    assert mask_pixels(after.image) > mask_pixels(before.image) + 10_000


def test_a_srcdoc_frame_is_excluded_and_masked_from_the_start(surface) -> None:
    with surface() as page:
        structure = page.observe(STRUCTURED)
        picture = page.observe(VISUAL)

    assert [node for node in structure.nodes if node.frame == ("notes",)] == []
    assert "frame(s) outside the profile were not read" in " ".join(structure.notes)
    assert picture.visual is not None
    assert "a frame outside the profile" in picture.visual.masked


# 4. Permission to look is rechecked after a person approves looking.


def test_a_denied_route_reached_during_approval_prevents_every_capture(
    edited_profile,
) -> None:
    """The gate said yes about the old screen. The screen changed while waiting."""
    profile = edited_profile(
        actions=grants({"observe": "risky", "read": "safe", "click": "safe"})
    )
    here = "https://sandbox.example.test/members/12345"
    elsewhere = "https://sandbox.example.test/admin"
    surface = ScriptedSurface(screens=[Screen(location=here)])

    class MoveDuringApproval:
        def __init__(self) -> None:
            self.requests: list[InterventionRequest] = []

        def request(self, intervention: InterventionRequest) -> Handoff:
            self.requests.append(intervention)
            surface.screens[0] = Screen(location=elsewhere)
            return Handoff(HandoffOutcome.APPROVED)

    escalator = MoveDuringApproval()
    journal = MemoryJournal()
    result = discover(
        "read the balance",
        profile,
        surface=surface,
        decider=ScriptedDecider(decisions=[Observe(VISUAL, "look at it")]),
        escalator=escalator,
        journal=journal,
        clock=FakeClock(),
    )

    assert escalator.requests != []
    assert surface.looks == []
    assert [event for event in journal.events if isinstance(event, Observed)] == []
    assert result.ending is not None


# 5. An offered anchor id comes with the picture it was cut from.


def test_every_painted_region_carries_its_own_crop(surface) -> None:
    with surface() as page:
        picture = page.observe(VISUAL)

    regions = list(picture.regions)
    assert len(regions) >= 2
    assert all(region.image is not None for region in regions)
    assert len({region.image for region in regions}) == len(regions)
    assert all(
        region.anchor_id.startswith(f"anchor-{picture.observation_id[4:]}-")
        for region in regions
    )


def test_reversing_the_canvas_order_keeps_each_id_on_its_own_control(
    surface,
) -> None:
    """The ids are re-issued, and the crop beside each one still names it."""
    with surface() as page:
        before = page.observe(VISUAL)
        crops = {region.anchor_id: region.image for region in before.regions}
        click(page, SWAP_PADS)
        after = page.observe(VISUAL)

        assert {region.anchor_id for region in after.regions}.isdisjoint(crops)
        posting = next(
            region
            for region in after.regions
            if region.image == crops[before.regions[0].anchor_id]
        )
        result = page.act(Action(ActionKind.CLICK, VisualAnchor(posting.anchor_id)))
        state = read(page, APPROVAL_STATE)

    assert result.outcome is Outcome.OK
    assert state == "posted"


def test_an_anchor_from_an_earlier_capture_is_refused(surface) -> None:
    with surface() as page:
        first = page.observe(VISUAL)
        stale = VisualAnchor(first.regions[0].anchor_id)
        page.observe(VISUAL)
        result = page.act(Action(ActionKind.CLICK, stale))
        state = read(page, APPROVAL_STATE)

    assert result.outcome is Outcome.STALE
    assert state == "pending"


def test_each_anchor_id_is_sent_to_the_model_beside_its_crop(profile) -> None:
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "name": "ask_human",
                        "arguments": json.dumps(
                            {"reason": "no_progress", "detail": ""}
                        ),
                    }
                ],
            },
        )

    observation = Observation(
        observation_id="obs-3",
        mode=ObservationMode.VISUAL,
        status=ObservationStatus.COMPLETE,
        page_state=PageState("https://sandbox.example.test/members/12345"),
        image=b"\x89PNG-viewport",
        regions=(
            VisualRegion("anchor-3-0-0-0", (), 150, 46, image=b"\x89PNG-post"),
            VisualRegion("anchor-3-0-1-0", (), 150, 46, image=b"\x89PNG-void"),
        ),
    )

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        LunaDecider(api_key="not-a-real-key", client=client).decide(
            Transcript(
                goal="post the entry",
                location=observation.location,
                allowed=profile.actions,
                permitted_modes=(ObservationMode.VISUAL,),
                observations=(observation,),
            )
        )

    content = json.loads(sent[0].content)["input"][0]["content"]
    labelled = [
        (content[index]["text"], content[index + 1]["type"])
        for index, item in enumerate(content[:-1])
        if item["type"] == "input_text" and "painted region" in item["text"]
    ]
    assert len(labelled) == 2
    assert all(kind == "input_image" for _, kind in labelled)
    assert "anchor-3-0-0-0" in labelled[0][0]
    assert "anchor-3-0-1-0" in labelled[1][0]


# 6. Observation does not scroll, and a click is computed where the canvas is.


def test_observation_leaves_the_scroll_position_alone(surface) -> None:
    with surface() as page:
        first = page.observe(VISUAL)
        page.observe(STRUCTURED)
        second = page.observe(VISUAL)

    assert first.visual is not None
    assert second.visual is not None
    assert (first.visual.scroll_x, first.visual.scroll_y) == (0, 0)
    assert (second.visual.scroll_x, second.visual.scroll_y) == (0, 0)
    assert "a canvas sits outside the viewport and was not read" in first.notes


def test_an_off_screen_canvas_is_reachable_after_a_scroll_action(surface) -> None:
    with surface() as page:
        picture = page.observe(VISUAL)
        scrolls = 0
        while scrolls < 20 and _archive(picture) is None:
            page.act(Action(ActionKind.SCROLL, value="down"))
            picture = page.observe(VISUAL)
            scrolls += 1
        archive = _archive(picture)
        assert archive is not None, "scrolling never revealed the archive canvas"
        assert picture.visual is not None
        assert picture.visual.scroll_y > 0
        result = page.act(Action(ActionKind.CLICK, VisualAnchor(archive.anchor_id)))
        state = read(page, APPROVAL_STATE)

    assert result.outcome is Outcome.OK
    assert state == "archived"


def _archive(picture: Observation) -> VisualRegion | None:
    """Return the region cut from the third canvas, the one below the fold."""
    for region in picture.regions:
        if region.anchor_id.endswith("-0-2-0"):
            return region
    return None


def test_a_canvas_scrolled_out_of_view_refuses_the_click(surface) -> None:
    with surface() as page:
        picture = page.observe(VISUAL)
        anchor = VisualAnchor(picture.regions[0].anchor_id)
        for _ in range(6):
            page.act(Action(ActionKind.SCROLL, value="down"))
        result = page.act(Action(ActionKind.CLICK, anchor))
        state = read(page, APPROVAL_STATE)

    assert result.outcome is Outcome.NOT_ACTIONABLE
    assert state == "pending"


def test_a_covered_painted_control_is_not_reported_as_clicked(surface) -> None:
    with surface() as page:
        picture = page.observe(VISUAL)
        anchor = VisualAnchor(picture.regions[0].anchor_id)
        click(page, COVER_PADS)
        result = page.act(Action(ActionKind.CLICK, anchor))
        state = read(page, APPROVAL_STATE)

    assert result.outcome is Outcome.NOT_ACTIONABLE
    assert result.detail is not None
    assert "covers" in result.detail
    assert state == "pending"


# 8. Open shadow roots are walked whether or not the host is a control.


def test_a_button_inside_a_plain_div_shadow_root_is_reported_and_operable(
    surface,
) -> None:
    with surface() as page:
        observation = page.observe(STRUCTURED)
        result = page.act(
            Action(ActionKind.CLICK, AxLocator("button", "Release the hold"))
        )
        state = read(page, HOLD_STATE)

    held = [node for node in observation.nodes if node.name == "Release the hold"]
    assert len(held) == 1
    assert held[0].shadow is True
    assert result.outcome is Outcome.OK
    assert state == "released"


def test_a_nested_open_root_is_reported_once_and_is_operable(surface) -> None:
    with surface() as page:
        observation = page.observe(STRUCTURED)
        result = page.act(
            Action(ActionKind.CLICK, AxLocator("button", "Release the nested hold"))
        )
        state = read(page, HOLD_STATE)

    nested = [
        node for node in observation.nodes if node.name == "Release the nested hold"
    ]
    assert len(nested) == 1
    assert result.outcome is Outcome.OK
    assert state == "nested release"


def test_a_custom_element_host_is_traversed(surface) -> None:
    with surface() as page:
        observation = page.observe(STRUCTURED)
        result = page.act(
            Action(ActionKind.CLICK, AxLocator("button", "Open the card"))
        )
        state = read(page, HOLD_STATE)

    assert [node.name for node in observation.nodes].count("Open the card") == 1
    assert result.outcome is Outcome.OK
    assert state == "card opened"


# 9. The dialog that is open now is answered by a declared action, once.


def answering(seen: Observation | ActionResult) -> Expectation:
    """Build what the loop sends with an answer to the dialog ``seen`` reported."""
    assert seen.dialog is not None
    return Expectation(page_state=seen.page_state, dialog=seen.dialog.dialog_id)


def test_a_dialog_is_held_open_and_reported_rather_than_dismissed(surface) -> None:
    with surface() as page:
        opened = page.act(Action(ActionKind.CLICK, POST_ENTRY))
        observation = page.observe(STRUCTURED)
        blocked = page.act(Action(ActionKind.CLICK, MOVE_APPROVE))

    assert opened.outcome is Outcome.OK
    assert "a dialog opened and is waiting for a decision" in opened.side_effects
    assert opened.dialog is not None
    assert observation.status is ObservationStatus.UNAVAILABLE
    assert observation.dialog == opened.dialog
    assert observation.dialog.message == "Post this entry to the ledger?"
    assert observation.dialog.kind == "confirm"
    assert blocked.outcome is Outcome.NOT_ACTIONABLE


def test_the_current_dialog_is_accepted_only_by_a_declared_action(surface) -> None:
    with surface(actions=grants(DIALOGS)) as page:
        opened = page.act(Action(ActionKind.CLICK, POST_ENTRY))
        answered = page.act(Action(ActionKind.ACCEPT_DIALOG), expect=answering(opened))
        state = read(page, DIALOG_STATE)

    assert answered.outcome is Outcome.OK
    assert state == "posted"


def test_an_acceptance_does_not_carry_over_to_a_later_dialog(surface) -> None:
    with surface(actions=grants(DIALOGS)) as page:
        early = page.act(Action(ActionKind.ACCEPT_DIALOG))
        page.act(Action(ActionKind.CLICK, POST_ENTRY))
        pending = page.observe(STRUCTURED)
        page.act(Action(ActionKind.DISMISS_DIALOG), expect=answering(pending))
        state = read(page, DIALOG_STATE)

    assert early.outcome is Outcome.NOT_ACTIONABLE
    assert pending.dialog is not None
    assert pending.dialog.message == "Post this entry to the ledger?"
    assert state == "not posted"


def test_an_answer_that_names_no_dialog_is_refused(surface) -> None:
    with surface(actions=grants(DIALOGS)) as page:
        page.act(Action(ActionKind.CLICK, POST_ENTRY))
        unnamed = page.act(Action(ActionKind.ACCEPT_DIALOG))
        pending = page.observe(STRUCTURED)

    assert unnamed.outcome is Outcome.NOT_ACTIONABLE
    assert pending.dialog is not None


# 10. A dialog is something to decide about, and an answer names that dialog.

REVERSE_ENTRY = DomLocator("button", DomAttribute.ID, "reverse-entry")
REVERSE_STATE = DomLocator("output", DomAttribute.ID, "reverse-state")

type Move = Callable[[Transcript], Decision]


@dataclasses.dataclass
class Moves:
    """A decider that plays moves, each free to read the transcript it is given."""

    moves: list[Move]
    seen: list[Transcript] = dataclasses.field(default_factory=list)
    task: Task = dataclasses.field(default_factory=Task)

    def interpret(self, goal: str, notices: tuple[str, ...] = ()) -> Task:
        del goal, notices
        return self.task

    def decide(self, transcript: Transcript) -> Decision:
        self.seen.append(transcript)
        if not self.moves:
            return AskHuman(Trigger.NO_PROGRESS, "the script ran out")
        return self.moves.pop(0)(transcript)


def look(_: Transcript) -> Decision:
    return Observe(STRUCTURED, "read the controls")


def see(_: Transcript) -> Decision:
    return Observe(VISUAL, "look at the screen")


def do(action: Action) -> Move:
    return lambda _: Propose(action, "next step")


def done(transcript: Transcript) -> Decision:
    """Claim the page's own heading, which the loop checks against the page.

    Every evaluation page has one top-level heading, so the claim holds on
    any of them and is checked like any other claim.
    """
    for observation in transcript.observations:
        for node in observation.nodes:
            if node.tag == "h1":
                heading = AxLocator("heading", node.name, node.frame)
                check = ResultCheck(CheckKind.RESULT, heading, node.name, output="page")
                return Finish({"page": node.name}, "the goal is met", checks=(check,))
    return Finish({}, "the goal is met")


@dataclasses.dataclass
class Ran:
    """A finished run, next to the live page it ran against."""

    result: RunResult
    decider: Moves
    journal: MemoryJournal
    page: BrowserSurface
    escalator: Escalator

    def requests(self) -> list[InterventionRequest]:
        requests = getattr(self.escalator, "requests", [])
        assert isinstance(requests, list)
        return requests

    def outcomes(self) -> list[Outcome]:
        return [
            event.outcome for event in self.journal.events if isinstance(event, Acted)
        ]

    def notices(self) -> list[str]:
        return [notice for seen in self.decider.seen for notice in seen.notices]


@pytest.fixture
def drive(site: str, site_profile):
    """Run the real loop against a real page, with moves and an operator given.

    ``moves`` receives the live page. These cases change the page while the
    model decides or a person approves, when a real page can change under a
    run.
    """

    @contextlib.contextmanager
    def _drive(
        moves: Callable[[BrowserSurface], list[Move]],
        *,
        path: str = "/members/12345",
        escalator: Callable[[BrowserSurface], Escalator] | None = None,
        clock: FakeClock | None = None,
        limits: BrowserLimits | None = None,
        **edits: object,
    ) -> Iterator[Ran]:
        profile: Profile = site_profile(**edits)
        journal = MemoryJournal()
        # A click that opens a dialog blocks the driver until its timeout, so
        # these runs use a short one.
        bounds = limits or BrowserLimits(operation_ms=1_500)
        with open_session(profile, f"{site}{path}", limits=bounds) as page:
            decider = Moves(moves(page))
            operator = escalator(page) if escalator else ScriptedEscalatorLike()
            result = discover(
                "handle the screen",
                profile,
                surface=page,
                decider=decider,
                escalator=operator,
                journal=journal,
                clock=clock or FakeClock(),
            )
            yield Ran(result, decider, journal, page, operator)

    return _drive


class ScriptedEscalatorLike:
    """An operator who never answers."""

    def __init__(self) -> None:
        self.requests: list[InterventionRequest] = []

    def request(self, intervention: InterventionRequest) -> Handoff:
        self.requests.append(intervention)
        return Handoff(HandoffOutcome.TIMED_OUT)


def answer(kind: ActionKind) -> Move:
    """Answer the dialog the transcript shows, and fail if it shows none."""

    def _answer(transcript: Transcript) -> Decision:
        assert transcript.dialog is not None, "the dialog never reached the model"
        return Propose(Action(kind), "answer the dialog shown")

    return _answer


def swap_dialog(page: BrowserSurface) -> None:
    """Answer the pending dialog as someone else would, and open another."""
    pending = page.observe(STRUCTURED)
    page.act(Action(ActionKind.DISMISS_DIALOG), expect=answering(pending))
    page.act(Action(ActionKind.CLICK, REVERSE_ENTRY))


def finish_off(page: BrowserSurface) -> tuple[str | None, str | None]:
    """Answer whatever dialog is left with a refusal, then read both outcomes."""
    pending = page.observe(STRUCTURED)
    if pending.dialog is not None:
        page.act(Action(ActionKind.DISMISS_DIALOG), expect=answering(pending))
    return read(page, DIALOG_STATE), read(page, REVERSE_STATE)


def test_the_loop_answers_a_dialog_it_opened_and_the_entry_is_posted(drive) -> None:
    moves = [
        look,
        do(Action(ActionKind.CLICK, POST_ENTRY)),
        answer(ActionKind.ACCEPT_DIALOG),
        done,
    ]

    with drive(lambda _: moves, actions=grants(DIALOGS)) as ran:
        posted = read(ran.page, DIALOG_STATE)

    deciding = ran.decider.seen[2]
    assert deciding.dialog is not None
    assert deciding.dialog.message == "Post this entry to the ledger?"
    assert deciding.observations == ()
    assert ran.outcomes() == [Outcome.OK, Outcome.OK]
    assert ran.result.ending is Ending.COMPLETED
    assert posted == "posted"


def test_the_loop_dismisses_a_dialog_and_nothing_is_posted(drive) -> None:
    moves = [
        look,
        do(Action(ActionKind.CLICK, POST_ENTRY)),
        answer(ActionKind.DISMISS_DIALOG),
        done,
    ]

    with drive(lambda _: moves, actions=grants(DIALOGS)) as ran:
        posted = read(ran.page, DIALOG_STATE)

    assert ran.result.ending is Ending.COMPLETED
    assert posted == "not posted"


def test_opening_a_dialog_does_not_complete_the_operation(drive) -> None:
    """The click was delivered. The run may not finish until the dialog is."""
    moves = [
        look,
        do(Action(ActionKind.CLICK, POST_ENTRY)),
        done,
        do(Action(ActionKind.CLICK, MOVE_APPROVE)),
        answer(ActionKind.ACCEPT_DIALOG),
        done,
    ]

    with drive(lambda _: moves, actions=grants(DIALOGS)) as ran:
        posted = read(ran.page, DIALOG_STATE)

    assert any("still waiting" in notice for notice in ran.notices())
    assert any("cannot be operated" in notice for notice in ran.notices())
    assert ran.outcomes() == [Outcome.OK, Outcome.OK]
    assert ran.result.ending is Ending.COMPLETED
    assert ran.result.steps == 6
    assert posted == "posted"


def test_an_undeclared_dialog_answer_is_refused_and_the_dialog_stays(drive) -> None:
    undeclared = {"observe": "safe", "read": "safe", "click": "safe"}
    moves = [
        look,
        do(Action(ActionKind.CLICK, POST_ENTRY)),
        answer(ActionKind.ACCEPT_DIALOG),
    ]

    with drive(lambda _: moves, actions=grants(undeclared)) as ran:
        still = ran.page.observe(STRUCTURED)

    refused = [event for event in ran.journal.events if isinstance(event, Refused)]
    assert [event.reason for event in refused] == [Denial.UNDECLARED_ACTION]
    assert ran.outcomes() == [Outcome.OK]
    assert still.dialog is not None
    assert still.dialog.message == "Post this entry to the ledger?"


class Approves:
    """An operator who approves, taking a long time about it."""

    def __init__(
        self, clock: FakeClock, during: Callable[[], object] | None = None
    ) -> None:
        self.clock = clock
        self.during = during
        self.requests: list[InterventionRequest] = []

    def request(self, intervention: InterventionRequest) -> Handoff:
        self.requests.append(intervention)
        self.clock.advance(10_000)
        if self.during is not None and len(self.requests) == 1:
            self.during()
        if len(self.requests) == 1:
            return Handoff(HandoffOutcome.APPROVED)
        return Handoff(HandoffOutcome.TIMED_OUT)


def test_a_risky_dialog_answer_is_approved_before_it_runs(drive) -> None:
    clock = FakeClock()
    operators: list[Approves] = []

    def operator(_: BrowserSurface) -> Approves:
        operators.append(Approves(clock))
        return operators[-1]

    moves = [
        look,
        do(Action(ActionKind.CLICK, POST_ENTRY)),
        answer(ActionKind.ACCEPT_DIALOG),
        done,
    ]
    risky = {**DIALOGS, "accept_dialog": "risky"}

    with drive(
        lambda _: moves, escalator=operator, clock=clock, actions=grants(risky)
    ) as ran:
        posted = read(ran.page, DIALOG_STATE)

    request = operators[0].requests[0]
    assert request.trigger is Trigger.RISKY_ACTION
    assert request.action == Action(ActionKind.ACCEPT_DIALOG)
    assert ran.result.ending is Ending.COMPLETED
    assert posted == "posted"


def test_an_approval_does_not_transfer_to_a_replacement_dialog(drive) -> None:
    """Approved for "post". What is waiting when the answer arrives is "reverse"."""
    clock = FakeClock()

    def operator(page: BrowserSurface) -> Approves:
        return Approves(clock, during=lambda: swap_dialog(page))

    moves = [
        look,
        do(Action(ActionKind.CLICK, POST_ENTRY)),
        answer(ActionKind.ACCEPT_DIALOG),
    ]
    risky = {**DIALOGS, "accept_dialog": "risky"}

    with drive(
        lambda _: moves, escalator=operator, clock=clock, actions=grants(risky)
    ) as ran:
        replacement = ran.page.observe(STRUCTURED).dialog
        posted, reversed_ = finish_off(ran.page)

    assert ran.outcomes() == [Outcome.OK, Outcome.STALE]
    assert replacement is not None
    assert replacement.message == "Reverse this entry?"
    assert posted == "not posted"
    assert reversed_ == "not reversed"


def test_an_answer_decided_before_the_dialog_changed_is_not_applied(drive) -> None:
    """The dialog is replaced while the model decides. Nobody approves anything."""

    def moves(page: BrowserSurface) -> list[Move]:
        def replace_then_accept(transcript: Transcript) -> Decision:
            assert transcript.dialog is not None
            swap_dialog(page)
            return Propose(Action(ActionKind.ACCEPT_DIALOG), "accept it")

        return [look, do(Action(ActionKind.CLICK, POST_ENTRY)), replace_then_accept]

    with drive(moves, actions=grants(DIALOGS)) as ran:
        posted, reversed_ = finish_off(ran.page)

    assert ran.outcomes() == [Outcome.OK, Outcome.STALE]
    assert posted == "not posted"
    assert reversed_ == "not reversed"


# 11. A field that received a declared secret is known by element, not by label.

BRANCH_SECRET = "synthetic-branch-passcode-4411"
BRANCH_CODE = AxLocator("textbox", "Branch approval code")
SUPERVISOR_CODE = AxLocator("textbox", "Supervisor code")
RENAME_BRANCH_CODE = DomLocator("button", DomAttribute.ID, "rename-branch-code")
REDRAW_BRANCH_CODE = DomLocator("button", DomAttribute.ID, "redraw-branch-code")
TYPE_THE_SECRET = SecretRef("approver_passcode")
TALL = BrowserLimits(viewport=(1280, 2400), operation_ms=1_500)


@pytest.fixture
def tall(site: str, site_profile, monkeypatch):
    """Open the member page in a viewport tall enough to show the code fields."""
    monkeypatch.setenv("APPROVER_PASSCODE", BRANCH_SECRET)

    def _open():
        return open_session(site_profile(), f"{site}/members/12345", limits=TALL)

    return _open


def field_share_masked(page: BrowserSurface, css: str, image: bytes | None) -> float:
    """Return the share of one field's own pixels that the mask covered.

    The field is measured on the live page, so this reads the area the field
    occupies rather than counting mask pixels anywhere in the picture.
    """
    assert image is not None
    box = page._page.locator(css).bounding_box()
    assert box is not None
    assert box["y"] + box["height"] <= TALL.viewport[1], "the field is off screen"
    left, top = int(box["x"]) + 1, int(box["y"]) + 1
    right, bottom = int(box["x"] + box["width"]) - 1, int(box["y"] + box["height"]) - 1
    with Image.open(io.BytesIO(image)) as picture:
        pixels = picture.convert("RGB").crop((left, top, right, bottom)).tobytes()
    total = len(pixels) // 3
    covered = sum(
        1
        for index in range(0, len(pixels), 3)
        if tuple(pixels[index : index + 3]) == MASK
    )
    return covered / total


def live_value(page: BrowserSurface, css: str) -> str:
    """Read what the field really holds, so a withheld value is known to exist."""
    return page._page.locator(css).input_value()


def model_payload(observations: tuple[Observation, ...], profile) -> str:
    """Return exactly what the decider would send the provider."""
    sent: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        call = {
            "type": "function_call",
            "name": "ask_human",
            "arguments": json.dumps({"reason": "no_progress", "detail": ""}),
        }
        return httpx.Response(200, json={"status": "completed", "output": [call]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        LunaDecider(api_key="not-a-real-key", client=client).decide(
            Transcript(
                goal="verify the branch",
                location=observations[0].location,
                allowed=profile.actions,
                permitted_modes=(ObservationMode.STRUCTURED, ObservationMode.VISUAL),
                observations=observations,
            )
        )
    return sent[0].content.decode()


@pytest.mark.rule(18)
def test_a_renamed_secret_field_is_still_withheld_and_masked(tall, profile) -> None:
    with tall() as page:
        before = page.observe(VISUAL)
        unmasked = field_share_masked(page, "#branch-code", before.image)
        typed = page.act(Action(ActionKind.TYPE, BRANCH_CODE, TYPE_THE_SECRET))
        click(page, RENAME_BRANCH_CODE)
        structure = page.observe(STRUCTURED)
        picture = page.observe(VISUAL)
        masked = field_share_masked(page, "#branch-code", picture.image)
        holding = live_value(page, "#branch-code")

    field = node_for(structure, "branch-code")
    assert typed.outcome is Outcome.OK
    assert holding == BRANCH_SECRET
    assert field.name == "Code on file"
    assert field.secret is True
    assert field.value is None
    assert BRANCH_SECRET not in repr(structure)
    assert unmasked < 0.1
    assert masked > 0.95
    assert picture.visual is not None
    assert "a field a declared secret was typed into" in picture.visual.masked
    assert BRANCH_SECRET not in model_payload((structure, picture), profile)


def test_a_redrawn_secret_field_refuses_observation_until_the_page_is_gone(
    tall,
) -> None:
    """The value moved into a new element. Nothing is read until the page goes."""
    with tall() as page:
        page.act(Action(ActionKind.TYPE, BRANCH_CODE, TYPE_THE_SECRET))
        click(page, REDRAW_BRANCH_CODE)
        holding = live_value(page, "#code-on-file")
        structure = page.observe(STRUCTURED)
        picture = page.observe(VISUAL)
        here = page.location()
        page.act(Action(ActionKind.NAVIGATE, destination=here))
        reloaded = page.observe(STRUCTURED)

    assert holding == BRANCH_SECRET
    assert structure.status is ObservationStatus.FAILED
    assert structure.nodes == ()
    assert BRANCH_SECRET not in repr(structure)
    assert picture.status is ObservationStatus.FAILED
    assert picture.image is None
    assert reloaded.usable
    assert node_for(reloaded, "branch-code").value == ""
    assert BRANCH_SECRET not in repr(reloaded)


def test_an_interrupted_secret_fill_leaves_the_field_protected(tall) -> None:
    """The page stops the fill with a dialog after the value has landed."""
    with tall() as page:
        typed = page.act(Action(ActionKind.TYPE, SUPERVISOR_CODE, TYPE_THE_SECRET))
        page.act(Action(ActionKind.DISMISS_DIALOG), expect=answering(typed))
        holding = live_value(page, "#supervisor-code")
        structure = page.observe(STRUCTURED)
        picture = page.observe(VISUAL)
        masked = field_share_masked(page, "#supervisor-code", picture.image)

    field = node_for(structure, "supervisor-code")
    assert typed.dialog is not None
    assert holding == BRANCH_SECRET
    assert field.secret is True
    assert field.value is None
    assert BRANCH_SECRET not in repr(structure)
    assert masked > 0.95


# 12. An action on a record proves it is still that record, from the page.

MEMBER_NUMBER = DomLocator("dd", DomAttribute.ID, "member-number")
APPROVED_MEMBER = DomLocator("output", DomAttribute.ID, "approved-member")
NEXT_MEMBER = DomLocator("button", DomAttribute.ID, "next-member")
CHANGE_NOTE = DomLocator("button", DomAttribute.ID, "change-note")
REPEAT_NUMBER = DomLocator("button", DomAttribute.ID, "repeat-number")
DROP_NUMBER = DomLocator("button", DomAttribute.ID, "drop-number")
QUEUES = ["/queue", "/queue/bare"]


def member(value: str) -> RecordEvidence:
    return RecordEvidence(MEMBER_NUMBER, value, Relation.CONTAINER)


def approve(value: str) -> Action:
    return Action(ActionKind.CLICK, APPROVE, evidence=member(value))


def then(page: BrowserSurface, *changes: DomLocator, move: Move) -> Move:
    """Change the page while the model decides, then make ``move``."""

    def _move(transcript: Transcript) -> Decision:
        for change in changes:
            assert click(page, change) is Outcome.OK
        return move(transcript)

    return _move


def approve_context(observation: Observation) -> tuple[str, ...]:
    return expectation(observation, APPROVE).context


@pytest.mark.parametrize("path", QUEUES)
def test_a_member_swapped_under_the_same_heading_is_not_approved(
    drive, path: str
) -> None:
    """Member 10001 is decided on. Member 10002 is on screen when it runs."""
    contexts: list[tuple[str, ...]] = []

    def moves(page: BrowserSurface) -> list[Move]:
        def swap(transcript: Transcript) -> Decision:
            contexts.append(approve_context(transcript.observations[0]))
            click(page, NEXT_MEMBER)
            contexts.append(approve_context(page.observe(STRUCTURED)))
            return Propose(approve("10001"), "approve member 10001")

        return [look, swap]

    with drive(moves, path=path) as ran:
        approved = read(ran.page, APPROVED_MEMBER)
        showing = read(ran.page, MEMBER_NUMBER)
        written = repr([as_record(event) for event in ran.journal.events])

    assert contexts[0] == contexts[1], "the old check alone would have passed"
    if path == "/queue/bare":
        assert contexts[0] == ()
    assert showing == "10002"
    assert ran.outcomes() == [Outcome.STALE]
    assert approved == "none"
    assert "10001" not in written
    assert "10002" not in written


@pytest.mark.parametrize("path", QUEUES)
def test_the_same_member_is_approved_after_the_layout_changes(drive, path) -> None:
    def moves(page: BrowserSurface) -> list[Move]:
        change = then(page, MOVE_APPROVE, CHANGE_NOTE, move=do(approve("10001")))
        return [look, change]

    with drive(moves, path=path) as ran:
        approved = read(ran.page, APPROVED_MEMBER)

    assert ran.outcomes() == [Outcome.OK]
    assert approved == "10001"


def test_an_action_without_record_evidence_is_refused_by_the_gate(drive) -> None:
    moves = [look, do(Action(ActionKind.CLICK, APPROVE))]

    with drive(lambda _: moves, path="/queue") as ran:
        approved = read(ran.page, APPROVED_MEMBER)

    refused = [event for event in ran.journal.events if isinstance(event, Refused)]
    assert [event.reason for event in refused] == [Denial.RECORD_EVIDENCE_MISSING]
    assert ran.outcomes() == []
    assert approved == "none"


def test_evidence_the_observation_never_showed_is_refused(drive) -> None:
    """The model names member 10002. The screen it looked at showed 10001."""
    moves = [look, do(approve("10002"))]

    with drive(lambda _: moves, path="/queue") as ran:
        approved = read(ran.page, APPROVED_MEMBER)

    assert any("not a value the observation showed" in n for n in ran.notices())
    assert ran.outcomes() == []
    assert approved == "none"


def test_evidence_outside_the_targets_container_is_refused(drive) -> None:
    unrelated = RecordEvidence(APPROVED_MEMBER, "none", Relation.CONTAINER)
    moves = [look, do(Action(ActionKind.CLICK, APPROVE, evidence=unrelated))]

    with drive(lambda _: moves, path="/queue") as ran:
        approved = read(ran.page, APPROVED_MEMBER)

    assert any("share no container" in n for n in ran.notices())
    assert ran.outcomes() == []
    assert approved == "none"


@pytest.mark.parametrize(
    ("change", "outcome"),
    [(DROP_NUMBER, Outcome.STALE), (REPEAT_NUMBER, Outcome.AMBIGUOUS)],
)
def test_evidence_that_vanished_or_repeated_refuses_the_action(
    drive, change: DomLocator, outcome: Outcome
) -> None:
    def moves(page: BrowserSurface) -> list[Move]:
        return [look, then(page, change, move=do(approve("10001")))]

    with drive(moves, path="/queue") as ran:
        approved = read(ran.page, APPROVED_MEMBER)

    assert ran.outcomes() == [outcome]
    assert approved == "none"


def test_a_member_swapped_during_approval_is_not_approved(drive) -> None:
    clock = FakeClock()

    def operator(page: BrowserSurface) -> Approves:
        return Approves(clock, during=lambda: click(page, NEXT_MEMBER))

    risky = {"observe": "safe", "read": "safe", "click": "risky"}

    with drive(
        lambda _: [look, do(approve("10001"))],
        path="/queue",
        escalator=operator,
        clock=clock,
        actions=grants(risky),
    ) as ran:
        approved = read(ran.page, APPROVED_MEMBER)

    assert ran.outcomes() == [Outcome.STALE]
    assert approved == "none"


def painted_approve(value: str) -> Move:
    def _move(transcript: Transcript) -> Decision:
        regions = [
            region
            for observation in transcript.observations
            for region in observation.regions
        ]
        assert len(regions) == 1
        target = VisualAnchor(regions[0].anchor_id)
        return Propose(
            Action(ActionKind.CLICK, target, evidence=member(value)), "approve"
        )

    return _move


@pytest.mark.parametrize(("swapped", "expected"), [(False, "10001"), (True, "none")])
def test_a_painted_approve_checks_the_member_too(
    drive, swapped: bool, expected: str
) -> None:
    """The crop matches either way. Only the member number says which record."""

    def moves(page: BrowserSurface) -> list[Move]:
        changes = (NEXT_MEMBER,) if swapped else ()
        return [look, see, then(page, *changes, move=painted_approve("10001"))]

    with drive(moves, path="/queue") as ran:
        approved = read(ran.page, APPROVED_MEMBER)

    assert ran.outcomes() == [Outcome.STALE if swapped else Outcome.OK]
    assert approved == expected


def test_a_visual_only_screen_can_hand_over_at_once(drive) -> None:
    """No structured observation means no checked evidence, so a person is asked."""
    operators: list[ScriptedEscalatorLike] = []

    def operator(_: BrowserSurface) -> ScriptedEscalatorLike:
        operators.append(ScriptedEscalatorLike())
        return operators[-1]

    moves: list[Move] = [
        see,
        painted_approve("10001"),
        lambda _: AskHuman(Trigger.INSUFFICIENT_OBSERVATION, "no member number"),
    ]

    with drive(lambda _: moves, path="/queue", escalator=operator) as ran:
        approved = read(ran.page, APPROVED_MEMBER)

    assert any("structured observation" in n for n in ran.notices())
    assert operators[0].requests[0].trigger is Trigger.INSUFFICIENT_OBSERVATION
    assert operators[0].requests[0].observed_modes == (ObservationMode.VISUAL,)
    assert ran.outcomes() == []
    assert approved == "none"


def test_scrolling_a_record_screen_needs_no_evidence(drive) -> None:
    moves = [look, do(Action(ActionKind.SCROLL, value="down"))]

    with drive(lambda _: moves, path="/queue") as ran:
        refused = [e for e in ran.journal.events if isinstance(e, Refused)]

    assert refused == []
    assert ran.outcomes() == [Outcome.OK]


@pytest.mark.parametrize(
    ("row", "outcome", "landed"),
    [("12345", Outcome.OK, "/members/12345"), ("67890", Outcome.AMBIGUOUS, "/members")],
)
def test_row_evidence_must_sit_in_the_targets_row(
    surface, row: str, outcome: Outcome, landed: str
) -> None:
    open_row = AxLocator("button", "Open", scope=Scope(ScopeKind.ROW, "12345"))
    evidence = RecordEvidence(
        DomLocator("th", DomAttribute.TEXT, row), row, Relation.ROW
    )

    with surface("/members") as page:
        result = page.act(Action(ActionKind.CLICK, open_row, evidence=evidence))
        where = page.location()

    assert result.outcome is outcome
    assert where.endswith(landed)


# 13. An action says what the browser does and what that accomplishes, and the
# model may add restrictions but never lift one.

AMOUNT = AxLocator("textbox", "Amount")
MEMO = AxLocator("textbox", "Memo")
LOOKUP = AxLocator("textbox", "Find a payee")
PAYEE = DomLocator("dd", DomAttribute.ID, "payee-number")
PAYMENTS_SENT = DomLocator("output", DomAttribute.ID, "payments-sent")
SEARCHES = DomLocator("output", DomAttribute.ID, "searches")
NEXT_PAYEE = DomLocator("button", DomAttribute.ID, "next-payee")
TRAP_FOCUS = DomLocator("button", DomAttribute.ID, "trap-focus")
NO_RECORDS = {"actions": [], "routes": []}
PAYING = {
    "observe": "safe",
    "read": "safe",
    "type": "safe",
    "click": "safe",
    "press_key": "safe",
}


def paying(**effects: str) -> dict[str, dict[str, object]]:
    """Grant every paying action, with these exceptions on key presses."""
    actions = grants(PAYING)
    if effects:
        actions["press_key"] = {"any": "safe", "effects": effects}
    return actions


def enter(
    target: AxLocator | None,
    effect: str,
    *,
    flag: bool = False,
    payee: str | None = None,
) -> Action:
    """Press Enter into ``target``, or into whatever has focus when None."""
    evidence = None
    if payee is not None:
        evidence = RecordEvidence(PAYEE, payee, Relation.CONTAINER)
    return Action(
        ActionKind.PRESS_KEY,
        target,
        "Enter",
        evidence=evidence,
        effect=effect,
        flag_risky=flag,
    )


TYPE_AMOUNT = Action(ActionKind.TYPE, AMOUNT, "12.00", effect="enter_amount")


class Rejects:
    """An operator who refuses every request and keeps a list of them."""

    def __init__(self) -> None:
        self.requests: list[InterventionRequest] = []

    def request(self, intervention: InterventionRequest) -> Handoff:
        self.requests.append(intervention)
        return Handoff(HandoffOutcome.REJECTED, "not this one")


def rejecting(_: BrowserSurface) -> Rejects:
    return Rejects()


def sent_and_searched(page: BrowserSurface) -> tuple[str | None, str | None]:
    return read(page, PAYMENTS_SENT), read(page, SEARCHES)


def refusals(ran: Ran) -> list[Denial]:
    return [event.reason for event in ran.journal.events if isinstance(event, Refused)]


def test_a_wildcard_grant_with_a_risky_exception_escalates_only_that_effect(
    drive,
) -> None:
    moves = [
        look,
        do(TYPE_AMOUNT),
        do(enter(LOOKUP, "search")),
        do(enter(AMOUNT, "submit_payment")),
    ]

    with drive(
        lambda _: moves,
        path="/payments",
        escalator=rejecting,
        actions=paying(submit_payment="risky"),
        records=NO_RECORDS,
    ) as ran:
        sent, searched = sent_and_searched(ran.page)

    risky = [r for r in ran.requests() if r.trigger is Trigger.RISKY_ACTION]
    assert [r.action for r in risky] == [enter(AMOUNT, "submit_payment")]
    assert searched == "1"
    assert sent == "none"


def test_an_effect_the_operator_denies_stays_denied_under_another_name(
    drive,
) -> None:
    moves = [
        look,
        do(TYPE_AMOUNT),
        do(enter(AMOUNT, "submit_payment")),
        do(enter(AMOUNT, "save_amount")),
        do(Action(ActionKind.PRESS_KEY, AMOUNT, "Enter")),
    ]

    with drive(
        lambda _: moves,
        path="/payments",
        actions=paying(submit_payment="deny"),
        records=NO_RECORDS,
    ) as ran:
        sent, _ = sent_and_searched(ran.page)

    assert refusals(ran) == [
        Denial.EFFECT_DENIED,
        Denial.EFFECT_DENIED,
        Denial.EFFECT_MISSING,
    ]
    assert sent == "none"


def test_an_action_type_the_operator_never_granted_stays_denied(drive) -> None:
    ungranted = {k: v for k, v in PAYING.items() if k != "press_key"}
    moves = [look, do(TYPE_AMOUNT), do(enter(AMOUNT, "save", flag=True))]

    with drive(
        lambda _: moves,
        path="/payments",
        actions=grants(ungranted),
        records=NO_RECORDS,
    ) as ran:
        sent, _ = sent_and_searched(ran.page)

    assert refusals(ran) == [Denial.UNDECLARED_ACTION]
    assert ran.requests()[0].trigger is Trigger.NO_PROGRESS
    assert sent == "none"


def test_a_model_flag_reaches_a_person_before_the_key_is_sent(drive) -> None:
    """The model says so first. The operator never declared this effect."""
    moves = [
        look,
        do(TYPE_AMOUNT),
        do(enter(AMOUNT, "submit_payment", flag=True)),
        do(enter(AMOUNT, "update_amount")),
        do(enter(MEMO, "add_memo")),
    ]

    with drive(
        lambda _: moves,
        path="/payments",
        escalator=rejecting,
        actions=paying(),
        records=NO_RECORDS,
    ) as ran:
        sent, _ = sent_and_searched(ran.page)

    flagged = [e for e in ran.journal.events if isinstance(e, Flagged)]
    assert [e.source for e in flagged] == [Source.PROPOSAL]
    assert [r.trigger for r in ran.requests()[:3]] == [Trigger.RISKY_ACTION] * 3
    assert [e.kind for e in ran.journal.events if isinstance(e, Acted)] == [
        ActionKind.TYPE
    ]
    assert sent == "none"


def the_step_that_pressed(transcript: Transcript) -> int:
    return next(
        turn.step
        for turn in transcript.history
        if turn.action is not None and turn.action.kind is ActionKind.PRESS_KEY
    )


def test_a_finding_after_the_fact_restricts_every_later_attempt(drive) -> None:
    """Enter posted a payment. After the finding, no relabelling gets it again."""
    moves: list[Move] = [
        look,
        do(TYPE_AMOUNT),
        do(enter(AMOUNT, "save_form")),
        lambda seen: FlagRisk(
            RiskFinding(the_step_that_pressed(seen), "submit_payment", "it posted")
        ),
        do(enter(AMOUNT, "save_form")),
        do(enter(MEMO, "add_memo")),
        do(Action(ActionKind.PRESS_KEY, None, "Enter", effect="confirm")),
        done,
    ]

    with drive(
        lambda _: moves,
        path="/payments",
        escalator=rejecting,
        actions=paying(),
        records=NO_RECORDS,
    ) as ran:
        sent, _ = sent_and_searched(ran.page)

    finding = [r for r in ran.result.restrictions if r.source is Source.FINDING]
    assert len(finding) == 1
    assert finding[0].step == 3
    assert finding[0].effect == "submit_payment"
    # Each later attempt asked and was declined. Three declines in a row are
    # no progress, so the run then hands over to a person (rule 17).
    triggers = [r.trigger for r in ran.requests()]
    assert triggers == [Trigger.RISKY_ACTION] * 3 + [Trigger.NO_PROGRESS]
    assert sent == "10001"
    assert "it posted" not in repr([as_record(e) for e in ran.journal.events])


@pytest.mark.parametrize("step", [99, 1])
def test_a_finding_about_a_step_that_took_no_action_is_refused(drive, step) -> None:
    moves: list[Move] = [
        look,
        lambda _: FlagRisk(RiskFinding(step, "submit_payment", "guess")),
        do(TYPE_AMOUNT),
        do(enter(AMOUNT, "save_form")),
        done,
    ]

    with drive(
        lambda _: moves,
        path="/payments",
        actions=paying(),
        records=NO_RECORDS,
    ) as ran:
        sent, _ = sent_and_searched(ran.page)

    assert any(f"step {step} performed no action" in n for n in ran.notices())
    assert ran.result.restrictions == ()
    assert sent == "10001"


def test_a_key_is_not_sent_when_the_page_moves_focus_elsewhere(drive) -> None:
    def moves(page: BrowserSurface) -> list[Move]:
        send = do(enter(AMOUNT, "submit_payment", payee="10001"))
        return [look, do(TYPE_AMOUNT), then(page, TRAP_FOCUS, move=send)]

    with drive(moves, path="/payments") as ran:
        sent, searched = sent_and_searched(ran.page)

    assert ran.outcomes() == [Outcome.OK, Outcome.NOT_ACTIONABLE]
    assert sent == "none"
    assert searched == "0"


@pytest.mark.parametrize(("swapped", "expected"), [(False, "10001"), (True, "none")])
def test_a_submission_key_checks_the_payee_first(drive, swapped, expected) -> None:
    def moves(page: BrowserSurface) -> list[Move]:
        send = do(enter(AMOUNT, "submit_payment", payee="10001"))
        changes = (NEXT_PAYEE,) if swapped else ()
        return [look, do(TYPE_AMOUNT), then(page, *changes, move=send)]

    with drive(moves, path="/payments") as ran:
        sent, _ = sent_and_searched(ran.page)

    assert ran.outcomes() == [Outcome.OK, Outcome.STALE if swapped else Outcome.OK]
    assert sent == expected


def test_an_untargeted_key_on_a_record_screen_is_refused(drive) -> None:
    moves = [look, do(Action(ActionKind.PRESS_KEY, None, "Enter", effect="submit"))]

    with drive(lambda _: moves, path="/payments") as ran:
        sent, _ = sent_and_searched(ran.page)

    assert refusals(ran) == [Denial.RECORD_EVIDENCE_MISSING]
    assert sent == "none"


# 14. One member's number cannot vouch for another member's control.

CARD_NUMBER_A = DomLocator("dd", DomAttribute.ID, "card-number-a")
CARD_NUMBER_B = DomLocator("dd", DomAttribute.ID, "card-number-b")
CARD_APPROVE_B = DomLocator("button", DomAttribute.ID, "card-approve-b")
FLAT_NUMBER_A = DomLocator("dd", DomAttribute.ID, "flat-number-a")
FLAT_NUMBER_B = DomLocator("dd", DomAttribute.ID, "flat-number-b")
FLAT_APPROVE_B = DomLocator("button", DomAttribute.ID, "flat-approve-b")
REDRAW_CARDS = DomLocator("button", DomAttribute.ID, "redraw-cards")


def approve_with(
    target: DomLocator, source: DomLocator, value: str, relation: Relation
) -> Action:
    return Action(
        ActionKind.CLICK,
        target,
        evidence=RecordEvidence(source, value, relation),
        effect="approve_member",
    )


def test_member_a_cannot_approve_member_b_in_a_shared_section(drive) -> None:
    """Both cards sit in one section. The section is not either member's record."""
    wrong = approve_with(CARD_APPROVE_B, CARD_NUMBER_A, "10001", Relation.CONTAINER)
    moves: list[Move] = [
        look,
        do(wrong),
        lambda _: AskHuman(Trigger.AMBIGUOUS_TARGET, "which member is this"),
    ]

    with drive(lambda _: moves, path="/queue/shared") as ran:
        approved = read(ran.page, APPROVED_MEMBER)
        direct = ran.page.act(wrong)
        still = read(ran.page, APPROVED_MEMBER)

    assert any("holds 2 controls like the target" in n for n in ran.notices())
    assert ran.outcomes() == []
    assert ran.requests()[0].trigger is Trigger.AMBIGUOUS_TARGET
    assert approved == "none"
    assert direct.outcome is Outcome.AMBIGUOUS
    assert still == "none"


def test_a_members_own_card_still_vouches_after_the_cards_are_redrawn(
    drive,
) -> None:
    right = approve_with(CARD_APPROVE_B, CARD_NUMBER_B, "10002", Relation.CONTAINER)

    def moves(page: BrowserSurface) -> list[Move]:
        return [look, then(page, REDRAW_CARDS, move=do(right))]

    with drive(moves, path="/queue/shared") as ran:
        approved = read(ran.page, APPROVED_MEMBER)

    assert ran.outcomes() == [Outcome.OK]
    assert approved == "10002"


@pytest.mark.parametrize(
    ("source", "value", "relation", "approved"),
    [
        (FLAT_NUMBER_B, "10004", Relation.CONTAINER, "none"),
        (FLAT_NUMBER_A, "10003", Relation.LABELLED, "none"),
        (FLAT_NUMBER_B, "10004", Relation.LABELLED, "10004"),
    ],
)
def test_a_flat_list_binds_a_member_only_through_the_pages_own_reference(
    surface, source, value, relation, approved
) -> None:
    """Without a card, only the button's own aria-describedby ties it to a member."""
    with surface("/queue/shared") as page:
        result = page.act(approve_with(FLAT_APPROVE_B, source, value, relation))
        seen = read(page, APPROVED_MEMBER)

    assert result.outcome is (Outcome.OK if approved != "none" else Outcome.AMBIGUOUS)
    assert seen == approved


def card_pad(transcript: Transcript, card: int) -> VisualAnchor:
    """Return the painted Approve cut from the ``card``-th canvas on the page."""
    regions = [
        region
        for observation in transcript.observations
        for region in observation.regions
        if region.anchor_id.endswith(f"-0-{card}-0")
    ]
    assert len(regions) == 1
    return VisualAnchor(regions[0].anchor_id)


@pytest.mark.parametrize(
    ("source", "value", "outcome", "approved"),
    [
        (CARD_NUMBER_A, "10001", None, "none"),
        (CARD_NUMBER_B, "10002", Outcome.OK, "10002"),
    ],
)
def test_a_painted_approve_is_bound_to_its_own_card_only(
    drive, source, value, outcome, approved
) -> None:
    def paint(transcript: Transcript) -> Decision:
        evidence = RecordEvidence(source, value, Relation.CONTAINER)
        action = Action(
            ActionKind.CLICK,
            card_pad(transcript, 1),
            evidence=evidence,
            effect="approve_member",
        )
        return Propose(action, "approve the second card")

    with drive(lambda _: [look, see, paint], path="/queue/shared") as ran:
        seen = read(ran.page, APPROVED_MEMBER)

    assert ran.outcomes() == ([outcome] if outcome else [Outcome.AMBIGUOUS])
    assert seen == approved


# 15. A record that changes while the driver waits to click gets no click.

DELAYED_SWAP = DomLocator("button", DomAttribute.ID, "delayed-swap")
SWAP_ON_PRESS = DomLocator("button", DomAttribute.ID, "swap-on-press")


def test_a_member_swapped_while_the_click_waits_is_not_approved(drive) -> None:
    """Covered at 0 ms, member 10002 at 600 ms, uncovered at 1400 ms."""

    def moves(page: BrowserSurface) -> list[Move]:
        return [look, then(page, DELAYED_SWAP, move=do(approve("10001")))]

    patient = BrowserLimits(operation_ms=5_000)
    with drive(moves, path="/queue", limits=patient) as ran:
        approved = read(ran.page, APPROVED_MEMBER)
        showing = read(ran.page, MEMBER_NUMBER)

    assert showing == "10002"
    assert ran.outcomes() == [Outcome.STALE]
    assert approved == "none"


def test_a_member_swapped_as_the_click_arrives_is_not_approved(drive) -> None:
    """Every check before the click passes. The member changes during it."""

    def moves(page: BrowserSurface) -> list[Move]:
        return [look, then(page, SWAP_ON_PRESS, move=do(approve("10001")))]

    with drive(moves, path="/queue") as ran:
        approved = read(ran.page, APPROVED_MEMBER)
        showing = read(ran.page, MEMBER_NUMBER)

    assert showing == "10002"
    assert ran.outcomes() == [Outcome.STALE]
    assert approved == "none"


# 16. A restriction follows the control and the submission, not the locator or
# the physical action.

SEND_BY_NAME = AxLocator("button", "Send payment")
SEND_BY_TYPE = DomLocator("button", DomAttribute.TYPE, "submit")
SAVE_DRAFT = AxLocator("button", "Save draft")
PAY_NOW = AxLocator("button", "Pay now")
REDRAW_SEND = DomLocator("button", DomAttribute.ID, "redraw-send")
TRANSFER = AxLocator("button", "Send transfer")
TRANSFER_AMOUNT = AxLocator("textbox", "Transfer amount")
DRAFTS = DomLocator("output", DomAttribute.ID, "drafts")
TRANSFERS = DomLocator("output", DomAttribute.ID, "transfers")


def press(
    target: AxLocator | DomLocator,
    effect: str,
    *,
    flag: bool = False,
    payee: str | None = None,
) -> Action:
    """Click ``target``, with record evidence when a payee is named."""
    evidence = None
    if payee is not None:
        evidence = RecordEvidence(PAYEE, payee, Relation.CONTAINER)
    return Action(
        ActionKind.CLICK, target, evidence=evidence, effect=effect, flag_risky=flag
    )


def risky_requests(ran: Ran) -> list[InterventionRequest]:
    return [r for r in ran.requests() if r.trigger is Trigger.RISKY_ACTION]


def counters(page: BrowserSurface) -> dict[str, str | None]:
    return {
        "sent": read(page, PAYMENTS_SENT),
        "drafts": read(page, DRAFTS),
        "searches": read(page, SEARCHES),
        "transfers": read(page, TRANSFERS),
    }


class Approves_All:  # noqa: N801  named for what it does in a test
    """An operator who approves everything, which must not reach a denial."""

    def __init__(self) -> None:
        self.requests: list[InterventionRequest] = []

    def request(self, intervention: InterventionRequest) -> Handoff:
        self.requests.append(intervention)
        return Handoff(HandoffOutcome.APPROVED)


@pytest.mark.parametrize(
    ("first", "second"),
    [(SEND_BY_NAME, SEND_BY_TYPE), (SEND_BY_TYPE, SEND_BY_NAME)],
    ids=["accessible-then-dom", "dom-then-accessible"],
)
def test_a_rejected_click_stays_restricted_under_another_locator(
    drive, first, second
) -> None:
    """Same button, new locator, new label, no flag: still a person's call."""
    moves = [
        look,
        do(TYPE_AMOUNT),
        do(press(first, "submit_payment", flag=True, payee="10001")),
        look,
        see,
        do(press(second, "confirm", payee="10001")),
        do(press(first, "submit_payment", payee="10001")),
    ]

    with drive(lambda _: moves, path="/payments", escalator=rejecting) as ran:
        seen = counters(ran.page)

    assert len(risky_requests(ran)) == 3
    assert ran.outcomes() == [Outcome.OK]
    assert seen["sent"] == "none"


def test_a_bound_denial_holds_under_another_locator_and_an_approver(
    drive,
) -> None:
    actions = grants(PAYING)
    actions["click"] = {"any": "safe", "effects": {"submit_payment": "deny"}}
    moves = [
        look,
        do(TYPE_AMOUNT),
        do(press(SEND_BY_NAME, "submit_payment", payee="10001")),
        do(press(SEND_BY_TYPE, "confirm", payee="10001")),
        do(press(SEND_BY_TYPE, "confirm", flag=True, payee="10001")),
    ]
    operators: list[Approves_All] = []

    def approver(_: BrowserSurface) -> Approves_All:
        operators.append(Approves_All())
        return operators[-1]

    with drive(
        lambda _: moves, path="/payments", escalator=approver, actions=actions
    ) as ran:
        seen = counters(ran.page)

    assert refusals(ran) == [Denial.EFFECT_DENIED] * 3
    assert all(r.trigger is Trigger.NO_PROGRESS for r in operators[0].requests)
    assert ran.outcomes() == [Outcome.OK]
    assert seen["sent"] == "none"


def test_a_rejected_click_restricts_enter_in_the_same_form(drive) -> None:
    moves = [
        look,
        do(TYPE_AMOUNT),
        do(press(SEND_BY_NAME, "submit_payment", flag=True, payee="10001")),
        do(enter(AMOUNT, "submit_payment", payee="10001")),
        do(enter(MEMO, "confirm", payee="10001")),
    ]

    with drive(lambda _: moves, path="/payments", escalator=rejecting) as ran:
        seen = counters(ran.page)

    assert len(risky_requests(ran)) == 3
    assert seen["sent"] == "none"


def test_a_rejected_enter_restricts_the_submit_button(drive) -> None:
    moves = [
        look,
        do(TYPE_AMOUNT),
        do(enter(AMOUNT, "submit_payment", flag=True, payee="10001")),
        do(press(SEND_BY_NAME, "confirm", payee="10001")),
        do(press(SEND_BY_TYPE, "confirm", payee="10001")),
    ]

    with drive(lambda _: moves, path="/payments", escalator=rejecting) as ran:
        seen = counters(ran.page)

    assert len(risky_requests(ran)) == 3
    assert seen["sent"] == "none"


def test_a_finding_about_a_click_protects_enter_and_other_locators(drive) -> None:
    """The click posted once. After the finding, no other path posts again."""

    def the_click(seen: Transcript) -> int:
        return next(
            turn.step
            for turn in seen.history
            if turn.action is not None and turn.action.kind is ActionKind.CLICK
        )

    moves: list[Move] = [
        look,
        do(TYPE_AMOUNT),
        do(press(SEND_BY_NAME, "save_form", payee="10001")),
        lambda seen: FlagRisk(RiskFinding(the_click(seen), "submit_payment", "posted")),
        do(enter(AMOUNT, "confirm", payee="10001")),
        do(press(SEND_BY_TYPE, "confirm", payee="10001")),
    ]

    with drive(lambda _: moves, path="/payments", escalator=rejecting) as ran:
        seen = counters(ran.page)

    assert len(risky_requests(ran)) == 2
    assert [r.source for r in ran.result.restrictions] == [Source.FINDING]
    assert seen["sent"] == "10001"


def test_other_controls_and_forms_stay_usable_under_a_restriction(drive) -> None:
    """Editing, Tab, another submit button, and another form are not the payment."""
    moves = [
        look,
        do(TYPE_AMOUNT),
        do(press(SEND_BY_NAME, "submit_payment", flag=True)),
        do(Action(ActionKind.PRESS_KEY, AMOUNT, "Tab", effect="next_field")),
        do(press(SAVE_DRAFT, "save_draft")),
        do(Action(ActionKind.TYPE, LOOKUP, "Riverside", effect="enter_search")),
        do(enter(LOOKUP, "search")),
        done,
    ]

    with drive(
        lambda _: moves, path="/payments", escalator=rejecting, records=NO_RECORDS
    ) as ran:
        seen = counters(ran.page)

    assert len(risky_requests(ran)) == 1
    assert ran.outcomes() == [Outcome.OK] * 5
    assert seen == {"sent": "none", "drafts": "1", "searches": "1", "transfers": "0"}


def test_an_enter_whose_submission_is_not_native_goes_to_a_person(drive) -> None:
    """The page sends the transfer by script, so Enter cannot be proven different."""
    moves = [
        look,
        do(press(TRANSFER, "send_transfer", flag=True)),
        do(Action(ActionKind.TYPE, TRANSFER_AMOUNT, "5.00", effect="enter_amount")),
        do(enter(TRANSFER_AMOUNT, "save_transfer")),
    ]

    with drive(
        lambda _: moves, path="/payments", escalator=rejecting, records=NO_RECORDS
    ) as ran:
        seen = counters(ran.page)

    assert len(risky_requests(ran)) == 2
    assert ran.outcomes() == [Outcome.OK]
    assert seen["transfers"] == "0"


def test_a_restricted_button_drawn_again_under_a_new_name_goes_to_a_person(
    drive,
) -> None:
    def moves(page: BrowserSurface) -> list[Move]:
        return [
            look,
            do(TYPE_AMOUNT),
            do(press(SEND_BY_NAME, "submit_payment", flag=True, payee="10001")),
            then(page, REDRAW_SEND, move=look),
            do(press(PAY_NOW, "pay", payee="10001")),
        ]

    with drive(moves, path="/payments", escalator=rejecting) as ran:
        seen = counters(ran.page)

    assert len(risky_requests(ran)) == 2
    assert seen["sent"] == "none"


# 17. A restriction reaches the same submission after a reload or a redraw.

REDRAW_SEND_SAME = DomLocator("button", DomAttribute.ID, "redraw-send-same")
REBUILD_FORM = DomLocator("button", DomAttribute.ID, "rebuild-form")
REBUILD_RENAMED = DomLocator("button", DomAttribute.ID, "rebuild-form-renamed")
RELOADING = grants({**PAYING, "navigate": "safe"})


def reload(page: BrowserSurface) -> Move:
    """Load the page the run is on again, as a proposal the gate sees."""
    return lambda _: Propose(
        Action(ActionKind.NAVIGATE, destination=page.location(), effect="reload"),
        "load the page again",
    )


def approving(_: BrowserSurface) -> Approves_All:
    return Approves_All()


def test_a_rejected_click_still_restricts_enter_after_a_reload(drive) -> None:
    def moves(page: BrowserSurface) -> list[Move]:
        return [
            look,
            do(TYPE_AMOUNT),
            do(press(SEND_BY_NAME, "submit_payment", flag=True, payee="10001")),
            reload(page),
            do(enter(AMOUNT, "confirm", payee="10001")),
            do(press(SEND_BY_TYPE, "save_form", payee="10001")),
        ]

    with drive(moves, path="/payments", escalator=rejecting, actions=RELOADING) as ran:
        seen = counters(ran.page)

    assert len(risky_requests(ran)) == 3
    assert ran.outcomes() == [Outcome.OK, Outcome.OK]
    assert seen["sent"] == "none"


def test_a_rejected_click_still_restricts_enter_after_a_same_name_redraw(
    drive,
) -> None:
    def moves(page: BrowserSurface) -> list[Move]:
        return [
            look,
            do(TYPE_AMOUNT),
            do(press(SEND_BY_NAME, "submit_payment", flag=True, payee="10001")),
            then(page, REDRAW_SEND_SAME, move=look),
            do(enter(AMOUNT, "confirm", payee="10001")),
        ]

    with drive(moves, path="/payments", escalator=rejecting) as ran:
        seen = counters(ran.page)

    assert len(risky_requests(ran)) == 2
    assert seen["sent"] == "none"


def test_a_rejected_enter_still_restricts_the_click_after_a_reload(drive) -> None:
    def moves(page: BrowserSurface) -> list[Move]:
        return [
            look,
            do(TYPE_AMOUNT),
            do(enter(AMOUNT, "submit_payment", flag=True, payee="10001")),
            reload(page),
            do(press(SEND_BY_NAME, "confirm", payee="10001")),
            do(press(SEND_BY_TYPE, "confirm", payee="10001")),
        ]

    with drive(moves, path="/payments", escalator=rejecting, actions=RELOADING) as ran:
        seen = counters(ran.page)

    assert len(risky_requests(ran)) == 3
    assert seen["sent"] == "none"


def test_a_bound_denial_survives_a_reload_and_the_other_submission_path(
    drive,
) -> None:
    actions = grants({**PAYING, "navigate": "safe"})
    actions["click"] = {"any": "safe", "effects": {"submit_payment": "deny"}}

    def moves(page: BrowserSurface) -> list[Move]:
        return [
            look,
            do(TYPE_AMOUNT),
            do(press(SEND_BY_NAME, "submit_payment", payee="10001")),
            reload(page),
            do(enter(AMOUNT, "confirm", payee="10001")),
            do(press(SEND_BY_TYPE, "confirm", flag=True, payee="10001")),
        ]

    with drive(moves, path="/payments", escalator=approving, actions=actions) as ran:
        seen = counters(ran.page)

    assert refusals(ran) == [Denial.EFFECT_DENIED] * 3
    assert risky_requests(ran) == []
    assert seen["sent"] == "none"


def test_a_finding_survives_a_reload_and_a_rebuilt_form(drive) -> None:
    """The click posted once. Neither a reload nor a rebuilt form lets it again."""
    before_reload: list[str | None] = []

    def the_click(seen: Transcript) -> int:
        return next(
            turn.step
            for turn in seen.history
            if turn.action is not None and turn.action.kind is ActionKind.CLICK
        )

    def moves(page: BrowserSurface) -> list[Move]:
        return [
            look,
            do(TYPE_AMOUNT),
            do(press(SEND_BY_NAME, "save_form", payee="10001")),
            lambda seen: FlagRisk(
                RiskFinding(the_click(seen), "submit_payment", "it posted")
            ),
            lambda seen: (
                before_reload.append(read(page, PAYMENTS_SENT)) or reload(page)(seen)
            ),
            do(enter(AMOUNT, "confirm", payee="10001")),
            then(page, REBUILD_FORM, move=look),
            do(press(SEND_BY_TYPE, "confirm", payee="10001")),
            do(enter(MEMO, "add_memo", payee="10001")),
        ]

    with drive(moves, path="/payments", escalator=rejecting, actions=RELOADING) as ran:
        seen = counters(ran.page)

    assert before_reload == ["10001"]
    assert len(risky_requests(ran)) == 3
    assert [r.source for r in ran.result.restrictions] == [Source.FINDING]
    # The reload reset the page's counter. Nothing was sent after it.
    assert seen["sent"] == "none"


@pytest.mark.parametrize(
    ("rule", "operator"),
    [("risky", rejecting), ("deny", approving)],
    ids=["risky-goes-to-a-person", "deny-stays-denied"],
)
def test_a_form_that_cannot_be_reidentified_stops_without_a_payment(
    drive, rule, operator
) -> None:
    """Renamed form, relabelled button: nothing observed says it is the same one."""
    actions = grants(PAYING)
    actions["click"] = {"any": "safe", "effects": {"submit_payment": rule}}

    def moves(page: BrowserSurface) -> list[Move]:
        return [
            look,
            do(TYPE_AMOUNT),
            do(press(SEND_BY_NAME, "submit_payment", payee="10001")),
            then(page, REBUILD_RENAMED, move=look),
            do(enter(AMOUNT, "confirm", payee="10001")),
            do(press(AxLocator("button", "Pay"), "confirm", payee="10001")),
        ]

    with drive(moves, path="/payments", escalator=operator, actions=actions) as ran:
        seen = counters(ran.page)

    if rule == "deny":
        assert refusals(ran) == [Denial.EFFECT_DENIED] * 3
        assert risky_requests(ran) == []
    else:
        assert len(risky_requests(ran)) == 3
    assert seen["sent"] == "none"


def test_other_submissions_stay_usable_after_a_reload(drive) -> None:
    """Save draft and the search form are told apart from the payment by what shows."""

    def moves(page: BrowserSurface) -> list[Move]:
        return [
            look,
            do(TYPE_AMOUNT),
            do(press(SEND_BY_NAME, "submit_payment", flag=True)),
            reload(page),
            do(press(SAVE_DRAFT, "save_draft")),
            do(Action(ActionKind.TYPE, LOOKUP, "Riverside", effect="enter_search")),
            do(enter(LOOKUP, "search")),
            done,
        ]

    with drive(
        moves,
        path="/payments",
        escalator=rejecting,
        actions=RELOADING,
        records=NO_RECORDS,
    ) as ran:
        seen = counters(ran.page)

    assert len(risky_requests(ran)) == 1
    assert seen == {"sent": "none", "drafts": "1", "searches": "1", "transfers": "0"}


# 18. A control an observation reports can be targeted by what it reported.

IDENTITY_PAGE = """<!doctype html><html><body>
<h1>Identity</h1>
<button id="save"><span aria-hidden="true">icon-star</span> Save</button>
<span id="verb">Approve</span><span id="who">member 10001</span>
<button id="labelled" aria-labelledby="verb who">x</button>
<section><h2>Savings</h2><button id="view-savings">View</button></section>
<section><h2>Checking</h2><button id="view-checking">View</button></section>
<form aria-label="Transfer"><button type="button" id="send-a">Send</button></form>
<form aria-label="Payment"><button type="button" id="send-b">Send</button></form>
<table><caption>Holds</caption>
  <tr><td></td><td><button id="release">Release</button></td></tr>
</table>
<table>
  <tr><th scope="row">1</th><td><button id="open-1">Open</button></td></tr>
  <tr><th scope="row">10001</th><td><button id="open-10001">Open</button></td></tr>
</table>
<button id="quoted">Say "hello" \\ back</button>
<button id="long">LONG</button>
<span id="lock" class="lock posting-lock icon" title="Lock" onclick="1"
      style="display:inline-block;width:20px;height:20px"></span>
<a id="bare">Not a link</a>
<div id="primary" role="button primary">Primary</div>
<output id="clicked">none</output>
<script>
  document.getElementById('long').textContent =
    'Confirm the transfer of funds ' + 'between the two accounts '.repeat(20);
  for (const button of document.querySelectorAll('button, [role], span[onclick]')) {
    button.addEventListener('click', () => {
      document.getElementById('clicked').textContent = button.id;
    });
  }
</script>
</body></html>"""

CLICKED = DomLocator("output", DomAttribute.ID, "clicked")


def clicked_by(page: BrowserSurface, target: AxLocator | DomLocator) -> str | None:
    """Click ``target`` and report which element's own handler saw the click."""
    assert page.act(Action(ActionKind.CLICK, target)).outcome is Outcome.OK
    return read(page, CLICKED)


def by_id(observation: Observation, element_id: str) -> AxNode:
    return node_for(observation, element_id)


def test_hidden_decoration_is_left_out_of_the_name_and_the_target(pages) -> None:
    with pages({"/": IDENTITY_PAGE}) as (page, _):
        seen = page.observe(STRUCTURED)
        save = by_id(seen, "save")
        assert save.name == "Save"
        assert clicked_by(page, AxLocator("button", "Save")) == "save"


def test_every_aria_labelledby_reference_names_the_control(pages) -> None:
    with pages({"/": IDENTITY_PAGE}) as (page, _):
        seen = page.observe(STRUCTURED)
        assert by_id(seen, "labelled").name == "Approve member 10001"
        target = AxLocator("button", "Approve member 10001")
        assert clicked_by(page, target) == "labelled"


@pytest.mark.parametrize(
    ("element_id", "scope"),
    [
        ("view-savings", Scope(ScopeKind.REGION, "Savings")),
        ("view-checking", Scope(ScopeKind.REGION, "Checking")),
        ("send-b", Scope(ScopeKind.FORM, "Payment")),
        ("release", Scope(ScopeKind.TABLE, "Holds")),
        ("open-1", Scope(ScopeKind.ROW, "1")),
        ("open-10001", Scope(ScopeKind.ROW, "10001")),
    ],
)
def test_every_reported_scope_resolves_to_its_own_control(
    pages, element_id: str, scope: Scope
) -> None:
    """A region named by its heading, a form by its label, a table by its caption.

    Row "1" is also a substring of row "10001", and it still names one row.
    """
    with pages({"/": IDENTITY_PAGE}) as (page, _):
        seen = page.observe(STRUCTURED)
        node = by_id(seen, element_id)
        assert node.scope == scope
        target = AxLocator(node.role, node.name, scope=scope)
        assert clicked_by(page, target) == element_id


def test_a_name_with_quotes_and_a_backslash_is_targeted_safely(pages) -> None:
    with pages({"/": IDENTITY_PAGE}) as (page, _):
        seen = page.observe(STRUCTURED)
        name = by_id(seen, "quoted").name
        assert name == 'Say "hello" \\ back'
        assert clicked_by(page, AxLocator("button", name)) == "quoted"


def test_a_long_name_is_shortened_for_display_and_targeted_whole(pages) -> None:
    from computeruse.model import _control, _state

    with pages({"/": IDENTITY_PAGE}) as (page, profile):
        seen = page.observe(STRUCTURED)
        node = by_id(seen, "long")
        assert len(node.name) > 400
        transcript = Transcript(
            goal="confirm",
            location=seen.location,
            allowed=profile.actions,
            permitted_modes=(ObservationMode.STRUCTURED,),
            observations=(seen,),
        )
        shown = json.loads(_state(transcript))["observations"][0]["controls"]
        entry = next(item for item in shown if item["ref"] == node.control)
        assert entry["name"].endswith("...")
        assert len(entry["name"]) < len(node.name)
        target = _control(node.control, transcript)
        assert isinstance(target, AxLocator)
        assert target.name == node.name
        assert clicked_by(page, target) == "long"


def test_one_class_of_several_names_the_control(pages) -> None:
    with pages({"/": IDENTITY_PAGE}) as (page, _):
        seen = page.observe(STRUCTURED)
        lock = by_id(seen, "lock")
        assert lock.classes == ("lock", "posting-lock", "icon")
        target = DomLocator("span", DomAttribute.CSS_CLASS, "posting-lock")
        assert names_node(target, lock)
        assert clicked_by(page, target) == "lock"


def test_roles_are_reported_as_a_target_can_use_them(pages) -> None:
    with pages({"/": IDENTITY_PAGE}) as (page, _):
        seen = page.observe(STRUCTURED)
        assert by_id(seen, "bare").role == "generic"
        assert by_id(seen, "primary").role == "button"
        assert clicked_by(page, AxLocator("button", "Primary")) == "primary"


def test_every_reported_control_can_be_named_back_to_itself(pages, surface) -> None:
    """Each control a ref names resolves to the element the observation saw."""
    from computeruse.model import _control

    def check(page: BrowserSurface, profile) -> int:
        seen = page.observe(STRUCTURED)
        transcript = Transcript(
            goal="check",
            location=seen.location,
            allowed=profile.actions,
            permitted_modes=(ObservationMode.STRUCTURED,),
            observations=(seen,),
        )
        named = 0
        for node in seen.nodes:
            try:
                target = _control(node.control, transcript)
            except ModelError:
                continue
            element = page._locate(target)
            held = page._controls[node.control].handle
            assert element.evaluate("(el, held) => el === held", held), node
            named += 1
        return named

    with pages({"/": IDENTITY_PAGE}) as (page, profile):
        assert check(page, profile) >= 20
    with surface("/members/12345") as page:
        assert check(page, page._profile) >= 40


# 19. Frames that share a name, or have none, are told apart.

PANE = """<!doctype html><html><body><div><h1>{label}</h1>
<button onclick="document.querySelector('output').textContent='{label}'">Go</button>
<output>idle</output></div></body></html>"""

FRAMES_PAGE = """<!doctype html><html><body><h1>Frames</h1>
<iframe name="pane" src="/left"></iframe>
<iframe name="pane" src="/right"></iframe>
<iframe src="/first"></iframe>
<iframe src="/second"></iframe>
<iframe name="solo" src="/solo"></iframe>
</body></html>"""

FRAME_PAGES = {
    "/": FRAMES_PAGE,
    **{
        f"/{label}": PANE.format(label=label)
        for label in ("left", "right", "first", "second", "solo", "moved")
    },
}


def loaded(page: BrowserSurface) -> BrowserSurface:
    """Wait for every frame to load, which the session's own entry does not."""
    page._page.wait_for_load_state("load")
    return page


def frame_of(observation: Observation, heading: str) -> tuple[str, ...]:
    found = [
        node.frame
        for node in observation.nodes
        if node.tag == "h1" and node.name == heading
    ]
    assert len(found) == 1, (heading, found)
    return found[0]


def pressed_label(page: BrowserSurface, frame: tuple[str, ...]) -> str:
    seen = page.observe(STRUCTURED)
    return next(
        node.name for node in seen.nodes if node.frame == frame and node.tag == "output"
    )


def test_repeated_and_unnamed_frames_get_distinct_ids(pages) -> None:
    with pages(FRAME_PAGES, "/") as (page, _):
        seen = loaded(page).observe(STRUCTURED)
        frames = {
            heading: frame_of(seen, heading)
            for heading in ("left", "right", "first", "second", "solo")
        }
        assert len(set(frames.values())) == 5
        assert frames["solo"] == ("solo",)
        assert frames["left"][0].startswith("pane#")
        assert frames["first"][0].startswith("#")
        for heading, frame in frames.items():
            assert pressed_label(page, frame) == "idle"
            go = AxLocator("button", "Go", frame=frame)
            assert page.act(Action(ActionKind.CLICK, go)).outcome is Outcome.OK
            assert pressed_label(page, frame) == heading


def test_a_shared_frame_name_is_refused_rather_than_resolved_first(pages) -> None:
    with pages(FRAME_PAGES, "/") as (page, _):
        loaded(page).observe(STRUCTURED)
        go = AxLocator("button", "Go", frame=("pane",))
        result = page.act(Action(ActionKind.CLICK, go))
        assert result.outcome is Outcome.AMBIGUOUS
        assert "share that name" in (result.detail or "")


def test_a_frame_id_goes_stale_when_its_frame_navigates(pages) -> None:
    with pages(FRAME_PAGES, "/") as (page, _):
        seen = loaded(page).observe(STRUCTURED)
        first = frame_of(seen, "first")
        page._page.evaluate("document.querySelectorAll('iframe')[2].src = '/moved'")
        page._page.wait_for_timeout(500)
        go = AxLocator("button", "Go", frame=first)
        result = page.act(Action(ActionKind.CLICK, go))
        assert result.outcome is Outcome.STALE
        again = page.observe(STRUCTURED)
        assert frame_of(again, "moved") != first


def test_record_evidence_uses_the_same_frame_identity(pages) -> None:
    """Evidence in the second pane is read from that pane, not the first."""
    with pages(FRAME_PAGES, "/") as (page, _):
        seen = loaded(page).observe(STRUCTURED)
        right = frame_of(seen, "right")
        go = AxLocator("button", "Go", frame=right)
        proof = RecordEvidence(
            AxLocator("heading", "right", frame=right), "right", Relation.CONTAINER
        )
        result = page.act(Action(ActionKind.CLICK, go, evidence=proof))
        assert result.outcome is Outcome.OK
        assert pressed_label(page, right) == "right"


# 20. A wait waits for its target, and reading a field reads its value.

FIELDS_PAGE = """<!doctype html><html><body><h1>Fields</h1>
<label>Amount <input id="amount" value="125.00"></label>
<label>Memo <textarea id="memo">first line
second line</textarea></label>
<label>Branch <select id="branch"><option>Riverside</option>
  <option selected>Eastgate</option></select></label>
<label>Notify <input id="notify" type="checkbox" checked></label>
<label>Paper <input id="paper" type="radio" name="kind"></label>
<label>Passcode <input id="passcode" type="password" value="synthetic-pass"></label>
<table><tr><td id="cell">$4,212.55</td></tr></table>
<button id="later" onclick="setTimeout(() => {
  const b = document.createElement('button');
  b.textContent = 'Arrived';
  document.body.appendChild(b);
}, 800)">Later</button>
</body></html>"""


@pytest.mark.parametrize(
    ("element_id", "tag", "shown"),
    [
        ("amount", "input", "125.00"),
        ("memo", "textarea", "first line second line"),
        ("branch", "select", "Eastgate"),
        ("notify", "input", "checked"),
        ("paper", "input", "not checked"),
        ("cell", "td", "$4,212.55"),
    ],
)
def test_reading_a_field_returns_its_value_or_state(
    pages, element_id: str, tag: str, shown: str
) -> None:
    with pages({"/": FIELDS_PAGE}) as (page, _):
        target = DomLocator(tag, DomAttribute.ID, element_id)
        result = page.act(Action(ActionKind.READ, target))
        assert result.outcome is Outcome.OK
        assert result.extracted == shown


def test_a_typed_value_is_read_back_from_the_field(pages) -> None:
    with pages({"/": FIELDS_PAGE}) as (page, _):
        amount = DomLocator("input", DomAttribute.ID, "amount")
        page.act(Action(ActionKind.TYPE, amount, "300.10"))
        assert read(page, amount) == "300.10"
        assert page.act(Action(ActionKind.ASSERT, amount, "300.10")).outcome is (
            Outcome.OK
        )


@pytest.mark.rule(18)
def test_a_secret_field_is_never_read_by_read_or_assert(pages, monkeypatch) -> None:
    monkeypatch.setenv("APPROVER_PASSCODE", BRANCH_SECRET)
    with pages({"/": FIELDS_PAGE}) as (page, _):
        passcode = DomLocator("input", DomAttribute.ID, "passcode")
        for action in (
            Action(ActionKind.READ, passcode),
            Action(ActionKind.ASSERT, passcode, "synthetic-pass"),
        ):
            result = page.act(action)
            assert result.outcome is Outcome.NOT_ACTIONABLE
            assert result.extracted is None
        memo = DomLocator("textarea", DomAttribute.ID, "memo")
        page.act(Action(ActionKind.TYPE, memo, TYPE_THE_SECRET))
        result = page.act(Action(ActionKind.READ, memo))
        assert result.outcome is Outcome.NOT_ACTIONABLE
        assert BRANCH_SECRET not in (result.extracted or "")
        # A value locator cannot be used to test what the secret field holds.
        guess = DomLocator("input", DomAttribute.VALUE, "synthetic-pass")
        assert page.act(Action(ActionKind.READ, guess)).outcome is Outcome.NOT_FOUND


def test_wait_for_waits_for_a_control_that_is_not_there_yet(pages) -> None:
    with pages({"/": FIELDS_PAGE}) as (page, _):
        later = AxLocator("button", "Later")
        page.act(Action(ActionKind.CLICK, later))
        arrived = AxLocator("button", "Arrived")
        waited = page.act(Action(ActionKind.WAIT_FOR, arrived, "3000"))
        assert waited.outcome is Outcome.OK
        missing = AxLocator("button", "Never")
        gone = page.act(Action(ActionKind.WAIT_FOR, missing, "300"))
        assert gone.outcome is Outcome.NOT_FOUND
        assert "300 ms" in (gone.detail or "")


# 21. Observation reaches past its node limit without scrolling.

LONG_PAGE = (
    "<!doctype html><html><body><style>button {display: block}</style><h1>Long</h1>"
    + "".join(f"<button>Item {n}</button>" for n in range(260))
    + '<div id="note">Balance on file: $77.10</div>'
    + '<div id="editor" contenteditable="true"></div>'
    + '<label for="late">Late field</label><input id="late">'
    + "</body></html>"
)


def test_a_window_reaches_controls_past_the_node_limit(pages) -> None:
    with pages({"/": LONG_PAGE}) as (page, _):
        before = page._page.evaluate("window.scrollY")
        first = page.observe(STRUCTURED)
        assert first.status is ObservationStatus.PARTIAL
        assert first.window is not None
        assert first.window.shown == 200
        assert first.window.total > 260
        assert not any(node.name == "Late field" for node in first.nodes)
        rest = page.observe(
            ObservationRequest(ObservationMode.STRUCTURED, start=first.window.shown)
        )
        late = next(n for n in rest.nodes if ("id", "late") in n.attributes)
        assert late.name == "Late field"
        assert late.tag == "input"
        assert ActionKind.TYPE in late.interactions
        assert not late.in_view
        note = next(n for n in rest.nodes if ("id", "note") in n.attributes)
        assert note.name == "Balance on file: $77.10"
        editor = next(n for n in rest.nodes if ("id", "editor") in n.attributes)
        assert editor.role == "textbox"
        assert ActionKind.TYPE in editor.interactions
        assert page._page.evaluate("window.scrollY") == before
        field = AxLocator("textbox", "Late field")
        assert page.act(Action(ActionKind.TYPE, field, "found")).outcome is Outcome.OK
        assert read(page, DomLocator("input", DomAttribute.ID, "late")) == "found"


def test_a_scope_or_a_frame_narrows_what_is_read(pages) -> None:
    with pages(FRAME_PAGES, "/") as (page, _):
        seen = loaded(page).observe(STRUCTURED)
        solo = frame_of(seen, "solo")
        narrow = page.observe(
            ObservationRequest(ObservationMode.STRUCTURED, frame=solo)
        )
        assert narrow.nodes
        assert {node.frame for node in narrow.nodes} == {solo}
        assert narrow.page_state.signature == ()
    with pages({"/": IDENTITY_PAGE}) as (page, _):
        savings = Scope(ScopeKind.REGION, "Savings")
        narrow = page.observe(
            ObservationRequest(ObservationMode.STRUCTURED, scope=savings)
        )
        assert {node.scope for node in narrow.nodes} == {savings}
        assert any(node.name == "View" for node in narrow.nodes)


def test_a_scroll_with_a_target_brings_that_control_into_view(pages) -> None:
    with pages({"/": LONG_PAGE}) as (page, _):
        field = AxLocator("textbox", "Late field")
        result = page.act(Action(ActionKind.SCROLL, field, "into_view"))
        assert result.outcome is Outcome.OK
        rest = page.observe(ObservationRequest(ObservationMode.STRUCTURED, start=200))
        late = next(n for n in rest.nodes if ("id", "late") in n.attributes)
        assert late.in_view


# 22. A permitted popup goes to a person, and the run resumes on its own page.

POPUP_PAGES = {
    "/opener": """<!doctype html><html><body><h1>Opener</h1>
<button onclick="window.open('/popup')">Open statements</button>
<button onclick="window.open('/forbidden/x')">Open elsewhere</button>
<button onclick="window.open('/alerting')">Open alerting</button>
<button onclick="document.querySelector('output').textContent = 'posted'"
  >Post entry</button>
<output id="posted">idle</output>
</body></html>""",
    "/popup": """<!doctype html><html><body><h1>Statements</h1>
<button onclick="document.querySelector('output').textContent =
  confirm('Mark as reviewed?') ? 'reviewed' : 'declined'">Confirm statement</button>
<button onclick="window.close()">Close window</button>
<label>Passcode <input id="pin" type="password" value="synthetic-pass"></label>
<a href="/file" download="statement.txt">Download statement</a>
<output id="status">open</output>
</body></html>""",
    "/archive": """<!doctype html><html><body><h1>Archive</h1></body></html>""",
    "/alerting": """<!doctype html><html><body><h1>Alerting</h1>
<script>setTimeout(() => alert('Session note'), 50)</script></body></html>""",
    "/file": "statement",
}

OPEN_STATEMENTS = AxLocator("button", "Open statements")
OPEN_ALERTING = AxLocator("button", "Open alerting")
POST = AxLocator("button", "Post entry")
STATUS = DomLocator("output", DomAttribute.ID, "status")
POSTED = DomLocator("output", DomAttribute.ID, "posted")


def popup(page: BrowserSurface, page_id: str = "page-2") -> None:
    """Wait until the session manages the popup, and until it has loaded."""
    for _ in range(40):
        if page_id in page._pages:
            page._pages[page_id].wait_for_load_state("load")
            return
        page._page.wait_for_timeout(50)
    raise AssertionError(f"{page_id} never opened")


def dialog_in(page: BrowserSurface, page_id: str = "page-2") -> None:
    """Wait until a dialog is held for ``page_id``."""
    for _ in range(40):
        if page_id in page._pending:
            return
        page._page.wait_for_timeout(50)
    raise AssertionError(f"no dialog opened in {page_id}")


def test_a_popup_is_listed_and_never_made_active(pages) -> None:
    with pages(POPUP_PAGES, "/opener", allow_new_windows=True) as (page, _):
        opened = page.act(Action(ActionKind.CLICK, OPEN_STATEMENTS))
        popup(page)
        effects = [*opened.side_effects, *page._result(Outcome.OK).side_effects]
        assert any("page-2" in effect and "person" in effect for effect in effects)
        assert [(item.page_id, item.active) for item in page.pages()] == [
            ("page-1", True),
            ("page-2", False),
        ]
        assert page.location().endswith("/opener")
        assert ActionKind.CLICK in page.capabilities()
        assert "switch_context" not in {kind.value for kind in page.capabilities()}
        # An action decided while only page-1 was known is refused before
        # any input reaches the page.
        known = Expectation(
            PageState(page.location(), page="page-1"), windows=("page-1",)
        )
        refused = page.act(Action(ActionKind.CLICK, POST), expect=known)
        assert refused.outcome is Outcome.STALE
        assert "page-2" in (refused.detail or "")
        assert read(page, POSTED) == "idle"
        both = dataclasses.replace(known, windows=("page-1", "page-2"))
        assert (
            page.act(Action(ActionKind.CLICK, POST), expect=both).outcome is Outcome.OK
        )
        assert read(page, POSTED) == "posted"


def test_a_popup_dialog_is_listed_with_its_page(pages) -> None:
    with pages(POPUP_PAGES, "/opener", allow_new_windows=True) as (page, _):
        page.act(Action(ActionKind.CLICK, OPEN_ALERTING))
        dialog_in(page)
        listed = {item.page_id: item for item in page.pages()}
        assert listed["page-2"].dialog.startswith("dialog-")
        assert listed["page-2"].route == "/**"
        home = page.observe(STRUCTURED)
        assert home.dialog is None
        assert home.status is ObservationStatus.UNAVAILABLE
        assert "a person must answer it" in home.notes[0]
        assert [item.page_id for item in home.pages] == ["page-1", "page-2"]
        blocked = page.act(Action(ActionKind.CLICK, OPEN_STATEMENTS))
        assert blocked.outcome is Outcome.NOT_ACTIONABLE
        page._pages["page-2"].close()
        page._page.wait_for_timeout(100)
        assert [item.page_id for item in page.pages()] == ["page-1"]
        assert page.observe(STRUCTURED).usable


def test_a_popup_that_closes_hands_the_session_back(pages) -> None:
    with pages(POPUP_PAGES, "/opener", allow_new_windows=True) as (page, _):
        page.act(Action(ActionKind.CLICK, OPEN_STATEMENTS))
        popup(page)
        # A person's own tooling may bring the popup forward. No decision can.
        page._activate("page-2")
        closed = page.act(Action(ActionKind.CLICK, AxLocator("button", "Close window")))
        page._page.wait_for_timeout(200)
        effects = [*closed.side_effects, *page._result(Outcome.OK).side_effects]
        assert any("page-1 is active again" in effect for effect in effects)
        assert page.location().endswith("/opener")


def test_a_forbidden_popup_stays_closed(pages) -> None:
    with pages(POPUP_PAGES, "/opener") as (page, _):
        opened = page.act(Action(ActionKind.CLICK, OPEN_STATEMENTS))
        page._page.wait_for_timeout(300)
        effects = [*opened.side_effects, *page._result(Outcome.OK).side_effects]
        assert "a new window was closed" in effects
        assert [item.page_id for item in page.pages()] == ["page-1"]


def test_a_popup_outside_the_routes_is_listed_without_its_location(pages) -> None:
    with pages(POPUP_PAGES, "/opener", allow_new_windows=True) as (page, _):
        page.act(Action(ActionKind.CLICK, AxLocator("button", "Open elsewhere")))
        page._page.wait_for_timeout(300)
        listed = page.pages()
        assert listed[1].route == ""
        assert listed[1].location == ""
        assert page.location().endswith("/opener")


def test_a_popup_is_masked_and_refuses_downloads(pages) -> None:
    with pages(POPUP_PAGES, "/opener", allow_new_windows=True) as (page, _):
        page.act(Action(ActionKind.CLICK, OPEN_STATEMENTS))
        popup(page)
        page._activate("page-2")
        picture = page.observe(VISUAL)
        assert picture.image is not None
        assert "credential fields" in picture.visual.masked
        pin = page.observe(STRUCTURED)
        assert node_for(pin, "pin").value is None
        link = AxLocator("link", "Download statement")
        downloaded = page.act(Action(ActionKind.CLICK, link))
        page._page.wait_for_timeout(300)
        effects = [*downloaded.side_effects, *page._result(Outcome.OK).side_effects]
        assert "a download was cancelled" in effects


@dataclasses.dataclass
class Person:
    """An operator who does something in the live session, then answers."""

    page: BrowserSurface
    acts: list[Callable[[BrowserSurface, InterventionRequest], Handoff]]
    requests: list[InterventionRequest] = dataclasses.field(default_factory=list)
    seen: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def request(self, intervention: InterventionRequest) -> Handoff:
        self.requests.append(intervention)
        self.seen.append(
            {
                "active": self.page._active,
                "status": self.page._pages["page-2"].locator("#status").inner_text()
                if "page-2" in self.page._pages
                and self.page._pages["page-2"].url.endswith("/popup")
                else None,
            }
        )
        if not self.acts:
            return Handoff(HandoffOutcome.TIMED_OUT)
        return self.acts.pop(0)(self.page, intervention)


def closes(page_id: str) -> Callable[[BrowserSurface, InterventionRequest], Handoff]:
    def _close(page: BrowserSurface, _: InterventionRequest) -> Handoff:
        page._pages[page_id].close()
        page._page.wait_for_timeout(100)
        return Handoff(HandoffOutcome.RESUMED, "I closed it")

    return _close


def navigates(page_id: str, path: str):
    def _go(page: BrowserSurface, _: InterventionRequest) -> Handoff:
        target = page._pages[page_id]
        target.goto(target.url.rsplit("/", 1)[0] + path)
        return Handoff(HandoffOutcome.RESUMED, "I left it open")

    return _go


def approves(_: BrowserSurface, __: InterventionRequest) -> Handoff:
    return Handoff(HandoffOutcome.APPROVED)


def popup_run(pages, moves: list[Move], acts, **edits):
    edits = {"allow_new_windows": True, "actions": grants(DIALOGS), **edits}
    with pages(POPUP_PAGES, "/opener", **edits) as (page, profile):
        person = Person(page, list(acts))
        decider = Moves(moves)
        journal = MemoryJournal()
        result = discover(
            "post the entry",
            profile,
            surface=page,
            decider=decider,
            escalator=person,
            journal=journal,
            clock=FakeClock(),
        )
        after = {
            "posted": read(page, POSTED),
            "pages": [item.page_id for item in page.pages()],
            "active": page._active,
        }
    return result, person, decider, journal, after


def test_a_permitted_popup_goes_to_a_person_before_more_input(pages) -> None:
    """The click opened it. Nothing else ran until the person finished."""
    moves: list[Move] = [
        look,
        do(Action(ActionKind.CLICK, OPEN_STATEMENTS)),
        look,
        done,
    ]
    result, person, decider, journal, after = popup_run(
        pages, moves, [closes("page-2")]
    )
    assert [request.trigger for request in person.requests] == [Trigger.NEW_WINDOW]
    request = person.requests[0]
    assert [(item.page_id, item.active) for item in request.pages] == [
        ("page-1", True),
        ("page-2", False),
    ]
    # The popup received no input and was never made active.
    assert person.seen == [{"active": "page-1", "status": "open"}]
    assert result.ending is Ending.COMPLETED
    assert after["pages"] == ["page-1"]
    notices = [notice for seen in decider.seen for notice in seen.notices]
    assert any("was performed" in notice for notice in notices)
    escalated = [event for event in journal.events if isinstance(event, Escalated)]
    assert [event.trigger for event in escalated] == [Trigger.NEW_WINDOW]
    assert "/popup" not in json.dumps([as_record(e) for e in journal.events])


def test_a_popup_with_a_dialog_already_open_goes_to_a_person(pages) -> None:
    """Found in review: the popup's dialog left the loop with nothing usable."""
    moves: list[Move] = [look, do(Action(ActionKind.CLICK, OPEN_ALERTING)), done]

    def waits_then_closes(page: BrowserSurface, request: InterventionRequest):
        dialog_in(page)
        return closes("page-2")(page, request)

    result, person, _, _, after = popup_run(pages, moves, [waits_then_closes])
    assert [request.trigger for request in person.requests] == [Trigger.NEW_WINDOW]
    assert result.ending is Ending.COMPLETED
    assert after["pages"] == ["page-1"]


def test_a_popup_the_person_leaves_open_is_not_handed_over_again(pages) -> None:
    moves: list[Move] = [
        look,
        do(Action(ActionKind.CLICK, OPEN_STATEMENTS)),
        look,
        do(Action(ActionKind.CLICK, POST)),
        look,
        done,
    ]
    result, person, _, _, after = popup_run(
        pages, moves, [navigates("page-2", "/archive")]
    )
    assert [request.trigger for request in person.requests] == [Trigger.NEW_WINDOW]
    assert result.ending is Ending.COMPLETED
    assert after == {
        "posted": "posted",
        "pages": ["page-1", "page-2"],
        "active": "page-1",
    }


def test_a_newly_opened_popup_starts_a_new_intervention(pages) -> None:
    moves: list[Move] = [
        look,
        do(Action(ActionKind.CLICK, OPEN_STATEMENTS)),
        look,
        do(Action(ActionKind.CLICK, OPEN_STATEMENTS)),
        look,
        done,
    ]
    result, person, _, _, _ = popup_run(
        pages, moves, [navigates("page-2", "/archive"), closes("page-3")]
    )
    assert [request.trigger for request in person.requests] == [
        Trigger.NEW_WINDOW,
        Trigger.NEW_WINDOW,
    ]
    assert [item.page_id for item in person.requests[1].pages] == [
        "page-1",
        "page-2",
        "page-3",
    ]
    assert result.ending is Ending.COMPLETED


def test_returning_control_is_not_approval_for_a_risky_action(pages) -> None:
    moves: list[Move] = [
        look,
        do(Action(ActionKind.CLICK, OPEN_STATEMENTS)),
        look,
        do(Action(ActionKind.CLICK, POST, flag_risky=True, effect="post_entry")),
    ]
    result, person, _, _, after = popup_run(pages, moves, [closes("page-2")])
    assert [request.trigger for request in person.requests] == [
        Trigger.NEW_WINDOW,
        Trigger.RISKY_ACTION,
    ]
    assert after["posted"] == "idle"
    assert result.ending is Ending.HANDED_OFF


def test_a_forbidden_popup_is_blocked_and_the_run_goes_on(pages) -> None:
    moves: list[Move] = [
        look,
        do(Action(ActionKind.CLICK, OPEN_STATEMENTS)),
        look,
        done,
    ]
    result, person, decider, _, after = popup_run(
        pages, moves, [], allow_new_windows=False
    )
    assert person.requests == []
    assert result.ending is Ending.COMPLETED
    assert after["pages"] == ["page-1"]
    effects = [
        effect
        for seen in decider.seen
        for turn in seen.history
        for effect in turn.side_effects
    ]
    assert "a new window was closed" in effects


def test_a_popup_left_with_its_dialog_waiting_ends_the_run(pages) -> None:
    """A dialog that stops every page, handed over once, is not handed over forever."""
    moves: list[Move] = [look, do(Action(ActionKind.CLICK, OPEN_ALERTING)), look]

    def waits(page: BrowserSurface, _: InterventionRequest) -> Handoff:
        dialog_in(page)
        return Handoff(HandoffOutcome.RESUMED)

    result, person, _, _, _ = popup_run(pages, moves, [waits, waits])
    assert [request.trigger for request in person.requests] == [Trigger.NEW_WINDOW]
    assert result.ending is Ending.HANDED_OFF
    assert "still stops this session" in result.detail


# 23. Only what the surface can perform is offered, and upload and delete are gone.


def test_upload_and_delete_are_rejected_by_name(edited_profile) -> None:
    from computeruse.profile import ProfileError

    for name in ("upload", "delete"):
        with pytest.raises(ProfileError) as refused:
            edited_profile(actions=grants({"observe": "safe", name: "risky"}))
        assert f"action {name} is not supported" in str(refused.value)
    with pytest.raises(ProfileError) as refused:
        edited_profile(records={"actions": ["delete"], "routes": []})
    assert "delete, which no adapter supports" in str(refused.value)


def test_the_browser_offers_only_combinations_it_performs() -> None:
    from computeruse.actions import TargetForm
    from computeruse.browser import SUPPORTED
    from computeruse.model import tools
    from computeruse.profile import ActionKind as Kind

    allowed = dict.fromkeys(Kind, Risk.SAFE)
    transcript = Transcript(
        goal="g",
        location="https://host.test/",
        allowed=allowed,
        permitted_modes=(ObservationMode.STRUCTURED, ObservationMode.VISUAL),
        observations=(
            Observation(
                "obs-1",
                ObservationMode.VISUAL,
                ObservationStatus.COMPLETE,
                PageState("https://host.test/"),
                image=b"png",
                visual=VisualMeta(10, 10, 0, 0),
            ),
        ),
        capabilities=SUPPORTED,
    )
    offered = {tool["name"]: tool for tool in tools(transcript)}
    act = set(offered["act"]["parameters"]["properties"]["action"]["enum"])
    screen = {
        entry["properties"]["kind"]["enum"][0]
        for entry in offered["computer"]["parameters"]["properties"]["command"]["anyOf"]
    }
    for kind, forms in SUPPORTED.items():
        assert (kind.value in act) == bool(forms - {TargetForm.SCREEN}), kind
        assert (kind.value in screen) == (TargetForm.SCREEN in forms), kind
    assert "select" not in screen
    assert "double_click" not in act
    assert "observe" not in act


def test_a_combination_the_surface_cannot_perform_is_refused_before_the_gate(
    profile,
) -> None:
    from computeruse.actions import TargetForm

    surface = ScriptedSurface(
        [Screen("https://sandbox.example.test/members/12345")],
        supports={ActionKind.CLICK: frozenset({TargetForm.ACCESSIBILITY})},
    )
    decider = ScriptedDecider(
        [
            Observe(STRUCTURED),
            Propose(Action(ActionKind.CLICK, VisualAnchor("anchor-1"))),
        ]
    )
    discover(
        "goal",
        profile,
        surface=surface,
        decider=decider,
        escalator=ScriptedEscalatorLike(),
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert surface.acted == []
    assert any(
        "does not perform click with a visual target" in notice
        for seen in decider.seen
        for notice in seen.notices
    )


# 24. A finish claim completes a run only when the page shows it.

LEDGER_FRAME = ("ledger",)
BALANCE_GOAL = "Report the balance of savings account S-1001 for member 12345."


def balance_claim(
    value: str, account: str = "S-1001", *, cell: str | None = None, member: bool = True
):
    """Claim ``value``, checked on the ledger cell showing ``cell``.

    The cell is tied to ``account`` by the row it sits in, and the member is
    checked by the page heading.
    """
    shown = DomLocator("td", DomAttribute.TEXT, cell or value, frame=LEDGER_FRAME)
    header = AxLocator("rowheader", account, frame=LEDGER_FRAME)
    checks = [
        ResultCheck(
            CheckKind.RESULT,
            shown,
            value,
            output="balance",
            record=RecordEvidence(header, account, Relation.ROW),
        )
    ]
    if member:
        checks.append(
            ResultCheck(
                CheckKind.RECORD,
                AxLocator("heading", "Member 12345"),
                "12345",
                match=Match.CONTAINS,
            )
        )
    return Finish({"balance": value}, "read it", checks=tuple(checks))


def run_claims(site, site_profile, claims, goal=BALANCE_GOAL, **edits):
    profile = site_profile(**edits)
    decider = ScriptedDecider([Observe(STRUCTURED), *claims])
    journal = MemoryJournal()
    with open_session(profile, f"{site}/members/12345") as page:
        page._page.wait_for_load_state("load")
        result = discover(
            goal,
            profile,
            surface=page,
            decider=decider,
            escalator=ScriptedEscalatorLike(),
            journal=journal,
            clock=FakeClock(),
        )
    return result, decider, journal


def test_a_balance_lookup_verifies_member_account_and_value(site, site_profile):
    result, _, journal = run_claims(site, site_profile, [balance_claim("$4,212.55")])
    assert result.ending is Ending.COMPLETED
    assert result.verification is Verification.EXECUTOR
    assert [check.passed for check in result.checks] == [True, True]
    assert result.checks[0].shown == "$4,212.55"
    verified = [event for event in journal.events if isinstance(event, Verified)]
    assert verified == [Verified(step=2, checks=2, passed=2)]
    assert "4,212.55" not in json.dumps([as_record(e) for e in journal.events])


def test_an_unsupported_or_wrong_claim_does_not_complete(site, site_profile):
    """No checks, an account the goal never named, and another account's row."""
    claims = [
        Finish({"balance": "$4,212.55"}, "trust me"),
        balance_claim("$318.00", account="S-1002"),
        balance_claim("$318.00", account="S-1001"),
    ]
    budgets = {
        "max_steps": 24,
        "max_wall_clock_s": 240,
        "max_retries_per_step": 3,
        "max_navigations": 8,
    }
    result, decider, _ = run_claims(site, site_profile, claims, budgets=budgets)
    assert result.ending is Ending.HANDED_OFF
    assert result.verification is None
    notices = [notice for seen in decider.seen for notice in seen.notices]
    assert any("carries no checks" in notice for notice in notices)
    assert any("goal does not name" in notice for notice in notices)
    # The observation already shows the S-1001 header outside that row, so
    # the loop refuses the evidence before the surface reads anything.
    assert any("share no row" in notice for notice in notices)


def test_a_claim_is_corrected_after_a_failed_check(site, site_profile):
    claims = [
        balance_claim("$4,212.00", cell="$4,212.55"),
        balance_claim("$4,212.55"),
    ]
    result, decider, _ = run_claims(site, site_profile, claims)
    assert result.ending is Ending.COMPLETED
    assert any("the page shows '$4,212.55'" in n for n in decider.seen[2].notices)


def test_a_claim_nothing_can_check_goes_to_a_person(profile) -> None:
    """A surface with no read confirms only through a person, and says so."""
    from computeruse.actions import TargetForm

    surface = ScriptedSurface(
        [Screen("https://sandbox.example.test/members/12345")],
        supports={ActionKind.CLICK: frozenset({TargetForm.SCREEN})},
    )
    screen_check = ResultCheck(
        CheckKind.RESULT, ScreenTarget("obs-1", Point(5, 5)), "done", output="state"
    )
    operator = Approves(FakeClock())
    result = discover(
        "finish the task",
        profile,
        surface=surface,
        decider=ScriptedDecider(
            [Observe(VISUAL), Finish({"state": "done"}, "", checks=(screen_check,))]
        ),
        escalator=operator,
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert operator.requests[0].trigger is Trigger.UNVERIFIED_RESULT
    assert result.ending is Ending.COMPLETED
    assert result.verification is Verification.PERSON


def test_a_screenshot_check_reads_the_element_at_its_point(pages) -> None:
    with pages({"/": FIELDS_PAGE}) as (page, _):
        picture = page.observe(VISUAL)
        box = page._page.locator("#cell").bounding_box()
        assert box is not None
        point = Point(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        target = ScreenTarget(picture.observation_id, point)
        first = page.act(Action(ActionKind.READ, target))
        second = page.act(Action(ActionKind.READ, target))
    assert first.extracted == second.extracted == "$4,212.55"


# 25. A long run keeps the facts it needs, and checks a changeable one again.


def test_a_fact_from_early_in_a_long_run_is_used_late(site, site_profile) -> None:
    from computeruse.model import MAX_HISTORY, _state

    profile = site_profile(
        budgets={
            "max_steps": 40,
            "max_wall_clock_s": 240,
            "max_retries_per_step": 20,
            "max_navigations": 8,
        }
    )
    member = AxLocator("rowheader", "67890", scope=Scope(ScopeKind.ROW, "67890"))
    search = AxLocator("textbox", "Member number")
    row = Scope(ScopeKind.ROW, "12345")
    reads = [
        Propose(Action(ActionKind.READ, AxLocator("cell", "Riverside", scope=row)))
    ]
    reads = reads * 16

    def kept(transcript: Transcript) -> Decision:
        payload = json.loads(_state(transcript))
        assert len(transcript.history) > MAX_HISTORY
        assert all(turn["step"] > 3 for turn in payload["recent"])
        memory = {item["key"]: item["value"] for item in payload["working_memory"]}
        assert memory["member"] == "67890"
        return Propose(Action(ActionKind.TYPE, search, "67890"), fact="member")

    ref = {}

    def remember(transcript: Transcript) -> Decision:
        node = next(
            node
            for observation in transcript.observations
            for node in observation.nodes
            if names_node(member, node)
        )
        ref["control"] = node.control
        return Remember("member", "67890", may_change=False, source=member)

    moves: list[Move] = [
        lambda _: Observe(STRUCTURED),
        remember,
        *[(lambda decision: lambda _: decision)(item) for item in reads],
        kept,
        done,
    ]
    decider = Moves(moves)
    with open_session(profile, f"{site}/members") as page:
        result = discover(
            "Find member 67890 and search for that member",
            profile,
            surface=page,
            decider=decider,
            escalator=ScriptedEscalatorLike(),
            journal=MemoryJournal(),
            clock=FakeClock(),
        )
        typed = page._page.locator("#member-id").input_value()
    assert typed == "67890"
    assert result.ending is Ending.COMPLETED


def test_a_fact_the_model_did_not_read_is_not_kept(profile) -> None:
    surface = ScriptedSurface([Screen("https://sandbox.example.test/members/12345")])
    decider = ScriptedDecider(
        [
            Observe(STRUCTURED),
            Remember("member", "99999"),
            Remember("button", "Wrong", source=AxLocator("button", "Search")),
            Remember("member", "12345"),
            Observe(STRUCTURED),
        ]
    )
    discover(
        "Open member 12345",
        profile,
        surface=surface,
        decider=decider,
        escalator=ScriptedEscalatorLike(),
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    notices = [notice for seen in decider.seen for notice in seen.notices]
    assert any("must appear in the goal" in notice for notice in notices)
    assert any("its source shows another value" in notice for notice in notices)
    assert [fact.key for fact in decider.seen[-1].memory] == ["member"]


def test_a_changeable_fact_is_checked_again_before_use(site, site_profile) -> None:
    profile = site_profile()
    amount = DomLocator("input", DomAttribute.ID, "amount")
    memo = DomLocator("input", DomAttribute.ID, "memo")

    def change(page: BrowserSurface) -> Callable[[Transcript], Decision]:
        def _change(_: Transcript) -> Decision:
            page._page.fill("#amount", "99.00")
            return Propose(Action(ActionKind.TYPE, memo, "12.50"), fact="amount")

        return _change

    with open_session(profile, f"{site}/payments") as page:
        page.act(Action(ActionKind.TYPE, amount, "12.50"))
        decider = Moves(
            [
                lambda _: Observe(STRUCTURED),
                lambda _: Remember("amount", "12.50", may_change=True, source=amount),
                change(page),
                lambda _: Propose(Action(ActionKind.TYPE, memo, "x"), fact="amount"),
            ]
        )
        discover(
            "Copy the amount into the memo",
            profile,
            surface=page,
            decider=decider,
            escalator=ScriptedEscalatorLike(),
            journal=MemoryJournal(),
            clock=FakeClock(),
        )
        typed = page._page.locator("#memo").input_value()
    assert any("changed since step" in notice for notice in decider.seen[3].notices)
    assert typed == "99.00"


def test_memory_never_carries_a_secret_value(tall, site_profile) -> None:
    """A field holding a secret is never a fact, whatever value is claimed for it."""
    profile = site_profile()
    with tall() as page:
        page.act(Action(ActionKind.TYPE, BRANCH_CODE, TYPE_THE_SECRET))
        decider = ScriptedDecider(
            [
                Observe(STRUCTURED),
                Remember("code", BRANCH_SECRET, source=BRANCH_CODE),
                Observe(STRUCTURED),
            ]
        )
        journal = MemoryJournal()
        discover(
            "keep the branch code",
            profile,
            surface=page,
            decider=decider,
            escalator=ScriptedEscalatorLike(),
            journal=journal,
            clock=FakeClock(),
        )
    assert all(not seen.memory for seen in decider.seen)
    assert any(
        "was not kept" in notice for seen in decider.seen for notice in seen.notices
    )
    assert BRANCH_SECRET not in json.dumps([as_record(e) for e in journal.events])


# 26. A step that must name its record goes to a person when the run cannot.


def test_a_screen_click_on_a_record_route_is_performed_by_a_person(
    site, site_profile
) -> None:
    """The person approves in the same live session. The run then resumes."""
    profile = site_profile(
        perception={
            "allowed_modes": ["visual"],
            "max_alternate_observations_per_step": 0,
        }
    )
    approve = AxLocator("button", "Approve")
    number = DomLocator("dd", DomAttribute.ID, "member-number")

    class Person:
        def __init__(self, page: BrowserSurface) -> None:
            self.page = page
            self.requests: list[InterventionRequest] = []

        def request(self, intervention: InterventionRequest) -> Handoff:
            self.requests.append(intervention)
            if intervention.trigger is Trigger.RECORD_EVIDENCE_REQUIRED:
                proof = RecordEvidence(number, "10001", Relation.CONTAINER)
                self.page.act(Action(ActionKind.CLICK, approve, evidence=proof))
                return Handoff(HandoffOutcome.RESUMED, "approved member 10001")
            return Handoff(HandoffOutcome.TIMED_OUT)

    def click_approve(transcript: Transcript) -> Decision:
        picture = next(o for o in transcript.observations if o.image is not None)
        box = page._page.locator("#approve-slot button").bounding_box()
        assert box is not None
        point = Point(box["x"] + 5, box["y"] + 5)
        target = ScreenTarget(picture.observation_id, point)
        return Propose(Action(ActionKind.CLICK, target, effect="approve_member"))

    def check_state(transcript: Transcript) -> Decision:
        picture = next(o for o in transcript.observations if o.image is not None)
        box = page._page.locator("#approved-member").bounding_box()
        assert box is not None
        point = Point(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        check = ResultCheck(
            CheckKind.RESULT,
            ScreenTarget(picture.observation_id, point),
            "10001",
            output="approved",
        )
        return Finish({"approved": "10001"}, "", checks=(check,))

    with open_session(profile, f"{site}/queue") as page:
        person = Person(page)
        decider = Moves(
            [
                lambda _: Observe(VISUAL),
                click_approve,
                lambda _: Observe(VISUAL),
                check_state,
            ]
        )
        result = discover(
            "Approve member 10001",
            profile,
            surface=page,
            decider=decider,
            escalator=person,
            journal=MemoryJournal(),
            clock=FakeClock(),
        )
        approved = read(page, APPROVED_MEMBER)
    assert [r.trigger for r in person.requests] == [Trigger.RECORD_EVIDENCE_REQUIRED]
    assert approved == "10001"
    assert result.ending is Ending.COMPLETED
    assert result.verification is Verification.EXECUTOR


def test_an_approval_does_not_perform_a_step_that_needs_evidence(
    site, site_profile
) -> None:
    profile = site_profile(
        perception={
            "allowed_modes": ["visual"],
            "max_alternate_observations_per_step": 0,
        }
    )

    def click_approve(transcript: Transcript) -> Decision:
        picture = next(o for o in transcript.observations if o.image is not None)
        target = ScreenTarget(picture.observation_id, Point(40, 200))
        return Propose(Action(ActionKind.CLICK, target, effect="approve_member"))

    operator = Approves(FakeClock())
    with open_session(profile, f"{site}/queue") as page:
        decider = Moves([lambda _: Observe(VISUAL), click_approve])
        discover(
            "Approve member 10001",
            profile,
            surface=page,
            decider=decider,
            escalator=operator,
            journal=MemoryJournal(),
            clock=FakeClock(),
        )
        approved = read(page, APPROVED_MEMBER)
    assert operator.requests[0].trigger is Trigger.RECORD_EVIDENCE_REQUIRED
    assert approved == "none"
    assert any(
        "an approval does not perform" in notice
        for seen in decider.seen
        for notice in seen.notices
    )


@pytest.mark.parametrize(
    ("may_change", "source", "completed"),
    [(False, True, True), (True, True, False), (False, False, False)],
)
def test_a_result_read_earlier_is_checked_through_a_stable_fact(
    profile, may_change: bool, source: bool, completed: bool
) -> None:
    """Only a fact read from a control, and marked as unchanging, backs a result."""
    from computeruse.decider import FactRef

    branch = AxLocator("cell", "Eastgate")
    surface = ScriptedSurface(
        [
            Screen(
                "https://sandbox.example.test/members",
                controls=(("cell", "Eastgate"),),
            ),
            Screen(
                "https://sandbox.example.test/members",
                controls=(("cell", "Eastgate"),),
            ),
            Screen("https://sandbox.example.test/members/12345"),
        ],
        extracts=["Eastgate"],
    )
    check = ResultCheck(
        CheckKind.RESULT, FactRef("branch"), "Eastgate", output="branch"
    )
    decider = ScriptedDecider(
        [
            Observe(STRUCTURED),
            Remember(
                "branch",
                "Eastgate",
                may_change=may_change,
                source=branch if source else None,
            ),
            Propose(Action(ActionKind.CLICK, branch, effect="open_member")),
            Finish({"branch": "Eastgate"}, "", checks=(check,)),
        ]
    )
    result = discover(
        "Note the Eastgate branch",
        profile,
        surface=surface,
        decider=decider,
        escalator=ScriptedEscalatorLike(),
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert (result.ending is Ending.COMPLETED) is completed


@pytest.mark.parametrize(
    ("between", "completed"), [(ActionKind.SCROLL, True), (ActionKind.CLICK, False)]
)
def test_a_value_read_on_this_page_backs_a_check_until_the_page_changes(
    profile, between: ActionKind, completed: bool
) -> None:
    """A heading read before scrolling still counts. After a click, it does not.

    The fact backs a result. A screenshot read cannot back a record check,
    because the point may be a field holding what the run typed.
    """
    from computeruse.decider import FactRef

    screen = Screen("https://sandbox.example.test/members/12345")
    heading = ScreenTarget("obs-1", Point(10, 10))
    surface = ScriptedSurface([screen], extracts=["Member 12345"])
    moved = (
        Action(ActionKind.SCROLL, value="down")
        if between is ActionKind.SCROLL
        else Action(ActionKind.CLICK, AxLocator("button", "Search"), effect="search")
    )
    record = ResultCheck(
        CheckKind.RESULT,
        FactRef("read_at_step_2"),
        "12345",
        match=Match.CONTAINS,
        output="member",
    )
    decider = ScriptedDecider(
        [
            Observe(VISUAL),
            Propose(Action(ActionKind.READ, heading)),
            Observe(STRUCTURED),
            Propose(moved),
            Finish(
                {"status": "Task done", "member": "12345"},
                "",
                checks=(*DONE.checks, record),
            ),
        ]
    )
    result = discover(
        "Check member 12345",
        profile,
        surface=surface,
        decider=decider,
        escalator=ScriptedEscalatorLike(),
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert (result.ending is Ending.COMPLETED) is completed


def test_a_frame_left_by_the_previous_document_does_not_share_its_name(pages) -> None:
    """A removed frame the driver still lists must not make a live one ambiguous.

    Found on the Northstar portal, whose shell names its only workspace frame
    the same on every screen.
    """
    written = {
        **FRAME_PAGES,
        "/first-shell": '<h1>Shell</h1><iframe name="work" src="/left"></iframe>',
        "/second-shell": '<h1>Shell</h1><iframe name="work" src="/right"></iframe>',
    }
    with pages(written, "/first-shell") as (page, _):
        page._page.goto(page.location().replace("first", "second"))
        loaded(page)
        seen = page.observe(STRUCTURED)
        assert frame_of(seen, "right") == ("work",)
        go = AxLocator("button", "Go", frame=("work",))
        assert page.act(Action(ActionKind.CLICK, go)).outcome is Outcome.OK
        page._page.evaluate(
            """() => {
              const old = document.querySelector('iframe');
              const next = document.createElement('iframe');
              next.name = 'work';
              next.src = '/solo';
              old.replaceWith(next);
            }"""
        )
        page._page.wait_for_timeout(500)
        again = page.observe(STRUCTURED)
        assert frame_of(again, "solo") == ("work",)
        assert page.act(Action(ActionKind.CLICK, go)).outcome is Outcome.OK


# 27. Found in review: each of these fails if its fix is removed.


def test_a_secret_is_never_typed_into_an_editable_region(pages, monkeypatch) -> None:
    """An editable region keeps its text in the document, where names read it."""
    monkeypatch.setenv("APPROVER_PASSCODE", BRANCH_SECRET)
    written = {
        "/": '<h1>Notes</h1><div>Note: <div id="note" contenteditable="true" '
        'aria-label="Note"></div></div>'
    }
    with pages(written) as (page, _):
        note = AxLocator("textbox", "Note")
        typed = page.act(Action(ActionKind.TYPE, note, TYPE_THE_SECRET))
        picture = page.observe(VISUAL)
        box = page._page.locator("#note").bounding_box()
        assert box is not None
        focus = page.act(
            Action(
                ActionKind.CLICK,
                ScreenTarget(picture.observation_id, Point(box["x"] + 5, box["y"] + 5)),
            )
        )
        assert focus.outcome is Outcome.OK
        later = page.observe(VISUAL)
        pressed = page.act(
            Action(ActionKind.TYPE, ScreenTarget(later.observation_id), TYPE_THE_SECRET)
        )
        text = page._page.locator("#note").inner_text()
        seen = page.observe(STRUCTURED)
    assert typed.outcome is Outcome.NOT_ACTIONABLE
    assert pressed.outcome is Outcome.NOT_ACTIONABLE
    assert BRANCH_SECRET not in text
    assert all(BRANCH_SECRET not in node.name for node in seen.nodes)


def test_a_field_holding_a_secret_left_behind_blocks_reading(tall) -> None:
    """A read, like an observation, waits until the protected field is found."""
    with tall() as page:
        page.act(Action(ActionKind.TYPE, BRANCH_CODE, TYPE_THE_SECRET))
        click(page, REDRAW_BRANCH_CODE)
        result = page.act(Action(ActionKind.READ, BRANCH_CODE))
    assert result.outcome is Outcome.NOT_ACTIONABLE
    assert BRANCH_SECRET not in (result.extracted or "")


def test_a_denied_effect_is_not_handed_to_a_person_for_want_of_evidence(
    profile,
) -> None:
    surface = ScriptedSurface(
        [Screen("https://sandbox.example.test/members/12345/savings/close")]
    )
    operator = Approves(FakeClock())
    target = ScreenTarget("obs-1", Point(10, 10))
    discover(
        "close the account",
        profile,
        surface=surface,
        decider=ScriptedDecider(
            [
                Observe(VISUAL),
                Propose(Action(ActionKind.CLICK, target, effect="close_account")),
            ]
        ),
        escalator=operator,
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert all(
        request.trigger is not Trigger.RECORD_EVIDENCE_REQUIRED
        for request in operator.requests
    )
    assert surface.acted == []


def test_a_record_check_cannot_read_what_the_run_typed(pages) -> None:
    with pages({"/": FIELDS_PAGE}) as (page, _):
        amount = DomLocator("input", DomAttribute.ID, "amount")
        page.act(Action(ActionKind.TYPE, amount, "12345"))
        seen = page.observe(STRUCTURED)
        shown_only = Expectation(seen.page_state, displayed=True)
        field = page.act(Action(ActionKind.READ, amount), expect=shown_only)
        cell = DomLocator("td", DomAttribute.ID, "cell")
        text = page.act(Action(ActionKind.READ, cell), expect=shown_only)
    assert field.outcome is Outcome.NOT_ACTIONABLE
    assert text.extracted == "$4,212.55"


def test_a_kept_fact_is_read_again_while_its_control_is_on_the_page(
    site, site_profile
) -> None:
    """The model calling a value unchanging does not stop the loop re-reading it."""
    from computeruse.decider import FactRef

    profile = site_profile()
    amount = DomLocator("input", DomAttribute.ID, "amount")
    check = ResultCheck(CheckKind.RESULT, FactRef("amount"), "12.50", output="amount")

    def change(page: BrowserSurface) -> Move:
        def _change(_: Transcript) -> Decision:
            page._page.fill("#amount", "99.00")
            return Finish({"amount": "12.50"}, "", checks=(check,))

        return _change

    with open_session(profile, f"{site}/payments") as page:
        page.act(Action(ActionKind.TYPE, amount, "12.50"))
        decider = Moves(
            [
                lambda _: Observe(STRUCTURED),
                lambda _: Remember("amount", "12.50", may_change=False, source=amount),
                change(page),
            ]
        )
        result = discover(
            "Report the amount",
            profile,
            surface=page,
            decider=decider,
            escalator=ScriptedEscalatorLike(),
            journal=MemoryJournal(),
            clock=FakeClock(),
        )
    assert result.ending is not Ending.COMPLETED
    assert any("the page shows '99.00'" in n for n in decider.seen[-1].notices)


def test_a_dialog_from_a_page_outside_the_profile_is_dismissed(pages) -> None:
    written = {
        **POPUP_PAGES,
        "/blank-alert": """<!doctype html><html><body><h1>Opener</h1>
<button onclick="const w = window.open();
  setTimeout(() => w.alert('from nowhere'), 200)">Open blank</button>
</body></html>""",
    }
    with pages(written, "/blank-alert", allow_new_windows=True) as (page, _):
        opened = page.act(Action(ActionKind.CLICK, AxLocator("button", "Open blank")))
        page._page.wait_for_timeout(600)
        effects = [*opened.side_effects, *page._result(Outcome.OK).side_effects]
        seen = page.observe(STRUCTURED)
    assert any("outside the profile, was dismissed" in effect for effect in effects)
    assert seen.status is ObservationStatus.COMPLETE


# 28. Every read the run makes for itself passes the gate, facts stay bound to
# their records, and completion covers the task the goal set.

MEMBER_PAGE = "https://sandbox.example.test/members/12345"
FACT_CHECK = ResultCheck(
    CheckKind.RESULT, FactRef("status"), "Task done", output="status"
)
KEEP_STATUS = Remember("status", "Task done", may_change=True, source=DONE_TARGET)


def fact_run(profile, decisions, handoffs=()):
    surface = ScriptedSurface(
        [Screen(MEMBER_PAGE, controls=(("status", "Task done"),))]
    )
    decider = ScriptedDecider(list(decisions))
    escalator = ScriptedEscalator(list(handoffs))
    journal = MemoryJournal()
    result = discover(
        "report the status",
        profile,
        surface=surface,
        decider=decider,
        escalator=escalator,
        journal=journal,
        clock=FakeClock(),
    )
    return result, surface, decider, escalator


def claim(*checks: ResultCheck) -> Finish:
    return Finish({"status": "Task done"}, "", checks=checks)


def test_a_fact_check_does_not_read_when_read_is_undeclared(edited_profile) -> None:
    """Found in review: a fact check read the surface with no policy check."""
    actions = grants({"observe": "safe", "click": "safe"})
    profile = edited_profile(actions=actions, records={"actions": [], "routes": []})
    result, surface, decider, escalator = fact_run(
        profile, [Observe(STRUCTURED), KEEP_STATUS, claim(FACT_CHECK)]
    )
    assert surface.checked == []
    assert [item for item in surface.acted if item.kind is ActionKind.READ] == []
    assert not decider.seen[-1].memory
    assert escalator.requests
    assert result.verification is not Verification.EXECUTOR


@pytest.mark.parametrize("answer", [HandoffOutcome.APPROVED, HandoffOutcome.REJECTED])
def test_a_fact_check_asks_before_a_risky_read(edited_profile, answer) -> None:
    actions = grants({"observe": "safe", "read": "risky"})
    profile = edited_profile(actions=actions, records={"actions": [], "routes": []})
    result, surface, _, escalator = fact_run(
        profile,
        [Observe(STRUCTURED), KEEP_STATUS, claim(FACT_CHECK)],
        handoffs=[Handoff(answer), Handoff(answer)],
    )
    first = escalator.requests[0]
    assert first.trigger is Trigger.RISKY_ACTION
    assert first.action is not None
    assert first.action.kind is ActionKind.READ
    if answer is HandoffOutcome.APPROVED:
        assert len(surface.checked) == 2
        assert len(escalator.requests) == 2
        assert result.verification is Verification.EXECUTOR
    else:
        assert surface.checked == []
        assert result.ending is not Ending.COMPLETED


def test_a_permitted_safe_read_completes_without_a_person(edited_profile) -> None:
    profile = edited_profile(records={"actions": [], "routes": []})
    result, surface, _, escalator = fact_run(
        profile, [Observe(STRUCTURED), KEEP_STATUS, claim(FACT_CHECK)]
    )
    assert escalator.requests == []
    assert len(surface.checked) == 2
    assert result.ending is Ending.COMPLETED
    assert result.verification is Verification.EXECUTOR


def test_a_learned_restriction_holds_for_a_completion_read(edited_profile) -> None:
    """A read the model flagged needs a person when a check makes it again."""
    profile = edited_profile(records={"actions": [], "routes": []})
    result, surface, _, escalator = fact_run(
        profile,
        [
            Observe(STRUCTURED),
            Propose(Action(ActionKind.READ, DONE_TARGET)),
            FlagRisk(RiskFinding(2, "read_member_data")),
            claim(DONE.checks[0]),
        ],
    )
    assert len(surface.checked) == 1
    assert escalator.requests[0].trigger is Trigger.RISKY_ACTION
    assert result.ending is not Ending.COMPLETED


@pytest.mark.parametrize("check", [DONE.checks[0], FACT_CHECK], ids=["direct", "fact"])
def test_required_evidence_holds_for_a_completion_read(edited_profile, check) -> None:
    """Neither a check nor a fact it cites reads a record route without evidence."""
    actions = grants({"observe": "safe", "read": "risky"})
    profile = edited_profile(
        actions=actions, records={"actions": ["read"], "routes": ["/members/:id"]}
    )
    _, surface, decider, escalator = fact_run(
        profile, [Observe(STRUCTURED), KEEP_STATUS, claim(check)]
    )
    assert surface.checked == []
    # The gate refuses for missing evidence before risk is considered, so
    # nobody is asked to approve a read that could not name its record.
    assert Trigger.RISKY_ACTION not in [
        request.trigger for request in escalator.requests
    ]
    notices = [notice for seen in decider.seen for notice in seen.notices]
    assert any("record_evidence_missing" in notice for notice in notices)


RECORD_PAGES = {
    "/members/12345": """<!doctype html><html><body><h1>Member 12345</h1>
<dl><dt>Member number</dt><dd id="member-number">12345</dd>
<dt>Balance</dt><dd id="balance">$100.00</dd></dl>
<label>Amount <input id="amount"></label>
<a href="/members/24680">Next member</a></body></html>""",
    "/members/24680": """<!doctype html><html><body><h1>Member 24680</h1>
<dl><dt>Member number</dt><dd id="member-number">24680</dd>
<dt>Balance</dt><dd id="balance">$900.00</dd></dl>
<label>Amount <input id="amount"></label></body></html>""",
    "/member": """<!doctype html><html><body><h1>Member view</h1>
<dl><dt>Member number</dt><dd id="member-number">12345</dd>
<dt>Balance</dt><dd id="balance">$100.00</dd></dl>
<label>Amount <input id="amount"></label>
<button onclick="document.getElementById('member-number').textContent = '24680';
  document.getElementById('balance').textContent =
  new URLSearchParams(location.search).get('next') || '$900.00'"
  >Load the next member</button></body></html>""",
    "/search": """<!doctype html><html><body><h1>Member search</h1>
<table><tr><th scope="col">Member</th><th scope="col">Branch</th></tr>
<tr><th scope="row">12345</th><td>Riverside</td></tr>
<tr><th scope="row">24680</th><td>Riverside</td></tr></table>
<a href="/members/12345">Open 12345</a></body></html>""",
}

NUMBER = DomLocator("dd", DomAttribute.ID, "member-number")
BALANCE_DD = DomLocator("dd", DomAttribute.ID, "balance")
AMOUNT_FIELD = DomLocator("input", DomAttribute.ID, "amount")
OF_12345 = RecordEvidence(NUMBER, "12345", Relation.CONTAINER)


def record_run(pages, path, moves, goal="Report member 12345's balance.", task=None):
    with pages(RECORD_PAGES, path) as (page, profile):
        decider = Moves(moves, task=task or Task())
        person = ScriptedEscalator()
        journal = MemoryJournal()
        result = discover(
            goal,
            profile,
            surface=page,
            decider=decider,
            escalator=person,
            journal=journal,
            clock=FakeClock(),
        )
        typed = page._page.locator("#amount").input_value()
    return result, decider, person, journal, typed


def type_fact(_: Transcript) -> Decision:
    return Propose(
        Action(ActionKind.TYPE, AMOUNT_FIELD, "x"), "copy it", fact="balance"
    )


def test_a_fact_is_not_refreshed_from_another_members_page(pages) -> None:
    """Found in review: a shared route template let 24680's balance replace 12345's."""
    moves: list[Move] = [
        look,
        lambda _: Remember("balance", "$100.00", source=BALANCE_DD),
        do(Action(ActionKind.CLICK, AxLocator("link", "Next member"))),
        look,
        type_fact,
    ]
    _, decider, _, _, typed = record_run(pages, "/members/12345", moves)
    assert typed == ""
    kept = {fact.key: fact for fact in decider.seen[-1].memory}
    assert kept["balance"].value == "$100.00"
    assert kept["balance"].location.endswith("/members/12345")
    assert any("kept on another page" in n for n in decider.seen[-1].notices)


@pytest.mark.parametrize("record", [None, OF_12345], ids=["untied", "tied"])
@pytest.mark.parametrize("shown", ["$900.00", "$100.00"], ids=["other", "same"])
def test_a_fact_is_not_refreshed_from_another_record_at_one_address(
    pages, record, shown
) -> None:
    """The page swaps to member 24680 in place. The fact stays with 12345.

    The same displayed value does not make the refresh pass either, because
    it is the record, not the value, that is checked.
    """
    moves: list[Move] = [
        look,
        lambda _: Remember("balance", "$100.00", source=BALANCE_DD, record=record),
        do(Action(ActionKind.CLICK, AxLocator("button", "Load the next member"))),
        look,
        type_fact,
    ]
    written = {**RECORD_PAGES}
    with pages(written, f"/member?next={shown}") as (page, profile):
        decider = Moves(moves)
        discover(
            "Copy member 12345's balance",
            profile,
            surface=page,
            decider=decider,
            escalator=ScriptedEscalator(),
            journal=MemoryJournal(),
            clock=FakeClock(),
        )
        typed = page._page.locator("#amount").input_value()
    assert typed == ""
    kept = {fact.key: fact for fact in decider.seen[-1].memory}
    assert kept["balance"].value == "$100.00"
    assert kept["balance"].record == record


def test_a_fact_tied_to_its_record_is_refreshed_on_that_record(pages) -> None:
    moves: list[Move] = [
        look,
        lambda _: Remember("balance", "$100.00", source=BALANCE_DD, record=OF_12345),
        do(Action(ActionKind.CLICK, AxLocator("label", "Amount"))),
        look,
        type_fact,
    ]
    _, _, _, _, typed = record_run(pages, "/members/12345", moves)
    assert typed == "$100.00"


def test_a_field_cannot_tie_a_fact_to_a_record(pages) -> None:
    """Typed input is not the page saying which record it shows."""
    moves: list[Move] = [
        look,
        do(Action(ActionKind.TYPE, AMOUNT_FIELD, "12345")),
        look,
        lambda _: Remember(
            "balance",
            "$100.00",
            source=BALANCE_DD,
            record=RecordEvidence(AMOUNT_FIELD, "12345", Relation.CONTAINER),
        ),
    ]
    _, decider, _, _, _ = record_run(pages, "/members/12345", moves)
    assert decider.seen[-1].memory == ()
    assert any("cannot name a record" in n for n in decider.seen[-1].notices)


MEMBER_TASK = Task((TaskRecord("member", "12345"),), (TaskOutput("balance", "member"),))
BALANCE_CHECK = ResultCheck(CheckKind.RESULT, BALANCE_DD, "$100.00", output="balance")
MEMBER_CHECK = ResultCheck(CheckKind.RECORD, NUMBER, "12345")


def finish_with(*checks: ResultCheck, value: str = "$100.00") -> Move:
    return lambda _: Finish({"balance": value}, "", checks=checks)


def test_a_claim_about_another_member_does_not_complete(pages) -> None:
    """Found in review: a balance check alone completed on member 99999's page."""
    other = {
        "/members/99999": RECORD_PAGES["/members/12345"].replace("12345", "99999"),
    }
    moves: list[Move] = [look, finish_with(BALANCE_CHECK)] * 3
    with pages(other, "/members/99999") as (page, profile):
        decider = Moves(moves, task=MEMBER_TASK)
        person = ScriptedEscalator()
        journal = MemoryJournal()
        result = discover(
            "Report member 12345's balance.",
            profile,
            surface=page,
            decider=decider,
            escalator=person,
            journal=journal,
            clock=FakeClock(),
        )
    assert result.ending is not Ending.COMPLETED
    # The checks pass but cover the wrong member, so a person is asked at
    # once, with every gap named, rather than the model retrying.
    confirm = person.requests[-1]
    assert confirm.trigger is Trigger.UNVERIFIED_RESULT
    assert dict(confirm.outputs) == {"balance": "$100.00"}
    assert "balance is not tied to member 12345" in confirm.unverified
    assert "member 12345 is not shown" in confirm.unverified
    recorded = json.dumps([as_record(event) for event in journal.events])
    assert "$100.00" not in recorded
    assert "12345" not in recorded


def test_a_record_check_on_the_wrong_member_fails(pages) -> None:
    other = {
        "/members/99999": RECORD_PAGES["/members/12345"].replace("12345", "99999"),
    }
    tied = dataclasses.replace(BALANCE_CHECK, record=OF_12345)
    moves: list[Move] = [look, finish_with(tied, MEMBER_CHECK)]
    with pages(other, "/members/99999") as (page, profile):
        decider = Moves(moves, task=MEMBER_TASK)
        result = discover(
            "Report member 12345's balance.",
            profile,
            surface=page,
            decider=decider,
            escalator=ScriptedEscalator(),
            journal=MemoryJournal(),
            clock=FakeClock(),
        )
    assert result.ending is not Ending.COMPLETED
    assert result.verification is None


def test_a_correctly_bound_result_completes(pages) -> None:
    tied = dataclasses.replace(BALANCE_CHECK, record=OF_12345)
    result, _, person, journal, _ = record_run(
        pages, "/members/12345", [look, finish_with(tied)], task=MEMBER_TASK
    )
    assert person.requests == []
    assert result.ending is Ending.COMPLETED
    assert result.verification is Verification.EXECUTOR
    assert [item.bound for item in result.checks] == ["12345"]
    interpreted = [event for event in journal.events if isinstance(event, Interpreted)]
    assert interpreted == [Interpreted(records=1, outputs=1)]


ACCOUNT_PAGE = {
    "/members/12345": """<!doctype html><html><body><h1>Member 12345</h1>
<section aria-label="Member"><dl><dt>Member number</dt>
<dd id="member-number">12345</dd></dl>
<table><tr><th scope="col">Account</th><th scope="col">Balance</th></tr>
<tr><th scope="row">S-1001</th><td>$4.00</td></tr>
<tr><th scope="row">S-1002</th><td>$9.00</td></tr></table></section>
</body></html>""",
}
ACCOUNT_TASK = Task(
    (TaskRecord("member", "12345"), TaskRecord("account", "S-1001", within="member")),
    (TaskOutput("balance", "account"),),
)
ACCOUNT_HEADER = AxLocator("rowheader", "S-1001")
ACCOUNT_BALANCE = ResultCheck(
    CheckKind.RESULT,
    DomLocator("td", DomAttribute.TEXT, "$4.00"),
    "$4.00",
    output="balance",
    record=RecordEvidence(ACCOUNT_HEADER, "S-1001", Relation.ROW),
)


@pytest.mark.parametrize("linked", [True, False])
def test_an_account_must_be_shown_to_belong_to_its_member(pages, linked) -> None:
    account = ResultCheck(
        CheckKind.RECORD, ACCOUNT_HEADER, "S-1001", record=OF_12345 if linked else None
    )
    checks = (ACCOUNT_BALANCE, account, MEMBER_CHECK)
    moves: list[Move] = [look, finish_with(*checks, value="$4.00")] * 3
    with pages(ACCOUNT_PAGE, "/members/12345") as (page, profile):
        decider = Moves(moves, task=ACCOUNT_TASK)
        person = ScriptedEscalator()
        result = discover(
            "Report the balance of account S-1001 for member 12345.",
            profile,
            surface=page,
            decider=decider,
            escalator=person,
            journal=MemoryJournal(),
            clock=FakeClock(),
        )
    if linked:
        assert result.verification is Verification.EXECUTOR
        return
    assert result.ending is not Ending.COMPLETED
    gap = "account S-1001 is not shown to belong to member 12345"
    assert gap in person.requests[-1].unverified


def test_members_sharing_a_branch_are_told_apart_by_record(pages) -> None:
    """Both members are at Riverside. Only a fact tied to 24680 is their branch."""
    branch = AxLocator("cell", "Riverside", scope=Scope(ScopeKind.ROW, "24680"))
    task = Task((TaskRecord("member", "24680"),), (TaskOutput("branch", "member"),))

    def run(
        record: RecordEvidence | None, key: str, source: AxLocator = branch
    ) -> RunResult:
        check = ResultCheck(
            CheckKind.RESULT, FactRef(key), "Riverside", output="branch"
        )
        moves: list[Move] = [
            look,
            lambda _: Remember(
                key, "Riverside", may_change=False, source=source, record=record
            ),
            do(Action(ActionKind.CLICK, AxLocator("link", "Open 12345"))),
            look,
            lambda _: Finish({"branch": "Riverside"}, "", checks=(check,)),
        ] + [look, lambda _: Finish({"branch": "Riverside"}, "", checks=(check,))] * 2
        with pages(RECORD_PAGES, "/search") as (page, profile):
            return discover(
                "Report the branch of member 24680.",
                profile,
                surface=page,
                decider=Moves(moves, task=task),
                escalator=ScriptedEscalator(),
                journal=MemoryJournal(),
                clock=FakeClock(),
            )

    header = AxLocator("rowheader", "24680")
    tied = run(RecordEvidence(header, "24680", Relation.ROW), "branch_24680")
    assert tied.verification is Verification.EXECUTOR
    assert [item.bound for item in tied.checks] == ["24680"]
    untied = run(None, "branch_24680")
    assert untied.verification is None
    # 12345's own row says Riverside too, and a fact tied to 12345 is not
    # 24680's branch whatever the model named it.
    other_row = dataclasses.replace(branch, scope=Scope(ScopeKind.ROW, "12345"))
    other = RecordEvidence(AxLocator("rowheader", "12345"), "12345", Relation.ROW)
    wrong = run(other, "branch_24680", other_row)
    assert wrong.verification is None


def test_an_ambiguous_task_goes_to_a_person_first(profile) -> None:
    unclear = Task(
        (TaskRecord("member", "12345"),), (), "Is 12345 a member or an account?"
    )
    surface = ScriptedSurface([Screen(MEMBER_PAGE)])
    decider = ScriptedDecider([Observe(STRUCTURED), DONE], tasks=[unclear])
    escalator = ScriptedEscalator([Handoff(HandoffOutcome.APPROVED)])
    journal = MemoryJournal()
    result = discover(
        "Check 12345",
        profile,
        surface=surface,
        decider=decider,
        escalator=escalator,
        journal=journal,
        clock=FakeClock(),
    )
    first = escalator.requests[0]
    assert first.trigger is Trigger.AMBIGUOUS_TASK
    assert first.step == 0
    assert "member 12345" in first.reason
    assert surface.looks[0].mode is ObservationMode.STRUCTURED
    triggers = [request.trigger for request in escalator.requests]
    assert triggers.count(Trigger.AMBIGUOUS_TASK) == 1
    assert decider.seen[0].task.question == ""
    assert Interpreted(records=1, outputs=0, confirmed=True) in journal.events
    # The member was never shown, so the claim cannot complete on its own.
    assert result.verification is not Verification.EXECUTOR


def test_a_refused_reading_is_asked_again_with_the_reason(profile) -> None:
    wrong = Task((TaskRecord("member", "99999"),))
    unclear = Task((TaskRecord("member", "12345"),), (), "Which member?")
    decider = ScriptedDecider([DONE], tasks=[wrong, unclear, Task()])
    escalator = ScriptedEscalator([Handoff(HandoffOutcome.REJECTED, "it is a report")])
    discover(
        "Check 12345",
        profile,
        surface=ScriptedSurface([Screen(MEMBER_PAGE)]),
        decider=decider,
        escalator=escalator,
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert decider.interpreted[0] == ()
    assert "99999 is not an identifier the goal names" in decider.interpreted[1][0]
    assert "operator: it is a report" in decider.interpreted[2]


def test_a_task_about_no_record_completes_without_one(profile) -> None:
    title = Task((), (TaskOutput("status"),))
    surface = ScriptedSurface([Screen(MEMBER_PAGE)])
    result = discover(
        "Report the status shown",
        profile,
        surface=surface,
        decider=ScriptedDecider([Observe(STRUCTURED), DONE], tasks=[title]),
        escalator=ScriptedEscalator(),
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert result.verification is Verification.EXECUTOR


def test_a_claim_cannot_drop_an_output_the_task_asks_for(profile) -> None:
    both = Task((), (TaskOutput("status"), TaskOutput("balance")))
    escalator = ScriptedEscalator()
    result = discover(
        "Report the status and the balance",
        profile,
        surface=ScriptedSurface([Screen(MEMBER_PAGE)]),
        decider=ScriptedDecider([Observe(STRUCTURED), DONE] * 3, tasks=[both]),
        escalator=escalator,
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert result.ending is not Ending.COMPLETED
    assert any("'balance'" in item for item in escalator.requests[-1].unverified)


# 29. Found in review of 6880ed3: an approval, a budget, and a wait could each
# let the run act on a state nobody decided about.

APPROVAL_PAGES = {
    "/start": """<!doctype html><html><body><h1>Status ready</h1>
<p id="status">ready</p></body></html>""",
    "/restricted": """<!doctype html><html><body><h1>Status ready</h1>
<p id="status">ready</p></body></html>""",
}
STATUS_P = DomLocator("p", DomAttribute.ID, "status")
STATUS_CLAIM = Finish(
    {"status": "ready"},
    "",
    checks=(ResultCheck(CheckKind.RESULT, STATUS_P, "ready", output="status"),),
)


@dataclasses.dataclass
class Approver:
    """A person who may move the page, then approves the read they were shown."""

    page: BrowserSurface
    goes_to: str = ""
    requests: list[InterventionRequest] = dataclasses.field(default_factory=list)

    def request(self, intervention: InterventionRequest) -> Handoff:
        self.requests.append(intervention)
        if intervention.trigger is not Trigger.RISKY_ACTION:
            return Handoff(HandoffOutcome.TIMED_OUT)
        if self.goes_to:
            here = self.page.location()
            self.page._page.goto(here.rsplit("/", 1)[0] + self.goes_to)
            self.goes_to = ""
        return Handoff(HandoffOutcome.APPROVED)


@pytest.mark.parametrize("moves_to", ["/restricted", ""], ids=["moved", "unchanged"])
def test_an_approved_read_is_rechecked_where_it_runs(pages, moves_to) -> None:
    """A person moving from /start to /restricted voids the approval.

    Found in review: with no observation in hand, the read ran on the page the
    person moved to. That page required record evidence, but the run completed
    with executor verification.
    """
    edits = {
        "actions": grants({"observe": "safe", "read": "risky"}),
        "records": {"actions": ["read"], "routes": ["/restricted"]},
    }
    with pages(APPROVAL_PAGES, "/start", **edits) as (page, profile):
        person = Approver(page, moves_to)
        decider = Moves([lambda _: STATUS_CLAIM])
        result = discover(
            "report the status",
            profile,
            surface=page,
            decider=decider,
            escalator=person,
            journal=MemoryJournal(),
            clock=FakeClock(),
        )
        landed = page.location()
    assert person.requests[0].trigger is Trigger.RISKY_ACTION
    if not moves_to:
        assert result.verification is Verification.EXECUTOR
        return
    assert landed.endswith("/restricted")
    assert result.verification is not Verification.EXECUTOR
    assert result.ending is not Ending.COMPLETED
    notices = [notice for seen in decider.seen for notice in seen.notices]
    assert any("approval was discarded" in notice for notice in notices)
    approvals = [r for r in person.requests if r.trigger is Trigger.RISKY_ACTION]
    assert len(approvals) == 1


@dataclasses.dataclass
class TimedSurface(ScriptedSurface):
    """A scripted surface whose every read takes ``read_s`` of the clock."""

    clock: FakeClock | None = None
    read_s: float = 0.0
    reads: int = 0

    def act(self, action: Action, *, expect: Expectation | None = None) -> ActionResult:
        if action.kind is ActionKind.READ:
            self.reads += 1
            if self.clock is not None:
                self.clock.advance(self.read_s)
        return ScriptedSurface.act(self, action, expect=expect)


def timed_run(edited_profile, decisions, *, read_s, handoffs=(), wait_s=0.0, **edits):
    clock = FakeClock()
    budgets = {
        "max_steps": 40,
        "max_wall_clock_s": 5,
        "max_retries_per_step": 2,
        "max_navigations": 15,
    }
    profile = edited_profile(
        budgets=budgets, records={"actions": [], "routes": []}, **edits
    )
    surface = TimedSurface(
        [Screen(MEMBER_PAGE, controls=(("status", "Task done"),))],
        clock=clock,
        read_s=read_s,
    )

    @dataclasses.dataclass
    class Waiting(ScriptedEscalator):
        def request(self, intervention: InterventionRequest) -> Handoff:
            clock.advance(wait_s)
            return ScriptedEscalator.request(self, intervention)

    escalator = Waiting(list(handoffs))
    result = discover(
        "report the status",
        profile,
        surface=surface,
        decider=ScriptedDecider(list(decisions)),
        escalator=escalator,
        journal=MemoryJournal(),
        clock=clock,
    )
    return result, surface, clock, escalator


def test_no_completion_read_starts_after_the_budget_expires(edited_profile) -> None:
    """Found in review: a second read started at ten seconds of a five-second budget."""
    twice = claim(DONE.checks[0], DONE.checks[0])
    result, surface, clock, _ = timed_run(
        edited_profile, [Observe(STRUCTURED), twice], read_s=10
    )
    assert surface.reads == 1
    assert clock() == 10
    assert result.ending is Ending.EXHAUSTED
    assert result.verification is None


def test_verification_that_ends_after_the_deadline_does_not_complete(
    edited_profile,
) -> None:
    result, surface, _, _ = timed_run(
        edited_profile, [Observe(STRUCTURED), claim(DONE.checks[0])], read_s=6
    )
    assert surface.reads == 1
    assert result.ending is Ending.EXHAUSTED
    assert result.detail == "max_wall_clock_s"


def test_a_fact_refresh_past_the_deadline_sends_nothing(edited_profile) -> None:
    field = AxLocator("textbox", "Amount")
    copy = Propose(Action(ActionKind.TYPE, field, "x"), "copy", fact="status")
    result, surface, _, _ = timed_run(
        edited_profile, [Observe(STRUCTURED), KEEP_STATUS, copy], read_s=10
    )
    assert surface.reads == 1
    assert [item.kind for item in surface.acted] == []
    assert result.ending is Ending.EXHAUSTED


@pytest.mark.parametrize(("read_s", "completed"), [(2, True), (6, False)])
def test_waiting_for_approval_is_not_verification_time(
    edited_profile, read_s, completed
) -> None:
    """Exclude the person's hundred seconds and count the read's own time."""
    result, surface, clock, escalator = timed_run(
        edited_profile,
        [Observe(STRUCTURED), claim(DONE.checks[0])],
        read_s=read_s,
        handoffs=[Handoff(HandoffOutcome.APPROVED)],
        wait_s=100,
        actions=grants({"observe": "safe", "read": "risky"}),
    )
    assert escalator.requests[0].trigger is Trigger.RISKY_ACTION
    assert clock() == 100 + read_s
    assert surface.reads == 1
    assert (result.ending is Ending.COMPLETED) is completed
    if not completed:
        assert result.ending is Ending.EXHAUSTED


def test_a_refresh_approved_for_one_page_does_not_run_on_another(
    edited_profile,
) -> None:
    """The fact-refresh read is rechecked after approval like a check's read."""
    profile = edited_profile(
        actions=grants({"observe": "safe", "read": "risky", "type": "safe"}),
        records={"actions": [], "routes": []},
    )
    other = "https://sandbox.example.test/members/24680"
    controls = (("status", "Task done"), ("textbox", "Amount"))
    surface = ScriptedSurface([Screen(MEMBER_PAGE, controls=controls)])

    class Mover(ScriptedEscalator):
        def request(self, intervention: InterventionRequest) -> Handoff:
            self.requests.append(intervention)
            if intervention.trigger is Trigger.RISKY_ACTION:
                surface.screens = [Screen(other, controls=controls)]
                return Handoff(HandoffOutcome.APPROVED)
            return Handoff(HandoffOutcome.TIMED_OUT)

    field = AxLocator("textbox", "Amount")
    copy = Propose(Action(ActionKind.TYPE, field, "x"), "copy", fact="status")
    decider = ScriptedDecider([Observe(STRUCTURED), KEEP_STATUS, copy])
    person = Mover()
    discover(
        "copy the status",
        profile,
        surface=surface,
        decider=decider,
        escalator=person,
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert surface.checked == []
    assert [item.kind for item in surface.acted] == []
    notices = [notice for seen in decider.seen for notice in seen.notices]
    assert any("approval was discarded" in notice for notice in notices)


WAIT_PAGES = {
    "/covered": """<!doctype html><html><body><h1>Posting</h1>
<button id="post" onclick="document.getElementById('posted').textContent = 'posted'"
  >Post</button>
<label>Memo <input id="memo" readonly></label>
<output id="posted">idle</output>
<div id="cover" style="position:fixed;inset:0;background:rgba(0,0,0,.2)"></div>
<script>
  // Late enough that the first look, which waits for a quiet page, is done
  // and the click is already waiting on the cover.
  setTimeout(() => window.open('/other'), 1500);
  setTimeout(() => {
    document.getElementById('cover').remove();
    document.getElementById('memo').readOnly = false;
  }, 2500);
</script></body></html>""",
    "/other": """<!doctype html><html><body><h1>Other</h1></body></html>""",
}
ONLY_PAGE_1 = ("page-1",)


def waiting(page: BrowserSurface) -> Expectation:
    return Expectation(PageState(page.location(), page="page-1"), windows=ONLY_PAGE_1)


@pytest.mark.parametrize(
    "action",
    [
        Action(ActionKind.CLICK, AxLocator("button", "Post")),
        Action(ActionKind.TYPE, DomLocator("input", DomAttribute.ID, "memo"), "x"),
    ],
    ids=["click", "type"],
)
def test_input_waiting_on_the_page_stops_when_a_window_opens(pages, action):
    """Found in review: a click waited out an overlay while page-2 opened, then ran.

    The window check ran only when the action started. The input now stops
    at the boundary after the wait, and the page shows nothing was sent.
    """
    with pages(WAIT_PAGES, "/covered", allow_new_windows=True) as (page, _):
        result = page.act(action, expect=waiting(page))
        shown = {
            "posted": page._page.locator("#posted").inner_text(),
            "memo": page._page.locator("#memo").input_value(),
        }
        listed = [item.page_id for item in page.pages()]
    assert result.outcome is Outcome.STALE
    assert "the input was not sent" in (result.detail or "")
    assert shown == {"posted": "idle", "memo": ""}
    # The permitted popup is left open for the person.
    assert listed == ["page-1", "page-2"]


def test_a_window_opening_during_a_wait_goes_to_a_person(pages) -> None:
    """In a run, the stopped click is reported as not performed, then handed over."""
    moves: list[Move] = [
        look,
        do(Action(ActionKind.CLICK, AxLocator("button", "Post"))),
    ]

    class Person(ScriptedEscalator):
        def request(self, intervention: InterventionRequest) -> Handoff:
            self.requests.append(intervention)
            return Handoff(HandoffOutcome.TIMED_OUT)

    edits = {"allow_new_windows": True, "actions": grants(DIALOGS)}
    with pages(WAIT_PAGES, "/covered", **edits) as (page, profile):
        person = Person()
        decider = Moves(moves)
        journal = MemoryJournal()
        result = discover(
            "post the entry",
            profile,
            surface=page,
            decider=decider,
            escalator=person,
            journal=journal,
            clock=FakeClock(),
        )
        posted = page._page.locator("#posted").inner_text()
    assert posted == "idle"
    assert [request.trigger for request in person.requests] == [Trigger.NEW_WINDOW]
    acted = [event for event in journal.events if isinstance(event, Acted)]
    assert [event.outcome for event in acted] == [Outcome.STALE]
    assert result.ending is Ending.HANDED_OFF


@pytest.mark.parametrize("path", ["key", "screen_click", "screen_type"])
def test_keys_and_screen_input_stop_for_an_unknown_window(pages, path) -> None:
    written = {
        "/plain": """<!doctype html><html><body><h1>Posting</h1>
<button id="post" onclick="document.getElementById('posted').textContent = 'posted'"
  >Post</button>
<label>Memo <input id="memo"></label>
<output id="posted">idle</output>
<button id="open" onclick="window.open('/other')">Open other</button>
</body></html>""",
        "/other": WAIT_PAGES["/other"],
    }
    with pages(written, "/plain", allow_new_windows=True) as (page, _):
        picture = page.observe(VISUAL)
        post = page._page.locator("#post").bounding_box()
        assert post is not None
        memo = DomLocator("input", DomAttribute.ID, "memo")
        if path == "screen_type":
            page.act(Action(ActionKind.CLICK, memo))
            picture = page.observe(VISUAL)
        both = Expectation(
            PageState(page.location(), page="page-1"), windows=("page-1", "page-2")
        )
        page.act(
            Action(ActionKind.CLICK, AxLocator("button", "Open other")), expect=both
        )
        popup(page)
        match path:
            case "key":
                sent = Action(ActionKind.PRESS_KEY, memo, "Enter")
            case "screen_click":
                point = Point(post["x"] + 5, post["y"] + 5)
                sent = Action(
                    ActionKind.CLICK, ScreenTarget(picture.observation_id, point)
                )
            case _:
                sent = Action(
                    ActionKind.TYPE, ScreenTarget(picture.observation_id), "x"
                )
        result = page.act(sent, expect=waiting(page))
        shown = (
            page._page.locator("#posted").inner_text(),
            page._page.locator("#memo").input_value(),
        )
    assert result.outcome is Outcome.STALE
    assert shown == ("idle", "")


# 30. Found in a visible run: a Stop and a Resume with nothing done in between
# saved a person's step, so every replay stopped there for a person.


def test_a_pause_where_nobody_acted_saves_no_persons_step(tmp_path):
    import time
    from pathlib import Path

    import yaml
    from pages import serve_pages

    from computeruse.capability import HumanNode
    from computeruse.control import Control, Order
    from computeruse.escalation import Command, Mode, State, Via
    from computeruse.profile import load_profile
    from computeruse.recording import DiscoveryTrace

    class Operator:
        """Stops the run at its first step, then resumes it without acting."""

        def __init__(self) -> None:
            self.control: Control | None = None
            self.sent: list[Command] = []

        def attach(self, control: Control) -> None:
            self.control = control

        def listening(self) -> bool:
            return True

        def show(self, status) -> None:
            if self.control is None:
                return
            if status.state is State.RUNNING and not self.sent:
                command = Command.STOP
            elif status.state is State.PAUSED and self.sent == [Command.STOP]:
                command = Command.RESUME
            else:
                return
            self.sent.append(command)
            intervention = status.offer.intervention if status.offer else ""
            self.control.submit(
                Order(command, status.run, status.revision, intervention, Via.PANEL)
            )

    done = (ResultCheck(CheckKind.STATE, AxLocator("heading", "Ready"), "Ready"),)
    with serve_pages({"/": "<h1>Ready</h1>"}) as base:
        document = yaml.safe_load(Path("evaluation/profile.yaml").read_text())
        document["base_url"] = base
        document["allow_routes"] = ["/**"]
        written = tmp_path / "paused.yaml"
        written.write_text(yaml.safe_dump(document))
        profile = load_profile(written)
        # A recorded session knows whether a person acted while it was paused.
        with open_session(profile, f"{base}/", record=True) as surface:
            operator = Operator()
            control = Control(
                mode=Mode.DISCOVERY,
                clock=time.monotonic,
                seat=surface,
                channels=[operator],
            )
            trace = DiscoveryTrace()
            result = discover(
                "Confirm the page is ready",
                profile,
                surface=surface,
                decider=ScriptedDecider(
                    [Observe(STRUCTURED), Observe(STRUCTURED), Finish({}, checks=done)]
                ),
                control=control,
                journal=MemoryJournal(),
                clock=time.monotonic,
                trace=trace,
            )
    assert operator.sent == [Command.STOP, Command.RESUME]
    assert result.ending is Ending.COMPLETED, result
    recording = trace.build(
        result,
        profile=profile,
        inputs={},
        capability_id="paused",
        run="scripted",
        safe_text=frozenset({"Ready"}),
    )
    assert recording.capability is not None, recording.issues
    assert not any(isinstance(node, HumanNode) for node in recording.capability.nodes)


def test_a_note_with_nothing_done_saves_no_persons_step(tmp_path):
    """The model asked for a person, who answered with a note and did nothing."""
    import time
    from pathlib import Path

    import yaml
    from pages import serve_pages

    from computeruse.capability import HumanNode
    from computeruse.control import Control, Order
    from computeruse.escalation import Command, Mode, Via
    from computeruse.profile import load_profile
    from computeruse.recording import DiscoveryTrace

    class Adviser:
        """Answers each request once with a note, touching nothing."""

        def __init__(self) -> None:
            self.control: Control | None = None
            self.answered: list[str] = []

        def attach(self, control: Control) -> None:
            self.control = control

        def listening(self) -> bool:
            return True

        def show(self, status) -> None:
            offer = status.offer
            if self.control is None or offer is None:
                return
            if offer.intervention in self.answered:
                return
            self.answered.append(offer.intervention)
            self.control.submit(
                Order(
                    Command.RESUME,
                    status.run,
                    status.revision,
                    offer.intervention,
                    Via.PANEL,
                    "it is ready",
                )
            )

    done = (ResultCheck(CheckKind.STATE, AxLocator("heading", "Ready"), "Ready"),)
    asked = AskHuman(Trigger.NO_PROGRESS, "Is the page ready?")
    with serve_pages({"/": "<h1>Ready</h1>"}) as base:
        document = yaml.safe_load(Path("evaluation/profile.yaml").read_text())
        document["base_url"] = base
        document["allow_routes"] = ["/**"]
        written = tmp_path / "advice.yaml"
        written.write_text(yaml.safe_dump(document))
        profile = load_profile(written)
        # A recorded session knows whether a person acted while they held it.
        with open_session(profile, f"{base}/", record=True) as surface:
            adviser = Adviser()
            control = Control(
                mode=Mode.DISCOVERY,
                clock=time.monotonic,
                seat=surface,
                channels=[adviser],
            )
            trace = DiscoveryTrace()
            result = discover(
                "Confirm the page is ready",
                profile,
                surface=surface,
                decider=ScriptedDecider(
                    [Observe(STRUCTURED), asked, Finish({}, checks=done)]
                ),
                control=control,
                journal=MemoryJournal(),
                clock=time.monotonic,
                trace=trace,
            )
    assert len(adviser.answered) == 1
    assert result.ending is Ending.COMPLETED, result
    recording = trace.build(
        result,
        profile=profile,
        inputs={},
        capability_id="advice",
        run="scripted",
        safe_text=frozenset({"Ready"}),
    )
    assert recording.capability is not None, recording.issues
    assert not any(isinstance(node, HumanNode) for node in recording.capability.nodes)


# 31. Found in a blind review: whether a click that submits a form asked a
# person depended on the effect the model named it, so a submission the model
# called "look_up" went through without asking under `click: {any: safe}`.

_FORM = (
    '<form action="/done" method="get">'
    '<input name="q" aria-label="Member number">'
    '<button type="submit">Look up</button></form>'
)


def _submit_innocently(pages, submissions: str):
    """Propose the form's submit click under a harmless label, then stop."""
    written = {"/": _FORM, "/done": "<h1>Done</h1>"}
    with pages(written, version=6, origins=[], submissions=submissions) as (
        surface,
        profile,
    ):
        person = ScriptedEscalator([Handoff(HandoffOutcome.REJECTED)])
        click = Action(
            ActionKind.CLICK, AxLocator("button", "Look up"), effect="look_up"
        )
        discover(
            "Look up a member",
            profile,
            surface=surface,
            decider=ScriptedDecider([Observe(STRUCTURED), Propose(click)]),
            escalator=person,
            journal=MemoryJournal(),
            clock=lambda: 0.0,
        )
        return person.requests, surface.location()


def test_a_submission_asks_a_person_whatever_the_model_called_it(pages):
    requests, location = _submit_innocently(pages, "risky")
    assert requests[0].trigger is Trigger.RISKY_ACTION
    assert location.endswith("/")


def test_a_denied_submission_is_never_sent(pages):
    requests, location = _submit_innocently(pages, "deny")
    assert all(request.trigger is not Trigger.RISKY_ACTION for request in requests)
    assert location.endswith("/")


def test_submissions_by_effect_keep_the_operators_effect_rules(pages):
    requests, location = _submit_innocently(pages, "by_effect")
    assert all(request.trigger is not Trigger.RISKY_ACTION for request in requests)
    assert "/done" in location


# 32. Found in a postback run: the model went between two pages nine times
# without the loop noticing, because every page load printed a new time and
# request number in a plain-text footer, so no state ever repeated.

_FOOTER = (
    '<div id="footer"></div><script>document.getElementById("footer")'
    '.textContent = "Rendered " + Date.now() + " · Request " + Math.random()'
    "</script>"
)


def test_a_changing_footer_does_not_hide_a_cycle(pages):
    written = {
        "/": f'<h1>Members</h1><a href="/operator">Switch operator</a>{_FOOTER}',
        "/operator": f'<h1>Operator</h1><a href="/">Members</a>{_FOOTER}',
    }
    with pages(written) as (surface, profile):
        there = Propose(Action(ActionKind.CLICK, AxLocator("link", "Switch operator")))
        back = Propose(Action(ActionKind.CLICK, AxLocator("link", "Members")))
        decider = ScriptedDecider([Observe(STRUCTURED), *[there, back] * 4])
        discover(
            "Look up a member",
            profile,
            surface=surface,
            decider=decider,
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=lambda: 0.0,
        )
    told = [notice for seen in decider.seen for notice in seen.notices]
    assert "the same action returned to an already visited state" in told


# 33. Found in a postback run: a table cell in a column with no header was
# named by position, as "td|1", which read like page text. The model rightly
# could not tell it from a record's data, so the capability was not saved.


def test_a_headerless_column_is_named_by_a_marked_position(pages):
    table = "<table><tr><td>NM000054</td><td>active</td></tr></table>"
    with pages({"/": table}) as (surface, _):
        seen = surface.observe(STRUCTURED)
    slots = {node.name: node.slot for node in seen.nodes if node.role == "cell"}
    assert slots == {"NM000054": "td#0", "active": "td#1"}


def test_a_positional_slot_is_saved_without_being_confirmed():
    from pathlib import Path

    from computeruse.capability import Limits, LocatorForm, RefKind
    from computeruse.profile import load_profile
    from computeruse.recorder import (
        DiscoveryRun,
        Place,
        Recorder,
        TargetSample,
        _Build,
    )

    recorder = Recorder(
        capability_id="collect",
        version=1,
        profile=load_profile(Path("evaluation/profile.yaml")),
        origin=DiscoveryRun("scripted"),
        inputs=(),
        outputs=(),
        variables=(),
        outcomes=(),
        limits=Limits(20, 120, 2, 3, 100, 5),
        samples={},
        collect=True,
        excluded=(),
    )
    build = _Build(recorder)

    def sample(value: str) -> TargetSample:
        return TargetSample(
            "/members",
            LocatorForm.DOM,
            tag="td",
            attribute=DomAttribute.SLOT,
            value=value,
        )

    kept = build._dom_value(sample("td#1"), Place.RECORD)
    assert kept.kind is RefKind.CONSTANT
    assert build.candidates == {}
    # A column header is page text, so it still needs confirming.
    build._dom_value(sample("td|Status"), Place.RECORD)
    assert "td|Status" in build.candidates


# 34. Found in a live write run: after an approved commit whose predicted
# postcondition failed, the loop said "inspect before retrying", and the model
# went on to open a second account. Only the step budget stopped it.

_SAVE = (
    '<form action="/done" method="get"><input name="q" aria-label="Nickname">'
    '<button type="submit">Commit</button></form>'
)


def _commit_twice(pages):
    # Each flow's form posts to its own address, as a per-flow token would.
    written = {
        "/": _SAVE.replace('action="/done"', 'action="/done?flow=1"'),
        "/done": '<h1>Committed</h1><a href="/again">Start another</a>',
        "/again": _SAVE.replace('action="/done"', 'action="/done?flow=2"'),
    }
    with pages(written, version=6, origins=[], submissions="risky") as (
        surface,
        profile,
    ):
        commit = Action(ActionKind.CLICK, AxLocator("button", "Commit"), effect="save")
        predicted = ResultCheck(
            CheckKind.STATE, AxLocator("heading", "Confirmation"), "Confirmation"
        )
        person = ScriptedEscalator([Handoff(HandoffOutcome.APPROVED)] * 3)
        decider = ScriptedDecider(
            [
                Observe(STRUCTURED),
                Propose(commit, after=(predicted,)),
                Propose(Action(ActionKind.CLICK, AxLocator("link", "Start another"))),
                Observe(STRUCTURED),
                Propose(commit),
            ]
        )
        discover(
            "Save the nickname",
            profile,
            surface=surface,
            decider=decider,
            escalator=person,
            journal=MemoryJournal(),
            clock=lambda: 0.0,
        )
    return person, [notice for seen in decider.seen for notice in seen.notices]


def test_a_sent_change_is_never_suggested_again_or_sent_twice(pages):
    person, told = _commit_twice(pages)
    approvals = [r for r in person.requests if r.trigger is Trigger.RISKY_ACTION]
    assert len(approvals) == 1
    assert any("do not send it again" in notice for notice in told)
    assert not any("inspect before retrying" in notice for notice in told)
    assert any("already sent in this run" in notice for notice in told)


@pytest.mark.rule(5, 12, 17)
def test_a_failed_picker_check_keeps_its_details_through_observations(pages):
    markup = """
        <label for="product">Product</label><input id="product">
        <button id="pick" role="option">TERM-ONE</button>
        <button id="review">Review</button>
        <p id="validation"></p>
        <script>
          window.picks = 0;
          document.getElementById('review').onclick = () => {
            document.getElementById('validation').textContent = 'Choose a plan.';
          };
          document.getElementById('pick').onclick = () => {
            window.picks += 1;
            document.getElementById('product').value = 'TERM-ONE';
          };
        </script>
    """
    product = AxLocator("textbox", "Product")
    validation = DomLocator("p", DomAttribute.ID, "validation")
    decider = ScriptedDecider(
        [
            Observe(STRUCTURED),
            Propose(
                Action(ActionKind.CLICK, AxLocator("button", "Review")),
                after=(ResultCheck(CheckKind.STATE, validation, "Choose a plan."),),
            ),
            Propose(
                Action(ActionKind.CLICK, AxLocator("option", "TERM-ONE")),
                after=(
                    ResultCheck(CheckKind.STATE, product, "Term one", Match.CONTAINS),
                ),
            ),
            Observe(STRUCTURED),
            Observe(STRUCTURED),
            Finish(
                {},
                "the selected product shows its code",
                checks=(ResultCheck(CheckKind.STATE, product, "TERM-ONE"),),
            ),
        ]
    )
    journal = MemoryJournal()
    with pages(
        {"/": markup},
        perception={
            "allowed_modes": ["structured", "visual"],
            "max_alternate_observations_per_step": 3,
        },
    ) as (surface, profile):
        result = discover(
            "Select a plan",
            profile,
            surface=surface,
            decider=decider,
            escalator=ScriptedEscalator(),
            journal=journal,
            clock=lambda: 0.0,
        )
        assert surface._page.locator("#product").input_value() == "TERM-ONE"
        assert surface._page.locator("#validation").inner_text() == "Choose a plan."
        assert surface._page.evaluate("window.picks") == 1
    assert result.ending is Ending.COMPLETED
    for transcript in decider.seen[3:]:
        assert any(
            all(
                detail in notice
                for detail in (
                    "step 3",
                    "Product",
                    "contains",
                    "'Term one'",
                    "the page shows 'TERM-ONE'",
                    "do not send it again",
                )
            )
            for notice in transcript.notices
        )
    recorded = json.dumps([as_record(event) for event in journal.events])
    assert "Term one" not in recorded
    assert "TERM-ONE" not in recorded


# 35. Found in a live write run: an account page listed its fields as rows of
# a key and value table, so every value cell had the same positional name,
# and nothing on the page could tie the account to its member.


def test_a_key_value_table_names_each_value_by_its_row_header(pages):
    table = (
        "<table><tbody>"
        "<tr><th>Account number</th><td>AC0000163</td></tr>"
        "<tr><th>Member</th><td>NM000054</td></tr>"
        "<tr><th>Nickname</th><td>Holiday fund</td></tr>"
        "</tbody></table>"
    )
    with pages({"/": table}) as (surface, _):
        seen = surface.observe(STRUCTURED)
    slots = {node.name: node.slot for node in seen.nodes if node.role == "cell"}
    assert slots == {
        "AC0000163": "td|Account number",
        "NM000054": "td|Member",
        "Holiday fund": "td|Nickname",
    }
    by_name = {node.name: node for node in seen.nodes}
    # The table now ties the nickname to the member the page shows.
    assert (
        unbound(seen, Relation.CONTAINER, by_name["Holiday fund"], by_name["NM000054"])
        is None
    )


# 36. Found in a live write run: a not-found ending for a goal that opens an
# account was refused five times, because the loop asked it to show the
# account's nickname and statement delivery, which no account had.


def test_an_outcome_proves_only_the_context_of_the_task():
    from computeruse.decider import TaskRecord, TaskRequirement
    from computeruse.loop import CheckResult, _outcome_gaps

    task = Task(
        (TaskRecord("member", "NM9"),),
        requirements=(
            TaskRequirement("operator", "OP2", context=True),
            TaskRequirement("nickname", "Holiday fund"),
        ),
    )
    shown = (
        CheckResult(
            ResultCheck(
                CheckKind.STATE, AxLocator("cell", "No records."), "No records."
            ),
            "No records.",
            True,
        ),
        CheckResult(
            ResultCheck(CheckKind.RECORD, AxLocator("textbox", "Search"), "NM9"),
            "NM9",
            True,
        ),
    )
    gaps = _outcome_gaps(task, "record_not_found", shown, None)
    # The operator must still be shown. The nickname the change would have
    # set is not asked for.
    assert len(gaps) == 1
    assert "operator" in gaps[0]
    signed = CheckResult(
        ResultCheck(
            CheckKind.REQUIREMENT,
            AxLocator("generic", "Signed on as OP2"),
            "OP2",
            Match.CONTAINS,
            requirement="operator",
        ),
        "Signed on as OP2",
        True,
    )
    assert _outcome_gaps(task, "record_not_found", (*shown, signed), None) == []


# 37. Found in a live write run: a result page linked its receipt by the
# receipt's own number, so the link could be saved only by a number that is
# different for every record.


def test_a_link_alone_in_a_value_cell_takes_the_cells_slot(pages):
    table = (
        "<table><tbody>"
        "<tr><th>Receipt</th><td><a class='mono' href='#r'>R-17</a></td></tr>"
        "<tr><th>Reference</th><td>X-9 <a href='#x'>open</a></td></tr>"
        "</tbody></table>"
    )
    with pages({"/": table}) as (surface, _):
        seen = surface.observe(STRUCTURED)
    slots = {node.name: node.slot for node in seen.nodes if node.role == "link"}
    # The receipt's link is all its cell holds. The other shares its cell.
    assert slots["R-17"] == "a|Receipt"
    assert slots["open"] != "a|Reference"


# 38. Found in a live write run: a list filtered to one member showed that
# member in every row, so evidence naming the member's cell named every row,
# and the new account's number was refused six times.

_ONE_MEMBERS_ACCOUNTS = (
    "<h1>Accounts</h1><table><thead><tr><th>Number</th><th>Member</th></tr>"
    "</thead><tbody>"
    "<tr><td><a href='#a1'>A-1</a></td><td>M-1</td></tr>"
    "<tr><td><a href='#a2'>A-2</a></td><td>M-1</td></tr>"
    "</tbody></table>"
)


def test_row_evidence_is_read_in_the_remembered_controls_own_row(pages):
    tied = RecordEvidence(AxLocator("cell", "M-1"), "M-1", Relation.ROW)
    with pages({"/": _ONE_MEMBERS_ACCOUNTS}) as (surface, profile):
        decider = ScriptedDecider(
            [
                Observe(STRUCTURED),
                Remember(
                    "account",
                    "A-2",
                    may_change=False,
                    source=AxLocator("link", "A-2"),
                    record=tied,
                ),
                Observe(STRUCTURED),
            ]
        )
        discover(
            "Report member M-1's new account number.",
            profile,
            surface=surface,
            decider=decider,
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=FakeClock(),
        )
    kept = {fact.key: fact for fact in decider.seen[-1].memory}
    assert kept["account"].value == "A-2"
    assert kept["account"].record is not None


# 39. Found in a live write run: a receipt's event rows were named by their
# sequence numbers, the first cell no other row repeats, so a read in the
# row that showed the member was saved by a number no other receipt has.

_EVENTS = (
    "<h1>Receipt</h1><table><thead><tr><th>Seq</th><th>Summary</th></tr></thead>"
    "<tbody>"
    "<tr><td>600471</td><td>Account A-2 (savings) opened.</td></tr>"
    "<tr><td>600472</td><td>Account A-2 opened for M-1.</td></tr>"
    "</tbody></table>"
)


def test_a_row_is_found_by_any_cell_no_other_row_repeats(pages):
    with pages({"/": _EVENTS}) as (surface, _):
        seen = surface.observe(STRUCTURED)
        summary = next(
            node for node in seen.nodes if node.name == "Account A-2 opened for M-1."
        )
        assert summary.scope == Scope(ScopeKind.ROW, "600472")
        assert summary.row_names == ("600472", "Account A-2 opened for M-1.")
        # The browser finds the row by its summary as well as by its number.
        by_summary = DomLocator(
            "td",
            DomAttribute.SLOT,
            "td|Summary",
            scope=Scope(ScopeKind.ROW, "Account A-2 opened for M-1."),
        )
        read = surface.act(Action(ActionKind.READ, by_summary))
    assert read.outcome is Outcome.OK
    assert read.extracted == "Account A-2 opened for M-1."


def test_a_row_named_by_data_is_saved_by_the_cell_holding_the_input():
    from computeruse.recorder import TargetSample
    from computeruse.recording import target_sample

    row = Scope(ScopeKind.ROW, "600472")
    summary = AxNode(
        "cell",
        "Account A-2 opened for M-1.",
        tag="td",
        scope=row,
        row="r2",
        slot="td|Summary",
        row_names=("600472", "Account A-2 opened for M-1."),
    )
    other = AxNode(
        "cell",
        "Account A-2 (savings) opened.",
        tag="td",
        scope=Scope(ScopeKind.ROW, "600471"),
        row="r1",
        slot="td|Summary",
        row_names=("600471", "Account A-2 (savings) opened."),
    )
    seen = Observation(
        "events",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        PageState("http://127.0.0.1:8787/receipt"),
        nodes=(other, summary),
    )
    sample = target_sample(
        DomLocator("td", DomAttribute.SLOT, "td|Summary", scope=row),
        "/receipt",
        seen,
        inputs=("M-1",),
        safe_text=frozenset({"td|Summary"}),
    )
    assert isinstance(sample, TargetSample)
    assert sample.scope == Scope(ScopeKind.ROW, "Account A-2 opened for M-1.")


def test_replay_finds_a_row_by_the_name_holding_its_input():
    from computeruse.capability import (
        LocatorForm,
        ScopeSpec,
        StructuralTarget,
        constant,
        ref,
    )
    from computeruse.capability import Match as SavedMatch
    from computeruse.capability import RefKind as SavedRef
    from computeruse.retarget import Found, resolve

    def event(key: str, seq: str, summary: str) -> AxNode:
        return AxNode(
            "cell",
            summary,
            tag="td",
            scope=Scope(ScopeKind.ROW, seq),
            row=key,
            slot="td|Summary",
            row_names=(seq, summary),
        )

    seen = Observation(
        "events",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        PageState("http://127.0.0.1:8787/receipt"),
        nodes=(
            event("r1", "600473", "Account A-3 (savings) opened."),
            event("r2", "600474", "Account A-3 opened for M-9."),
        ),
    )
    saved = StructuralTarget(
        "summary",
        "/receipt",
        LocatorForm.DOM,
        "",
        None,
        "td",
        DomAttribute.SLOT,
        constant("td|Summary"),
        (),
        ScopeSpec(ScopeKind.ROW, ref(SavedRef.INPUT, "member_id")),
        SavedMatch.CONTAINS,
    )
    inputs = {"member_id": "M-9"}
    found = resolve(
        saved, seen, lambda value: inputs.get(value.name, value.value), "/receipt"
    )
    assert found.found is Found.FOUND
    assert found.node is not None
    assert found.node.row == "r2"
    # The surface is handed the row's own first name, read from this page.
    assert isinstance(found.locator, DomLocator)
    assert found.locator.scope == Scope(ScopeKind.ROW, "600474")


# 40. Found in a live write run: a form drew each field's label beside the
# field without linking them, and put a picker's input inside a custom
# element's shadow root, so four fields had no name and no slot to find.

_UNLINKED = """<h1>Open account</h1>
<div class="field"><label>Nickname</label><div><input type="text"></div></div>
<div class="field"><label>Statement delivery</label>
  <div><select><option>paper</option><option>electronic</option></select></div></div>
<div class="field"><label>Member</label><div><x-picker></x-picker></div></div>
<script>
customElements.define('x-picker', class extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({mode: 'open'}).innerHTML =
      '<input role="combobox" aria-expanded="false">';
  }
});
</script>"""


def test_a_field_is_placed_by_the_one_label_beside_it(pages):
    with pages({"/": _UNLINKED}) as (surface, _):
        seen = surface.observe(STRUCTURED)
        slots = sorted(
            node.slot for node in seen.nodes if node.tag in {"input", "select"}
        )
        # The names stay empty. Only the position of each field is known.
        assert all(
            not node.name for node in seen.nodes if node.tag in {"input", "select"}
        )
        typed = surface.act(
            Action(
                ActionKind.TYPE,
                DomLocator("input", DomAttribute.SLOT, "input|Nickname"),
                "Rainy day",
            )
        )
    assert slots == ["input|Member", "input|Nickname", "select|Statement delivery"]
    assert typed.outcome is Outcome.OK


# 41. Found in a live canvas run: every keystroke goes to the canvas, so a
# read of its painted header was refused as a field the run had changed.

_PAINTED = """<canvas id="screen" width="600" height="200" tabindex="0"></canvas>
<script>
const paint = document.getElementById('screen').getContext('2d');
paint.fillStyle = '#ffffff'; paint.fillRect(0, 0, 600, 200);
paint.fillStyle = '#111111'; paint.font = '28px Helvetica, Arial, sans-serif';
paint.fillText('Status: active', 40, 80);
</script>"""


def test_a_painted_line_is_read_where_the_run_points(pages):
    from computeruse.actions import Expectation

    with pages({"/": _PAINTED}) as (surface, _):
        seen = surface.observe(VISUAL)
        surface.act(
            Action(ActionKind.TYPE, ScreenTarget(seen.observation_id), "x"),
        )
        seen = surface.observe(VISUAL)
        read = surface.act(
            Action(ActionKind.READ, ScreenTarget(seen.observation_id, Point(120, 70))),
            expect=Expectation(seen.page_state, committed=True),
        )
    assert read.outcome is Outcome.OK
    assert read.extracted is not None
    assert "".join(read.extracted.split()).casefold() == "status:active"


# 42. Found in a live write run: after the commit the model flagged it as
# risky, read "now needs a person's approval" as the application asking, and
# flagged it again 38 times until the run's steps ran out.


@pytest.mark.rule(17)
def test_a_finding_is_kept_once_and_the_task_goes_on(drive) -> None:
    def flag(seen):
        return FlagRisk(
            RiskFinding(the_step_that_pressed(seen), "submit_payment", "it posted")
        )

    moves: list[Move] = [
        look,
        do(TYPE_AMOUNT),
        do(enter(AMOUNT, "save_form")),
        flag,
        flag,
        done,
    ]

    with drive(
        lambda _: moves,
        path="/payments",
        escalator=rejecting,
        actions=paying(),
        records=NO_RECORDS,
    ) as ran:
        pass

    findings = [r for r in ran.result.restrictions if r.source is Source.FINDING]
    assert len(findings) == 1
    notices = ran.notices()
    assert any("nobody is asked now" in notice for notice in notices)
    assert any("already flagged" in notice for notice in notices)


# 43. Found in a live write run: the model kept the signed-on operator's code
# from a banner that says more around it, and the finish compared that fact
# with the operator input. The draft saved the whole banner's read into a
# variable, so a replay compared the whole banner with the code and failed.


@pytest.mark.rule(11)
@pytest.mark.rule(3)
def test_a_fact_inside_a_longer_text_is_checked_where_it_was_read() -> None:
    from pathlib import Path

    from computeruse.capability import (
        ActionNode,
        RefKind,
        ResultNode,
        Shows,
        dumps,
    )
    from computeruse.decider import Fact
    from computeruse.loop import CheckResult
    from computeruse.profile import load_profile
    from computeruse.recording import DiscoveryTrace

    profile = load_profile(Path("evaluation/profile.yaml"))
    state = PageState("http://127.0.0.1:8787/members")
    banner = DomLocator("div", DomAttribute.ID, "operator")
    text = "Signed on as Sam Lee (U-7) · Sign out"
    seen = Observation(
        "seen",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        state,
        nodes=(
            AxNode("text", text, tag="div", attributes=(("id", "operator"),)),
            AxNode("heading", "Posted", tag="h1"),
        ),
    )
    trace = DiscoveryTrace()
    trace.look(seen)
    trace.action(
        1,
        Action(ActionKind.READ, banner),
        ActionResult(Outcome.OK, state, text),
        state.location,
        seen,
    )
    # The loop marks a fact kept from inside a longer text as partial.
    trace.kept(
        Fact("operator", "U-7", 1, "/members", banner, may_change=False, partial=True)
    )
    result = RunResult(
        Ending.COMPLETED,
        2,
        "done",
        verification=Verification.EXECUTOR,
        checks=(
            CheckResult(
                ResultCheck(
                    CheckKind.REQUIREMENT,
                    FactRef("operator"),
                    "U-7",
                    requirement="operator",
                ),
                "U-7",
                True,
            ),
            CheckResult(
                ResultCheck(CheckKind.STATE, AxLocator("heading", "Posted"), "Posted"),
                "Posted",
                True,
            ),
        ),
    )
    recording = trace.build(
        result,
        profile=profile,
        inputs={"operator_id": "U-7"},
        capability_id="signed_on",
        run="scripted",
        safe_text=frozenset({"text", "operator", "heading", "Posted"}),
    )
    assert recording.complete, (recording.issues, recording.artifact_issues)
    capability = recording.capability
    assert capability is not None
    assert capability.variables == ()
    read = next(node for node in capability.nodes if isinstance(node, ActionNode))
    assert read.into is None
    check = read.verify[0]
    assert isinstance(check, Shows)
    assert (check.value.kind, check.value.name) == (RefKind.INPUT, "operator_id")
    assert check.match.value == "contains"
    # Discovery kept the value only where the banner held it exactly once, so
    # replay must too (finding 6).
    assert check.once
    finish = next(node for node in capability.nodes if isinstance(node, ResultNode))
    assert all(isinstance(item, Shows) for item in finish.checks)
    assert "Sam Lee" not in dumps(capability)


# 44. Found in review (finding 12): a fact kept from inside a longer text was
# replaced by that whole text when the finish read it again, so its check
# compared the whole banner with the operator code and failed.


@pytest.mark.rule(11)
def test_a_partial_fact_keeps_its_value_when_read_again(pages) -> None:
    import time

    from fakes import ScriptedDecider, ScriptedEscalator

    from computeruse.actions import ObservationMode, ObservationRequest
    from computeruse.decider import (
        CheckKind,
        FactRef,
        Finish,
        Observe,
        Remember,
        ResultCheck,
    )
    from computeruse.journal import MemoryJournal
    from computeruse.loop import Ending, discover

    markup = '<div id="op">Signed on as Sam Lee (U-7) · Sign out</div><h1>Done</h1>'
    banner = DomLocator("div", DomAttribute.ID, "op")
    with pages({"/": markup}) as (surface, profile):
        result = discover(
            "Confirm operator U-7 is signed on",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [
                    Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                    Remember("operator", "U-7", source=banner, may_change=False),
                    Finish(
                        {},
                        checks=(
                            ResultCheck(CheckKind.STATE, FactRef("operator"), "U-7"),
                        ),
                    ),
                ]
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
    assert result.ending is Ending.COMPLETED, result.detail


# 45. Found in review (finding 13): a fact longer than the limit was cut to
# the limit, so it became a different value that looked like a part of its
# control's text.


@pytest.mark.rule(11)
def test_a_value_longer_than_a_fact_may_be_is_refused_not_cut(pages) -> None:
    import time

    from fakes import ScriptedDecider, ScriptedEscalator

    from computeruse.actions import ObservationMode, ObservationRequest
    from computeruse.decider import MAX_FACT_VALUE, Observe, Remember
    from computeruse.journal import MemoryJournal
    from computeruse.loop import discover

    long = "word " * (MAX_FACT_VALUE // 4)
    markup = f'<p id="note">{long}</p>'
    note = DomLocator("p", DomAttribute.ID, "note")
    seen: list[str] = []

    class Watching(ScriptedDecider):
        def decide(self, transcript):
            seen.extend(transcript.notices)
            return super().decide(transcript)

    with pages({"/": markup}) as (surface, profile):
        discover(
            "Keep the note",
            profile,
            surface=surface,
            decider=Watching(
                [
                    Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                    Remember("note", long.strip(), source=note, may_change=False),
                    Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                ]
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
    assert any("longer than" in notice for notice in seen)


# 46. Found in review (R2): a page turned a plain button into a submit button
# after the gate judged it, and with submissions denied the form still sent.


@pytest.mark.rule(6)
def test_a_submission_changed_after_the_gate_is_not_sent(pages) -> None:
    # Graduated from tests/test_known_gaps.py (R2).
    import time

    from fakes import ScriptedDecider, ScriptedEscalator

    from computeruse.actions import ObservationMode, ObservationRequest
    from computeruse.decider import Observe, Propose
    from computeruse.journal import MemoryJournal
    from computeruse.loop import discover
    from computeruse.profile import Limit, Submissions

    markup = """<form onsubmit="event.preventDefault(); window.commits += 1">
    <button id="send" type="button">Continue</button></form>
    <script>window.commits = 0</script>"""
    with pages({"/": markup}) as (surface, loaded):
        blocked = dataclasses.replace(loaded, submissions=Submissions(Limit.DENY))
        surface._profile = blocked

        class ChangesBeforeActing(ScriptedDecider):
            def decide(self, transcript):
                decision = super().decide(transcript)
                if isinstance(decision, Propose):
                    surface._page.locator("#send").evaluate("el => el.type = 'submit'")
                return decision

        discover(
            "Continue",
            blocked,
            surface=surface,
            decider=ChangesBeforeActing(
                [
                    Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                    Propose(Action(ActionKind.CLICK, AxLocator("button", "Continue"))),
                ]
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
        assert surface._page.evaluate("window.commits") == 0


# 47. Found in review (rule 10): discovery read a painted line from one
# capture. A line whose far end changed after the capture, outside the
# pixels checked at the point, was read as it had been.


@pytest.mark.rule(10)
def test_a_painted_line_that_changed_since_the_capture_is_not_read(pages):
    from computeruse.actions import Expectation

    with pages({"/": _PAINTED}) as (surface, _):
        seen = surface.observe(VISUAL)
        surface._page.evaluate(
            """() => {
              const paint = document.getElementById('screen').getContext('2d');
              paint.fillStyle = '#ffffff'; paint.fillRect(150, 40, 300, 60);
              paint.fillStyle = '#111111';
              paint.font = '28px Helvetica, Arial, sans-serif';
              paint.fillText('closed', 150, 80);
            }"""
        )
        read = surface.act(
            Action(ActionKind.READ, ScreenTarget(seen.observation_id, Point(60, 70))),
            expect=Expectation(seen.page_state, committed=True),
        )
    assert read.outcome is Outcome.STALE
    assert read.extracted is None


# 48. Found in review (findings 48 and 49): refusals, declined approvals, and
# repeated findings never counted as lack of progress, so a run could repeat
# them until its step budget ran out.


@pytest.mark.rule(17)
def test_repeated_findings_hand_the_run_to_a_person(drive) -> None:
    def flag(seen):
        return FlagRisk(
            RiskFinding(the_step_that_pressed(seen), "submit_payment", "it posted")
        )

    moves: list[Move] = [
        look,
        do(TYPE_AMOUNT),
        do(enter(AMOUNT, "save_form")),
        *([flag] * 12),
        done,
    ]

    with drive(
        lambda _: moves,
        path="/payments",
        escalator=rejecting,
        actions=paying(),
        records=NO_RECORDS,
    ) as ran:
        pass

    assert ran.result.ending is not Ending.COMPLETED
    assert "repeated refusals" in ran.result.detail or any(
        request.trigger is Trigger.NO_PROGRESS for request in ran.requests()
    )


# 49. Found in batch 43: the model kept a new account's number from the
# sentence that announced it, then opened the account by it. The saved
# capability could not read the number back from the sentence. Composite
# reads now preserve known surrounding text, and the notice explains when
# a source that shows only the value is still needed.


@pytest.mark.rule(11)
def test_a_fact_kept_inside_a_sentence_explains_its_recording_requirements(
    pages,
) -> None:
    import time

    from fakes import ScriptedDecider, ScriptedEscalator

    from computeruse.actions import ObservationMode, ObservationRequest
    from computeruse.decider import Observe, Remember
    from computeruse.journal import MemoryJournal
    from computeruse.loop import discover

    markup = '<p id="done">Order O-2001 placed for C-1001.</p><a href="#">O-2001</a>'
    sentence = DomLocator("p", DomAttribute.ID, "done")
    seen: list[str] = []

    class Watching(ScriptedDecider):
        def decide(self, transcript):
            seen.extend(transcript.notices)
            return super().decide(transcript)

    with pages({"/": markup}) as (surface, profile):
        discover(
            "Keep the new order's number",
            profile,
            surface=surface,
            decider=Watching(
                [
                    Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                    Remember("order", "O-2001", source=sentence, may_change=False),
                    Observe(ObservationRequest(ObservationMode.STRUCTURED)),
                ]
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
    assert any("unique surrounding text" in notice for notice in seen)
    assert any(
        "Otherwise keep it from a control that shows it alone" in notice
        for notice in seen
    )


# 50. Found in canvas batches 48 and 50 (rule 15): a painted "not found"
# answer could not prove the searched record. The loop required a painted
# line to equal the identifier, which only the search field does, and the
# recorder could not save a line that ends with the identifier and a full
# stop. A person had to confirm every canvas "not found" run, and no branch
# was learned. The claim and branch tests are in tests/test_claims.py,
# tests/test_recording_bridge.py, and tests/test_replay.py.


@pytest.mark.rule(15)
def test_a_painted_answer_ending_in_its_record_keeps_the_words_before_it() -> None:
    from computeruse.recording import _label_before, _whole_line

    answer = "No order with number O-2001."
    assert _label_before(answer, "O-2001") == "No order with number"
    assert _whole_line(answer, "No order with number O-2001.")
    assert not _whole_line(answer, "No order")


# 51. Found in canvas batches 39 to 51 (rule 15): a painted screen could
# never tie a result to its record, so every canvas lookup, and every canvas
# replay, asked a person to confirm that the status belonged to the member.
# A label line that names the member once now ties the readings of its
# picture. A picture with two such lines is a list and ties nothing.


@pytest.mark.rule(15)
def test_a_painted_member_label_ties_only_when_shown_once() -> None:
    from computeruse.actions import Point, ScreenTarget
    from computeruse.decider import CheckKind, Match, ResultCheck, Task, TaskRecord
    from computeruse.loop import CheckResult, _painted_ties

    task = Task((TaskRecord("member", "U-7"),))
    member = ResultCheck(
        CheckKind.RECORD, ScreenTarget("c1", Point(1, 1)), "U-7", Match.CONTAINS
    )
    once = ("Member number: U-7", "Status: active")
    twice = ("Member number: U-7", "Member number: U-8")
    sentence = ("Opened for member U-7", "Status: active")
    for screen, bound in ((once, "U-7"), (twice, ""), (sentence, "")):
        read = CheckResult(member, screen[0], True, screen=screen)
        assert _painted_ties(task, (read,))[0].bound == bound


# 52. Found in canvas batch 52 (rules 15 and 16): the model read the goal's
# output ``account_number`` as ``account number``. The loop refused the
# claim under the declared name, and a person approved one under the other
# name without seeing it, because a result approval showed no outputs. The
# capability could not be saved. The reading must now name the contract's
# outputs, and every channel shows the result a person is asked to confirm.


@pytest.mark.rule(16)
def test_a_result_approval_shows_what_the_person_confirms() -> None:
    from computeruse.control import confirming
    from computeruse.escalation import InterventionRequest, Trigger

    request = InterventionRequest(
        Trigger.UNVERIFIED_RESULT,
        "",
        "p",
        4,
        "/",
        "",
        60.0,
        outputs={"order_total": "12.00"},
        unverified=("required state currency is not established",),
    )
    shown = confirming(request)
    assert "order_total = 12.00" in shown
    assert "currency is not established" in shown


@pytest.mark.rule(15)
def test_a_task_reading_must_use_the_declared_output_names() -> None:
    from computeruse.decider import Task, TaskOutput
    from computeruse.loop import _undeclared

    renamed = Task(outputs=(TaskOutput("order total"),))
    assert _undeclared(renamed, ("order_total",)) is not None
    assert _undeclared(renamed, ()) is None


# 53. Found in the first headed command-line discovery after the cleanup:
# saving opened the outcome case's browser while the discovery's browser was
# still open, and Playwright refused a second sync driver on the same thread.
# A session opened inside another now launches from the running driver.


def test_a_session_opened_inside_another_uses_the_running_driver(
    site: str, site_profile
) -> None:
    profile: Profile = site_profile()
    with open_session(profile, f"{site}/members/12345") as outer:
        with open_session(profile, f"{site}/members/12345") as inner:
            assert inner.browser is not outer.browser
            assert inner.browser.is_connected()
        assert not inner.browser.is_connected()
        assert outer.browser.is_connected()
    # A later session on the same thread starts a driver of its own again.
    with open_session(profile, f"{site}/members/12345") as later:
        assert later.browser.is_connected()
