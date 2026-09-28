"""A small stateful application and scripted people, for replay and recorder tests.

The application defines named states, the controls on each page, and the
action that moves the application to another state. Replay drives it through
the same ``Surface`` protocol as the browser adapter. Every replay action is
therefore visible here.

Everything in this file is synthetic. The capability built by
``transfer_capability`` is a hand-written test fixture, not a discovered one.
"""

from __future__ import annotations

import dataclasses
import io
from collections.abc import Callable

from PIL import Image, ImageDraw

from computeruse.actions import (
    DIALOG_ACTIONS,
    Action,
    ActionResult,
    AxLocator,
    AxNode,
    Capabilities,
    DomAttribute,
    DomLocator,
    Expectation,
    Observation,
    ObservationRequest,
    ObservationStatus,
    Outcome,
    PageInfo,
    PageState,
    PendingDialog,
    Relation,
    ScreenTarget,
    TargetForm,
    Window,
    names_node,
)
from computeruse.budget import Clock as ClockProtocol
from computeruse.capability import (
    SCHEMA_VERSION,
    ActionNode,
    Application,
    Approval,
    AtRoute,
    Bound,
    Capability,
    Edge,
    EdgeOrigin,
    Field,
    Limits,
    LocatorForm,
    Present,
    Provenance,
    ProvenanceKind,
    RecordSpec,
    RefKind,
    ResultKind,
    ResultNode,
    Review,
    Shows,
    StructuralTarget,
    SurfaceKind,
    ValueType,
    constant,
    ref,
)
from computeruse.capability import Match as TextMatch
from computeruse.control import Control
from computeruse.escalation import Handoff, HandoffOutcome, InterventionRequest, Mode
from computeruse.profile import ActionKind
from computeruse.retarget import Scene
from computeruse.surface import SurfaceError

ORIGIN = "https://bank.test"

PROFILE = """\
version: 4
profile_id: bank/core/sandbox
environment: sandbox
base_url: https://bank.test
allow_routes:
  - /members
  - /members/:id
  - /members/:id/transfer
  - /desk
deny_routes: []
allow_new_windows: false
allow_downloads: false
actions:
  observe: {any: safe}
  read: {any: safe}
  navigate: {any: safe}
  click:
    any: safe
    effects:
      submit_transfer: risky
  type: {any: safe}
  select: {any: safe}
  press_key: {any: safe}
  accept_dialog: {any: safe}
  dismiss_dialog: {any: safe}
perception:
  allowed_modes: [structured, visual]
  max_alternate_observations_per_step: 1
records:
  actions: [click]
  routes: [/members/:id/transfer]
budgets:
  max_steps: 40
  max_wall_clock_s: 300
  max_retries_per_step: 3
  max_navigations: 10
secrets:
  teller_pin: {env: TELLER_PIN}
escalation:
  handoff_timeout_s: 600
  on_timeout: abort
"""


def dd(slot: str, text: str, element: str) -> AxNode:
    """Return a definition cell showing ``text``, found by its element id."""
    return AxNode(
        "definition", text, tag="dd", attributes=(("id", element),), slot=slot
    )


def button(name: str) -> AxNode:
    return AxNode("button", name, tag="button")


def field(name: str, value: str = "", *, secret: bool = False) -> AxNode:
    return AxNode("textbox", name, value=value, tag="input", secret=secret)


def status(text: str) -> AxNode:
    return AxNode("status", text, tag="p")


@dataclasses.dataclass
class Page:
    path: str
    nodes: tuple[AxNode, ...]


def bank_pages(
    member: str = "10001", shown: str | None = None, swapped: str | None = None
) -> dict[str, Page]:
    """The transfer workflow's states.

    ``shown`` is the member every page displays. ``swapped`` is a different
    member the transfer screen displays, as if the record changed underneath
    the same kind of page.
    """
    shown = shown or member
    other = swapped or shown
    return {
        "search": Page("/members", (field("Member number"), button("Search"))),
        "not_found": Page(
            "/members",
            (field("Member number"), button("Search"), status("No member found")),
        ),
        "member": Page(
            f"/members/{shown}",
            (
                dd("Member number", shown, "member-number"),
                dd("Balance", "1204.50", "balance"),
                AxNode("link", "Transfer", tag="a"),
            ),
        ),
        "transfer": Page(
            f"/members/{other}/transfer",
            (
                dd("Member number", other, "member-number"),
                field("Amount"),
                field("Teller PIN", secret=True),
                button("Submit transfer"),
            ),
        ),
        "confirmed": Page(
            f"/members/{shown}/transfer",
            (dd("Member number", shown, "member-number"), status("Transfer submitted")),
        ),
    }


BANK_RULES = {
    ("search", ActionKind.CLICK, "Search"): "member",
    ("member", ActionKind.CLICK, "Transfer"): "transfer",
    ("transfer", ActionKind.CLICK, "Submit transfer"): "confirmed",
}


@dataclasses.dataclass
class App:
    """A scripted application behind the ``Surface`` protocol.

    ``outcomes`` overrides what the next actions report, in order. Each entry
    also says whether the action took effect, so a timeout that did submit
    and one that did not are both expressible.

    ``coverage`` is the status every observation reports, and ``omit`` names
    controls an observation leaves out while the application still has them,
    as a truncated tree would. ``opens`` maps an action to the native dialog
    it opens, and an answer to the dialog moves the state by the rule keyed
    on the name ``dialog``. ``on_observe`` runs before each observation.
    ``page_size`` shows at most that many controls per look, as a bounded
    adapter pages a long screen.
    """

    screens: dict[str, Page]
    state: str
    rules: dict[tuple[str, ActionKind, str], str]
    outcomes: list[tuple[Outcome, bool]] = dataclasses.field(default_factory=list)
    acted: list[Action] = dataclasses.field(default_factory=list)
    typed: dict[str, str] = dataclasses.field(default_factory=dict)
    fail_with: SurfaceError | None = None
    looks: int = 0
    expected: list[Expectation | None] = dataclasses.field(default_factory=list)
    on_act: Callable[[App, Action], None] | None = None
    on_observe: Callable[[App], None] | None = None
    coverage: ObservationStatus = ObservationStatus.COMPLETE
    omit: set[str] = dataclasses.field(default_factory=set)
    page_size: int | None = None
    opens: dict[tuple[str, ActionKind, str], str] = dataclasses.field(
        default_factory=dict
    )
    dialog: PendingDialog | None = None
    dialogs: int = 0
    answered: list[tuple[ActionKind, str]] = dataclasses.field(default_factory=list)
    observed_at: list[str] = dataclasses.field(default_factory=list)

    def pages(self) -> tuple[PageInfo, ...]:
        return ()

    def capabilities(self) -> Capabilities:
        return {kind: frozenset(TargetForm) for kind in ActionKind}

    def location(self) -> str:
        return ORIGIN + self.screens[self.state].path

    def observe(self, request: ObservationRequest) -> Observation:
        if self.on_observe is not None:
            self.on_observe(self)
        self.looks += 1
        self.observed_at.append(self.screens[self.state].path)
        if self.dialog is not None:
            return Observation(
                f"obs-{self.looks}",
                request.mode,
                ObservationStatus.UNAVAILABLE,
                PageState(self.location()),
                dialog=self.dialog,
            )
        nodes = tuple(
            dataclasses.replace(node, value=self.typed.get(node.name, node.value))
            if node.role in {"textbox", "combobox"} and not node.secret
            else node
            for node in self.screens[self.state].nodes
            if node.name not in self.omit
        )
        if self.page_size is not None:
            # A bounded adapter shows one part of the screen per look.
            total = len(nodes)
            part = nodes[request.start : request.start + self.page_size]
            window = Window(request.start, len(part), total)
            return Observation(
                f"obs-{self.looks}",
                request.mode,
                ObservationStatus.PARTIAL if window.rest else self.coverage,
                PageState(self.location()),
                nodes=part,
                window=window,
            )
        return Observation(
            f"obs-{self.looks}",
            request.mode,
            self.coverage,
            PageState(self.location(), tuple(node.identity for node in nodes)),
            nodes=nodes,
        )

    def open_dialog(self, kind: str = "confirm") -> PendingDialog:
        self.dialogs += 1
        self.dialog = PendingDialog(f"dialog-{self.dialogs}", kind, "Post this entry?")
        return self.dialog

    def act(self, action: Action, *, expect: Expectation | None = None) -> ActionResult:
        self.acted.append(action)
        self.expected.append(expect)
        if self.on_act is not None:
            self.on_act(self, action)
        if self.fail_with is not None:
            raise self.fail_with
        if action.kind in DIALOG_ACTIONS or self.dialog is not None:
            return self._answer(action, expect)
        outcome, lands = self.outcomes.pop(0) if self.outcomes else (Outcome.OK, True)
        found = self._find(action)
        if found is None and action.target is not None:
            return ActionResult(Outcome.NOT_FOUND, PageState(self.location()))
        if action.evidence is not None and not self._evidence_holds(action):
            return ActionResult(Outcome.STALE, PageState(self.location()))
        extracted = None
        if action.kind is ActionKind.READ and found is not None:
            extracted = found.value if found.value is not None else found.name
        if lands:
            self._apply(action, found)
        return ActionResult(outcome, PageState(self.location()), extracted)

    def _answer(self, action: Action, expect: Expectation | None) -> ActionResult:
        """Answer the waiting dialog only if the action names it, as a browser does."""
        here = PageState(self.location())
        if self.dialog is None or action.kind not in DIALOG_ACTIONS:
            return ActionResult(Outcome.NOT_ACTIONABLE, here)
        if expect is None or expect.dialog != self.dialog.dialog_id:
            return ActionResult(Outcome.STALE, here)
        self.answered.append((action.kind, self.dialog.dialog_id))
        self.dialog = None
        following = self.rules.get((self.state, action.kind, "dialog"))
        if following is not None:
            self.state = following
        return ActionResult(Outcome.OK, PageState(self.location()))

    def acted_on(self, name: str) -> int:
        return sum(1 for action in self.acted if _named(action) == name)

    def _find(self, action: Action) -> AxNode | None:
        if isinstance(action.target, ScreenTarget):
            return AxNode("canvas", "screen")
        if action.target is None:
            return None
        matches = [
            node
            for node in self.screens[self.state].nodes
            if node.visible and names_node(action.target, node)
        ]
        return matches[0] if len(matches) == 1 else None

    def _evidence_holds(self, action: Action) -> bool:
        evidence = action.evidence
        assert evidence is not None
        return any(
            names_node(evidence.source, node) and node.name == evidence.displayed
            for node in self.screens[self.state].nodes
        )

    def _apply(self, action: Action, found: AxNode | None) -> None:
        if action.kind in {ActionKind.TYPE, ActionKind.SELECT} and found is not None:
            if isinstance(action.value, str):
                self.typed[found.name] = action.value
            return
        name = _named(action)
        if (self.state, action.kind, name) in self.opens:
            self.open_dialog(self.opens[self.state, action.kind, name])
        following = self.rules.get((self.state, action.kind, name))
        if following is not None:
            self.state = following


def _named(action: Action) -> str:
    target = action.target
    if isinstance(target, AxLocator):
        return target.name
    if isinstance(target, DomLocator):
        return target.value
    if isinstance(target, ScreenTarget):
        return "screen"
    return action.destination or ""


def bank(
    member: str = "10001", *, shown: str | None = None, swapped=None, rules=None
) -> App:
    return App(bank_pages(member, shown, swapped), "search", dict(rules or BANK_RULES))


type Step = HandoffOutcome | Callable[[InterventionRequest], HandoffOutcome]


@dataclasses.dataclass
class People:
    """A synchronous operator channel that answers real control interventions."""

    script: list[Step]
    requests: list[InterventionRequest] = dataclasses.field(default_factory=list)
    controls: list[Control] = dataclasses.field(default_factory=list)

    def control(self, clock: ClockProtocol) -> Control:
        """Create a production control for an invocation through this channel."""
        control = Control(mode=Mode.REPLAY, clock=clock, escalator=self)
        self.controls.append(control)
        return control

    def request(self, intervention: InterventionRequest) -> Handoff:
        self.requests.append(intervention)
        step = self.script.pop(0) if self.script else HandoffOutcome.TIMED_OUT
        outcome = step(intervention) if callable(step) else step
        return Handoff(outcome, intervention=intervention.intervention)


class Clock:
    """A monotonic clock that moves only when told to."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def limits(**changes: int) -> Limits:
    base = Limits(
        max_steps=30,
        max_wall_clock_s=120,
        max_target_attempts=2,
        max_settle_observations=2,
        settle_interval_ms=100,
        max_help_requests=3,
    )
    return dataclasses.replace(base, **changes)


def _ax(target_id: str, route: str, role: str, name: str) -> StructuralTarget:
    return StructuralTarget(
        target_id,
        route,
        LocatorForm.ACCESSIBILITY,
        role,
        constant(name),
        "",
        None,
        None,
        (),
        None,
    )


def _dom(target_id: str, route: str, element: str) -> StructuralTarget:
    return StructuralTarget(
        target_id,
        route,
        LocatorForm.DOM,
        "",
        None,
        "dd",
        DomAttribute.ID,
        constant(element),
        (),
        None,
    )


def _node(node_id: str, kind: ActionKind, route: str, **changes) -> ActionNode:
    base = ActionNode(
        node_id=node_id,
        kind=kind,
        route=route,
        target=None,
        value=None,
        destination=None,
        effect=None,
        record=None,
        result_record=None,
        into=None,
        approval=Approval.NONE,
        mandatory=False,
        requires=(),
        verify=(),
        transitions=(),
    )
    return dataclasses.replace(base, **changes)


def go(to: str, *when, origin: EdgeOrigin = EdgeOrigin.OBSERVED, limit=None) -> Edge:
    return Edge(to, tuple(when), origin, limit)


MEMBER = ref(RefKind.INPUT, "member_id")
AMOUNT = ref(RefKind.INPUT, "amount")
BALANCE = ref(RefKind.OUTPUT, "balance")


def transfer_capability(**changes) -> Capability:
    """A synthetic capability for the transfer workflow. A test fixture only.

    The workflow searches for a member, branches on "no member found", reads
    the balance, opens the transfer screen, types the amount and teller PIN,
    and submits. The profile marks submission as risky and requires record
    evidence.
    """
    members = "/members"
    member = "/members/:id"
    transfer = "/members/:id/transfer"
    targets = (
        _ax("member_field", members, "textbox", "Member number"),
        _ax("search", members, "button", "Search"),
        _ax("no_member", members, "status", "No member found"),
        StructuralTarget(
            "member_shown",
            member,
            LocatorForm.DOM,
            "",
            None,
            "dd",
            DomAttribute.ID,
            constant("member-number"),
            (),
            None,
        ),
        _dom("balance", member, "balance"),
        _ax("transfer_link", member, "link", "Transfer"),
        _dom("transfer_member", transfer, "member-number"),
        _ax("amount", transfer, "textbox", "Amount"),
        _ax("pin", transfer, "textbox", "Teller PIN"),
        _ax("submit", transfer, "button", "Submit transfer"),
        _ax("submitted", transfer, "status", "Transfer submitted"),
    )
    nodes = (
        _node(
            "enter_member",
            ActionKind.TYPE,
            members,
            target="member_field",
            value=MEMBER,
            transitions=(go("search"),),
        ),
        _node(
            "search",
            ActionKind.CLICK,
            members,
            target="search",
            effect="search_member",
            transitions=(
                go("read_balance", Shows("member_shown", MEMBER, TextMatch.EQUALS)),
                go("not_found", Present("no_member")),
            ),
        ),
        _node(
            "read_balance",
            ActionKind.READ,
            member,
            target="balance",
            into=BALANCE,
            transitions=(go("open_transfer"),),
        ),
        _node(
            "open_transfer",
            ActionKind.CLICK,
            member,
            target="transfer_link",
            effect="open_transfer",
            verify=(AtRoute(transfer),),
            transitions=(go("enter_amount"),),
        ),
        _node(
            "enter_amount",
            ActionKind.TYPE,
            transfer,
            target="amount",
            value=AMOUNT,
            transitions=(go("enter_pin"),),
        ),
        _node(
            "enter_pin",
            ActionKind.TYPE,
            transfer,
            target="pin",
            value=ref(RefKind.SECRET, "teller_pin"),
            transitions=(go("submit"),),
        ),
        _node(
            "submit",
            ActionKind.CLICK,
            transfer,
            target="submit",
            effect="submit_transfer",
            record=RecordSpec("transfer_member", MEMBER, Relation.CONTAINER),
            approval=Approval.EACH_RUN,
            mandatory=True,
            verify=(Present("submitted"),),
            transitions=(go("done"),),
        ),
        ResultNode(
            "done",
            ResultKind.SUCCESS,
            "",
            (
                Present("submitted"),
                Shows("transfer_member", MEMBER, TextMatch.EQUALS),
                Bound(BALANCE),
            ),
        ),
        ResultNode(
            "not_found", ResultKind.OUTCOME, "member_not_found", (Present("no_member"),)
        ),
    )
    capability = Capability(
        schema_version=SCHEMA_VERSION,
        capability_id="member_transfer",
        version=1,
        application=Application(
            "bank/core/sandbox",
            SurfaceKind.BROWSER,
            ORIGIN,
            members,
            (Present("member_field"),),
        ),
        provenance=Provenance(
            ProvenanceKind.SYNTHETIC, "", "hand_written_fixture", Review.REVIEWED
        ),
        inputs=(
            Field("member_id", ValueType.DIGITS, True, 5, 10, ()),
            Field("amount", ValueType.DECIMAL, True, 1, 12, ()),
        ),
        outputs=(Field("balance", ValueType.DECIMAL, False, 1, 20, ()),),
        variables=(),
        secrets=("teller_pin",),
        outcomes=("member_not_found",),
        templates=(),
        targets=targets,
        entry="enter_member",
        nodes=nodes,
        restrictions=(),
        limits=limits(),
    )
    return dataclasses.replace(capability, **changes)


INPUTS = {"member_id": "10001", "amount": "250.00"}


def png(
    size: tuple[int, int], marks: tuple[tuple[int, int], ...], mark: int = 0
) -> bytes:
    """Draw a synthetic canvas with a patterned button at each position."""
    picture = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(picture)
    for left, top in marks:
        draw.rectangle((left, top, left + 39, top + 19), fill=(20, 60, 160))
        draw.rectangle((left + 6, top + 6, left + 14 + mark, top + 13), fill="yellow")
        draw.line((left + 20, top + 4, left + 34, top + 15), fill="white", width=2)
    stream = io.BytesIO()
    picture.save(stream, format="PNG")
    return stream.getvalue()


def crop(image: bytes, left: int, top: int, width: int, height: int) -> bytes:
    with Image.open(io.BytesIO(image)) as picture:
        piece = picture.crop((left, top, left + width, top + height))
        stream = io.BytesIO()
        piece.save(stream, format="PNG")
    return stream.getvalue()


@dataclasses.dataclass
class Captures:
    """Hands out fresh synthetic captures of a canvas."""

    images: list[bytes]
    taken: int = 0
    frames: list[tuple[str, ...]] = dataclasses.field(default_factory=list)

    def capture(self, frame: tuple[str, ...]) -> Scene | None:
        self.frames.append(frame)
        self.taken += 1
        image = self.images[0] if len(self.images) == 1 else self.images.pop(0)
        return Scene(f"cap-{self.taken}", image)
