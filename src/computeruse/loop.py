"""The discovery loop: look, decide, check policy, act, record.

The caller supplies every collaborator, so this module imports no browser
driver, model client, or clock. A third-party import here would bypass one of
those protocols.

One decision produces one move. The loop does not batch, because each move
needs its own policy verdict, escalation decision, and recorded outcome. A
step is also the unit the recorder writes to a capability artifact.

Perception follows the same sequence. The model names one tool, the gate
checks whether the operator granted it, and only then does that tool read the
page. The loop does not take an unused screenshot at the start.

A model saying the goal is met does not complete a run. Before the first
decision the run settles what the goal requires: the records it is about and
the outputs it asks for. The claim carries checks, and the loop reads each
checked control again from the live page, through the same checks as any
read, and then requires the passing checks to cover every record and output,
tied to one another. A claim it cannot check goes to a person, and one that
fails comes back to the model as a notice.

A new window or tab goes to a person. The loop looks for one before every
decision and after every action, and sends no more input until the person
hands the session back.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import deque
from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType

from computeruse import coverage, matching, operations, policy
from computeruse.actions import (
    DIALOG_ACTIONS,
    Action,
    ActionResult,
    AxLocator,
    AxNode,
    DomLocator,
    Expectation,
    Observation,
    ObservationMode,
    ObservationProvenance,
    ObservationRequest,
    Operation,
    Outcome,
    PageInfo,
    PageState,
    PendingDialog,
    RecordEvidence,
    Relation,
    ScopeKind,
    ScreenTarget,
    Target,
    VisualAnchor,
    describe,
    names_node,
    separates,
    target_form,
    unbound,
    word_positions,
)
from computeruse.budget import Budget, Clock
from computeruse.control import Control, ControlChanged, Overtaken, explain
from computeruse.decider import (
    MAX_FACT_VALUE,
    AskHuman,
    CheckKind,
    Decider,
    Decision,
    Fact,
    FactOrigin,
    FactRef,
    Finish,
    FlagRisk,
    HumanReturn,
    InvalidDecisionError,
    Labeller,
    ModelError,
    Observe,
    Propose,
    Remember,
    ResultCheck,
    RiskFinding,
    Task,
    Transcript,
    Turn,
    uncovered_requirements,
)
from computeruse.diagnostics import FailureEvidence, snapshot
from computeruse.escalation import (
    Ask,
    Escalator,
    Handoff,
    HandoffOutcome,
    InterventionRequest,
    Mode,
    Trigger,
)
from computeruse.journal import (
    Acted,
    Corrected,
    Decided,
    Escalated,
    Failed,
    Flagged,
    Interpreted,
    Journal,
    Observed,
    Refused,
    Remembered,
    RunEnded,
    RunStarted,
    Superseded,
    Verified,
    describe_target,
    describe_value,
)
from computeruse.manual import (
    ManualEvent,
    ManualSegment,
    single_click,
    summary,
)
from computeruse.policy import Restriction, Source
from computeruse.profile import (
    OUTCOMES,
    RECORD_NOT_FOUND,
    ActionKind,
    Environment,
    Limit,
    Profile,
    Risk,
    is_effect_name,
)
from computeruse.recording import DiscoveryTrace
from computeruse.surface import Surface, SurfaceError, control_at, operation_binding

_NO_OUTPUTS: Mapping[str, str] = MappingProxyType({})

MAX_FACTS = 24
"""Maximum facts in working memory before the oldest is dropped."""

MAX_REFUSALS = 12
"""Maximum total refusals before the run asks a person.

A refusal, a declined approval, and a repeated finding each count. The run
also hands over after more refusals in a row than the profile's retries
per step allow; this total stops a loop that interleaves harmless work
between refusals (rule 17)."""

_UNCHANGING = frozenset(
    {
        ActionKind.READ,
        ActionKind.ASSERT,
        ActionKind.WAIT_FOR,
        ActionKind.WAIT,
        ActionKind.SCROLL,
        ActionKind.MOVE,
    }
)
"""Actions that change what the page shows only by moving it, if at all.

A value the executor read stays current across these, as long as the page's
location stays the same. Any other action may have changed a record.
"""


class Ending(StrEnum):
    """The reason a run stopped."""

    COMPLETED = "completed"
    EXHAUSTED = "exhausted"
    HANDED_OFF = "handed_off"
    BLOCKED = "blocked"
    FAILED = "failed"
    TERMINATED = "terminated"


class Verification(StrEnum):
    """The source that established a completed run's result.

    ``EXECUTOR`` means every check passed against the live page. ``PERSON``
    means the executor could not check the claim on this surface and a person
    confirmed it in the live session.
    """

    EXECUTOR = "executor"
    PERSON = "person"


@dataclasses.dataclass(frozen=True, slots=True)
class CheckResult:
    """One check of a finish claim, and what the page showed for it.

    ``bound`` is the record identifier the reading was tied to, as the
    surface or the loop verified it: the check's own record evidence, the
    record a cited fact was kept with, or the record a painted screen names
    once. It is empty for a reading tied to no record. ``screen`` holds, for
    a painted reading, every line the same picture shows.
    """

    check: ResultCheck
    shown: str | None
    passed: bool
    detail: str = ""
    bound: str = ""
    screen: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class RunResult:
    """The caller-facing result of a run."""

    ending: Ending
    steps: int
    detail: str
    outputs: Mapping[str, str] = _NO_OUTPUTS
    restrictions: tuple[Restriction, ...] = ()
    """Every restriction the run bound to an operation, for a later recording.

    These are not written to the profile or anywhere else by the run. The
    discovery trace carries them to the capability recorder on export.
    """
    verification: Verification | None = None
    checks: tuple[CheckResult, ...] = ()
    """The checks that completed the run, with what the page showed.

    They hold member data, like ``outputs``, so they are returned to the
    caller and never journaled.
    """
    task: Task = dataclasses.field(default_factory=Task)
    """The requirements derived from the goal before the first step."""
    segments: tuple[ManualSegment, ...] = ()
    """Typed records of a person's actions during each intervention.

    Live data like ``outputs``: steps name controls by their accessible
    names. The journal holds the sanitized record of the same events.
    """
    outcome: str = ""
    """The business outcome a completed run showed instead of outputs, if any."""


def discover(
    goal: str,
    profile: Profile,
    *,
    surface: Surface,
    decider: Decider,
    journal: Journal,
    clock: Clock,
    escalator: Escalator | None = None,
    control: Control | None = None,
    trace: DiscoveryTrace | None = None,
    inputs: Mapping[str, str] = _NO_OUTPUTS,
    outputs: tuple[str, ...] = (),
) -> RunResult:
    """Drive ``surface`` toward ``goal`` under ``profile`` until it stops.

    The loop stops for exactly six reasons: the decider reported the goal
    met, a declared budget is exhausted, a human took the run and did not give
    it back, the surface moved outside the profile's routes, a collaborator
    failed in a way the run cannot continue past, or the operator terminated
    the run.

    Parameters
    ----------
    goal
        Natural-language objective, retained only in live working context.
    profile
        The operator's policy. The only source of what may be reached, looked
        at, and done; nothing here supplies a default.
    surface
        The live application. Owns its own entry point, so the loop never
        opens a session outside the profile.
    decider
        Chooses each next decision. The only place a model appears.
    journal
        Receives evidence as it happens.
    clock
        Monotonic seconds, for the wall-clock budget.
    escalator
        Routes risky actions and stuck runs to a human and waits for the
        answer. Give this or ``control``, not both.
    control
        The run's control, with its live operator channels and the seat a
        person takes the session at. Without one, the run gets a control
        whose only channel is ``escalator``.
    outputs
        The output names an operator's export contract declares. When given,
        the model's reading of the goal must name exactly these outputs, so
        a claim cannot complete under a name the capability could not save.

    Returns
    -------
    RunResult
        The ending, the number of steps taken, a human-readable reason, and
        any outputs the run reported on success.
    """
    if (escalator is None) == (control is None):
        raise TypeError("a run needs exactly one of an escalator or a control")
    if control is None:
        control = Control(mode=Mode.DISCOVERY, clock=clock, escalator=escalator)
    run = _Run(
        goal=goal,
        profile=profile,
        surface=control.guard(surface),
        decider=decider,
        control=control,
        journal=journal,
        budget=Budget(profile.budgets, clock),
        trace=trace,
        inputs=inputs,
        outputs=outputs,
    )
    return run.execute()


class _Terminated(Exception):  # noqa: N818  the operator's command, not a fault
    """The operator terminated the run; unwinds to ``execute``."""


@dataclasses.dataclass(slots=True, kw_only=True, eq=False, repr=False)
class _Run:
    """One run's mutable state, so each stage can be read on its own.

    The loop carries state that changes together: the observations that still
    describe the screen in view, the dialog in front of it if there is one,
    how much of the alternate allowance this unresolved action has spent, the
    consecutive failure count, the notices owed to the next decision, and the
    current route. Threading those through
    free functions would hide the control flow that has to stay auditable, so
    they live here and each stage is a method.
    """

    goal: str
    profile: Profile
    surface: Surface
    decider: Decider
    control: Control
    journal: Journal
    budget: Budget
    trace: DiscoveryTrace | None = None
    inputs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    observations: dict[ObservationMode, Observation] = dataclasses.field(
        init=False, default_factory=dict
    )
    pending: PendingDialog | None = dataclasses.field(init=False, default=None)
    pending_at: str = dataclasses.field(init=False, default="")
    restrictions: list[Restriction] = dataclasses.field(
        init=False, default_factory=list
    )
    performed: dict[int, Operation] = dataclasses.field(
        init=False, default_factory=dict
    )
    flagged: set[int] = dataclasses.field(init=False, default_factory=set)
    hits: dict[tuple[str, float, float], str] = dataclasses.field(
        init=False, default_factory=dict
    )
    risky_because: str = dataclasses.field(init=False, default="")
    refusals: int = dataclasses.field(init=False, default=0)
    refused_total: int = dataclasses.field(init=False, default=0)
    lost: set[str] = dataclasses.field(init=False, default_factory=set)
    missing: set[str] = dataclasses.field(init=False, default_factory=set)
    attempted: list[ObservationMode] = dataclasses.field(
        init=False, default_factory=list
    )
    history: list[Turn] = dataclasses.field(init=False, default_factory=list)
    notices: deque[str] = dataclasses.field(
        init=False, default_factory=lambda: deque(maxlen=24)
    )
    alternates_used: int = dataclasses.field(init=False, default=0)
    failures: int = dataclasses.field(init=False, default=0)
    verification_epoch: int = dataclasses.field(init=False, default=-1)
    verification_failures: int = dataclasses.field(init=False, default=0)
    cycles: dict[tuple[object, ...], int] = dataclasses.field(
        init=False, default_factory=dict
    )
    route: str = dataclasses.field(init=False, default="")
    memory: dict[str, Fact] = dataclasses.field(init=False, default_factory=dict)
    epoch: int = dataclasses.field(init=False, default=0)
    task: Task = dataclasses.field(init=False, default=Task())
    windows: set[str] = dataclasses.field(init=False, default_factory=set)
    page_id: str = dataclasses.field(init=False, default="")
    human_checks: tuple[ResultCheck, ...] = dataclasses.field(init=False, default=())
    human_trigger: Trigger | None = dataclasses.field(init=False, default=None)
    checked_after: tuple[ResultCheck, ...] = dataclasses.field(init=False, default=())
    human_verified: tuple[ResultCheck, ...] = dataclasses.field(init=False, default=())
    handed_dialogs: set[str] = dataclasses.field(init=False, default_factory=set)
    in_hand: int = dataclasses.field(init=False, default=0)
    """The step whose decision the loop is carrying out, or 0 between steps."""
    sent: set[str] = dataclasses.field(init=False, default_factory=set)
    sent_operations: dict[str, Operation] = dataclasses.field(
        init=False, default_factory=dict
    )
    """Risky operations already dispatched, keyed by operation identity."""
    uncovered: tuple[Finish, tuple[str, ...], tuple[CheckResult, ...], int] | None = (
        dataclasses.field(init=False, default=None)
    )
    """The last business outcome whose checks passed but left coverage gaps.

    It is kept with its gaps, its readings, and the epoch it was read in, so
    a request to verify the result can offer that claim for approval while
    nothing has changed since. A later claim that fails does not clear it.
    """

    def execute(self) -> RunResult:
        """Run until a stopping condition, reporting which one it was."""
        self.journal.record(RunStarted(profile_id=self.profile.profile_id))
        try:
            if not self.control.begin(
                profile=self.profile,
                budget=self.budget,
                journal=self.journal,
                context=self.goal,
                restrictions=lambda: tuple(self.restrictions),
            ):
                return self._end(
                    Ending.TERMINATED,
                    0,
                    "the operator terminated the run before it began",
                )
            return self._drive()
        except _Terminated:
            return self._terminated()
        except ModelError:
            # A call is given only the time the run has left, so a call cut
            # short by that limit is the budget running out, not the model.
            spent = self.budget.exhausted(starting_step=False)
            if spent is not None:
                return self._end(Ending.EXHAUSTED, self.budget.steps, spent.value)
            return self._fail(
                "model", "the decider could not produce a usable decision"
            )
        except SurfaceError:
            return self._fail("surface", "the surface could not continue")
        finally:
            self.control.finish(Ending.FAILED.value)

    def _drive(self) -> RunResult:
        if self.profile.environment is not Environment.SANDBOX:
            # Discovery tries operations to learn what they do. The profile
            # saying "sandbox" is the operator's statement that this is
            # acceptable; it does not make the target isolated.
            return self._end(Ending.BLOCKED, 0, "discovery runs only in a sandbox")
        route = policy.route_for(self.profile, self.surface.location())
        if route is None:
            return self._end(Ending.BLOCKED, 0, "surface opened outside the profile")
        self.route = route
        self._adopt_windows(everything=False)
        unsettled = self._interpret()
        if unsettled is not None:
            return unsettled
        while True:
            try:
                finished = self._step()
            except ControlChanged:
                # Control changed hands between a check and the surface call.
                # The refused operation was not sent, so the step pauses at its
                # next boundary. Only a pending decision is superseded. A step whose
                # action already reached the surface was performed, and a look
                # refused after a resume belongs to no decision at all.
                if self.in_hand and self.in_hand not in self.performed:
                    self.control.discard()
                    self.journal.record(Superseded(step=self.in_hand))
                finished = None
            finally:
                self.in_hand = 0
            if finished is not None:
                return finished

    def _step(self) -> RunResult | None:
        if self.control.halted():
            return self._hold()
        location = self.surface.location()
        route = policy.route_for(self.profile, location)
        if route is None:
            return self._blocked()
        self.route = route
        opened = self._new_windows()
        if opened:
            return self._hand_window(self.budget.steps, opened)
        spent = self.budget.exhausted()
        if spent is not None:
            return self._end(Ending.EXHAUSTED, self.budget.steps, spent.value)
        self.budget.charge_step()
        step = self.budget.steps

        transcript = self._transcript(location)
        try:
            decision = self.control.call(lambda: self.decider.decide(transcript))
        except InvalidDecisionError as error:
            return self._invalid_decision(step, error)
        if isinstance(decision, Overtaken) or self.control.halted():
            # Control changed hands while the model was deciding. What it
            # decided concerned a session somebody else now holds, so it is
            # dropped here, before anything reads it.
            self.control.discard()
            self.journal.record(Superseded(step=step))
            self.budget.refund_step()
            return None
        self.in_hand = step
        if not isinstance(decision, Observe):
            # Refreshing evidence must not consume the feedback needed to use it.
            self.notices.clear()
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)
        if self.surface.location() != location:
            self._forget()
            self.notices.append("the surface moved during the decision; look again")
            return None
        ended = self._dispatch(step, decision, location)
        if ended is None and (
            self.refusals > self.profile.budgets.max_retries_per_step
            or self.refused_total > MAX_REFUSALS
        ):
            # Refused proposals, declined approvals, and repeated findings
            # are no progress, however the model phrases them (rule 17).
            return self._hand_over(
                step, Trigger.NO_PROGRESS, "repeated refusals made no progress"
            )
        return ended

    def _dispatch(
        self, step: int, decision: Decision, location: str
    ) -> RunResult | None:
        """Carry out one decision, each kind by its own stage."""
        decision = _with_identifiers(
            decision, tuple(record.value for record in self.task.records)
        )
        match decision:
            case Observe(request=request):
                return self._requested_look(step, request, location)
            case Propose():
                return self._proposal(step, decision, location)
            case FlagRisk(finding=finding):
                self.journal.record(Decided(step=step, kind=None))
                self._flag(finding)
                return None
            case Remember():
                self.journal.record(Decided(step=step, kind=None))
                return self._keep(step, decision)
            case AskHuman():
                return self._human_request(step, decision, location)
            case Finish():
                if self.pending is not None:
                    self.notices.append(
                        f"{self.pending.dialog_id} is still waiting; what opened "
                        "it has not finished, so the goal cannot be met yet"
                    )
                    return None
                self.journal.record(Decided(step=step, kind=None))
                return self._finish(step, decision, location)

    def _proposal(
        self, step: int, decision: Propose, location: str
    ) -> RunResult | None:
        action = decision.action
        if decision.fact:
            recalled = self._recall(step, action, decision.fact)
            if not isinstance(recalled, Action):
                return recalled
            action = recalled
        if decision.input_name:
            if (
                decision.input_name not in self.inputs
                or action.value != self.inputs[decision.input_name]
            ):
                self.notices.append("the proposal does not match its named input")
                return None
            if self.trace is not None:
                self.trace.input_used(step, decision.input_name)
        if decision.fact and self.trace is not None:
            self.trace.used(step, decision.fact)
        action = self._row_evidence(action)
        ended = self._propose(step, action, location, decision.rationale)
        if ended is not None or step not in self.performed or not decision.after:
            return ended
        return self._after(step, decision.after, sent=True)

    def _human_request(
        self, step: int, decision: AskHuman, location: str
    ) -> RunResult | None:
        # A person's verified step is checked again before anyone is asked for
        # it again, whether the new request repeats its checks, adds to them,
        # or names none at all.
        checks = decision.after or self.human_verified
        if (
            self.human_verified
            and decision.trigger is self.human_trigger
            and set(self.human_verified) <= set(checks)
        ):
            ended = self._after(step, checks, notify=False)
            if ended is not None:
                return ended
            if self.checked_after:
                self.notices.append(
                    "the requested continuation already verifies; continue or finish"
                )
                return None
        uncovered, self.uncovered = self.uncovered, None
        if (
            decision.trigger is Trigger.UNVERIFIED_RESULT
            and uncovered is not None
            and uncovered[3] == self.epoch
        ):
            # The claim's checks passed on the page as it still is, and only
            # a person can cover the rest, so the person is asked to confirm
            # that claim rather than to help with no claim in hand.
            claim, gaps, results, _ = uncovered
            return self._confirm(step, claim, gaps, results)
        self.human_checks = decision.after
        self.human_trigger = decision.trigger
        self.human_verified = ()
        return self._ask_human(step, decision.trigger, decision.detail, location)

    def _after(
        self,
        step: int,
        checks: tuple[ResultCheck, ...],
        *,
        notify: bool = True,
        sent: bool = False,
    ) -> RunResult | None:
        self.checked_after = ()
        verified: list[ResultCheck] = []
        for check in checks:
            result = self._check(step, check)
            if isinstance(result, RunResult):
                return result
            if not result.passed:
                failure = (
                    f"step {step} postcondition on {check.target!r} "
                    f"expected {check.match.value} {check.expected!r}; {result.detail}"
                )
                if notify and sent:
                    # The action reached the application, so what it changed
                    # is there to see. A retry could make the change twice.
                    self.notices.append(
                        f"{failure}; the action was sent, and its postcondition did "
                        "not hold; look at what it did before anything else, and "
                        "do not send it again"
                    )
                elif notify:
                    self.notices.append(
                        f"{failure}; the action's postcondition did not hold; "
                        "inspect before retrying"
                    )
                return None
            verified.append(check)
        self.checked_after = tuple(verified)
        if self.trace is not None:
            self.trace.verified(
                step, tuple(verified), self.observations.get(ObservationMode.STRUCTURED)
            )
        return None

    # Check completion claims against the page.

    def _finish(self, step: int, claim: Finish, location: str) -> RunResult | None:
        """Check a finish claim against the live page, and complete only if it holds.

        The claim's checks are read again from the surface, each through the
        same checks as any read, and compared with what the claim says. Then
        the passing checks must cover the task: every record and output it
        requires, tied to one another as declared. Malformed or failed checks
        permit corrective retries. Passing checks with coverage gaps go
        directly to a person with those exact obligations. A claim this
        surface or profile cannot check also goes to a person, who may
        confirm it in the live session.
        """
        problem = self._claim_problem(claim)
        if problem is not None:
            return self._unverified(step, problem, claim, (problem,))
        if claim.outcome:
            return self._finish_outcome(step, claim, location)
        missing = [
            item.name for item in self.task.outputs if item.name not in claim.outputs
        ]
        if missing:
            named = ", ".join(repr(name) for name in missing)
            gap = f"the task asks for {named}, which the claim does not report"
            return self._unverified(step, gap, claim, (gap,))
        if self.control.halted():
            return None
        looked = self._look_before_checking(step, claim)
        if looked is not None:
            return looked
        if not all(self._checkable(check, location) for check in claim.checks):
            return self._confirm(
                step, claim, ("the executor cannot read these checks on this surface",)
            )
        results: list[CheckResult] = []
        for check in claim.checks:
            checked = self._check(step, check)
            if isinstance(checked, RunResult):
                return checked
            results.append(checked)
        results = list(_painted_ties(self.task, tuple(results)))
        passed = sum(1 for item in results if item.passed)
        gaps = _uncovered(self.task, tuple(results)) if passed == len(results) else []
        self.journal.record(
            Verified(step=step, checks=len(results), passed=passed, uncovered=len(gaps))
        )
        if passed < len(results):
            failed = [
                f"{item.check.kind.value} check expecting {item.check.expected!r} "
                f"{item.detail}"
                for item in results
                if not item.passed
            ]
            return self._unverified(
                step,
                f"the result did not verify: {'; '.join(failed)}",
                claim,
                tuple(failed),
                tuple(results),
            )
        if gaps:
            return self._confirm(step, claim, tuple(gaps), tuple(results))
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            # Verification ran past the deadline; a result checked late is not
            # reported as complete.
            return self._end(Ending.EXHAUSTED, step, spent.value)
        return self._end(
            Ending.COMPLETED,
            step,
            "every check of the result passed on the live page",
            claim.outputs,
            verification=Verification.EXECUTOR,
            checks=tuple(results),
        )

    def _label_person(
        self, handoff: Handoff, segments: tuple[ManualSegment, ...]
    ) -> None:
        """Label a person's single click, so a recording may automate it.

        The decider names what the click accomplished, as it names its own
        actions, and the gate judges the labelled click the same way. A denied
        label, a learned deny, or no label at all leaves the click a person's
        step. A risky one is automated only with an approval on every run.
        """
        event = single_click(segments)
        if (
            self.trace is None
            or event is None
            or event.target is None
            or not isinstance(self.decider, Labeller)
        ):
            return
        label = self.decider.label_effect(
            event.target.role, event.target.name, event.route
        )
        learned = (
            policy.learned_limit(
                self.restrictions,
                event.operation,
                frozenset(self.lost),
                frozenset(self.missing),
            )
            if event.operation is not None
            else Limit.RISKY
        )
        careful = _judged_label(self.profile, event, label, learned)
        if careful is None or label is None:
            return
        self.trace.labelled(handoff.intervention, label, careful=careful)

    def _claim_problem(self, claim: Finish) -> str | None:
        """Return why a claim cannot be checked at all, before any reading."""
        problem = _unsupported(
            claim, self.goal, tuple(record.value for record in self.task.records)
        )
        if problem is None and _changed_requirements(self.task, claim):
            problem = "a requirement check changed the settled task"
        return problem

    def _finish_outcome(
        self, step: int, claim: Finish, location: str
    ) -> RunResult | None:
        """Check a claim that the application answered with a business outcome.

        The claim names an outcome from ``OUTCOMES`` and no outputs. Its checks
        must pass, and together show a state that gives the outcome and a
        control holding each of the goal's record identifiers. For
        ``record_not_found`` that control holds exactly the identifier, and may
        be the search field itself, and no table row may show it as a result.
        On a painted screen the identifier is shown by a line that holds it
        among other words, such as the answer naming what was searched for.
        For the other outcomes the record exists, and a control showing its
        identifier once as a whole word, such as "NM38 - inactive", holds it.
        The task's required context, such as the operator, is covered as for
        any claim.
        """
        if claim.outcome not in OUTCOMES or claim.outputs:
            problem = "an outcome claim names a known outcome and reports no outputs"
            return self._unverified(step, problem, claim, (problem,))
        if self.control.halted():
            return None
        if not all(self._checkable(check, location) for check in claim.checks):
            problem = "the executor cannot read these checks on this surface"
            return self._unverified(step, problem, claim, (problem,))
        results: list[CheckResult] = []
        for check in claim.checks:
            checked = self._check(
                step, check, searched=True, exact=claim.outcome == RECORD_NOT_FOUND
            )
            if isinstance(checked, RunResult):
                return checked
            results.append(checked)
        passed = sum(1 for item in results if item.passed)
        gaps = _outcome_gaps(
            self.task,
            claim.outcome,
            tuple(results),
            self.observations.get(ObservationMode.STRUCTURED),
        )
        self.journal.record(
            Verified(step=step, checks=len(results), passed=passed, uncovered=len(gaps))
        )
        if passed < len(results) or gaps:
            failed = [
                f"{item.check.kind.value} check expecting {item.check.expected!r} "
                f"{item.detail}"
                for item in results
                if not item.passed
            ]
            problems = (*failed, *gaps)
            return self._unverified(
                step,
                f"the outcome did not verify: {'; '.join(problems)}",
                claim,
                problems,
                tuple(results),
            )
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)
        return self._end(
            Ending.COMPLETED,
            step,
            f"the page showed the outcome {claim.outcome}",
            verification=Verification.EXECUTOR,
            checks=tuple(results),
            outcome=claim.outcome,
        )

    def _look_before_checking(self, step: int, claim: Finish) -> RunResult | None:
        """Take one structured look, so record evidence is judged on the page now.

        Record evidence is checked against the structured observation. One
        from before the last change can name a control twice or not at all,
        and fail a claim the current page supports. Every other check reads
        the live page directly, so only a claim with record evidence needs
        this. The look is the run's own, gated like any other, and spends no
        alternate look.
        """
        if all(check.record is None for check in claim.checks):
            return None
        if ObservationMode.STRUCTURED not in self.profile.perception.allowed_modes:
            return None
        request = ObservationRequest(
            ObservationMode.STRUCTURED, ObservationProvenance.REFRESH
        )
        looked = self._look(step, request, self.surface.location(), alternate=False)
        return looked if isinstance(looked, RunResult) else None

    def _checkable(self, check: ResultCheck, location: str) -> bool:
        """Report whether this surface and profile let the loop read ``check``.

        A check that needs no read, such as one citing a fact kept on another
        page, is checkable here and judged by ``_check_fact``. One whose read
        the profile does not grant, or grants only on another route, cannot
        be checked by the executor at all. A read declared risky is checkable,
        and asks a person first. A read that lacks the record evidence the
        profile requires is checkable too, and fails, so the claim comes back
        to be corrected rather than to a person who could approve it.
        """
        target = check.target
        evidence = check.record
        if isinstance(target, FactRef):
            fact = self.memory.get(target.key)
            if fact is None or fact.source is None or not self._fact_here(fact):
                return True
            target, evidence = fact.source, fact.record
        read = Action(ActionKind.READ, target, evidence=evidence)
        supported = self.surface.capabilities().get(ActionKind.READ, frozenset())
        if target_form(target) not in supported:
            return False
        verdict = policy.evaluate(read, self.profile, location)
        if isinstance(verdict, policy.Denied):
            return verdict.reason is policy.Denial.RECORD_EVIDENCE_MISSING
        return True

    def _check(
        self,
        step: int,
        check: ResultCheck,
        *,
        searched: bool = False,
        exact: bool = True,
    ) -> CheckResult | RunResult:
        """Read one checked control from the live page and compare it.

        ``searched`` marks a check of an outcome claim. Its record check may
        read the search field, because a field shows what was searched for.
        With ``exact`` the control must hold exactly the identifier; without
        it, the identifier once as a whole word. A painted line is compared
        as any painted record check is: the identifier once among other
        words, such as an answer naming what was searched for. A line that
        is only the identifier may be the field the run typed it into, and
        recognition can misread a field, so it proves nothing.
        """
        target = check.target
        if isinstance(target, FactRef):
            return self._check_fact(step, check, target)
        read = Action(ActionKind.READ, target, evidence=check.record)
        record = check.kind is CheckKind.RECORD
        # A field shows what was typed into it, so it cannot say which record
        # the page is about. A required state may come from a field only when
        # nobody changed that field in the document it sits in.
        reading = self._gated_read(
            step,
            read,
            displayed=record and not searched,
            committed=check.kind is CheckKind.REQUIREMENT,
        )
        if isinstance(reading, RunResult):
            return reading
        if isinstance(reading, str):
            return CheckResult(check, None, False, reading)
        if reading.outcome is not Outcome.OK or reading.extracted is None:
            return CheckResult(
                check, None, False, f"could not be read: {_failure_detail(reading)}"
            )
        bound = check.record.value if check.record is not None else ""
        if searched and record and not isinstance(target, ScreenTarget):
            shown = reading.extracted.strip()
            held = shown == check.expected or (
                not exact and len(word_positions(shown, check.expected)) == 1
            )
            detail = (
                ""
                if held
                else "the control does not hold exactly that identifier"
                if exact
                else "the control does not show that identifier once as a whole word"
            )
            return CheckResult(check, reading.extracted, held, detail)
        compared = _compared(check, reading.extracted, bound=bound)
        return dataclasses.replace(compared, screen=reading.screen)

    def _check_fact(
        self, step: int, check: ResultCheck, target: FactRef
    ) -> CheckResult | RunResult:
        """Compare a check with a fact the page showed, if it cannot have changed.

        A fact kept on the page the run is on is read again from its control,
        through the gate, with the record evidence it was kept with. One kept
        from a control and marked as unchanging otherwise qualifies as kept.
        So does one the executor read on the page the run is still on, when no
        action that could change the page has run since. A fact from the goal
        never qualifies: it is what the claim is checked against, not
        evidence. Only the fact's record ties it to a record.
        """
        fact = self.memory.get(target.key)
        if fact is None:
            return CheckResult(check, None, False, "names no fact in working memory")
        if fact.origin is FactOrigin.GOAL:
            return CheckResult(check, None, False, "cites a fact the goal supplied")
        if _displayed_only(check) and not fact.displayed:
            return CheckResult(
                check, None, False, "cites a field, which shows what was typed"
            )
        bound = fact.record.value if fact.record is not None else ""
        if fact.source is not None and self._fact_here(fact):
            refreshed = self._reread(step, fact)
            if isinstance(refreshed, RunResult):
                return refreshed
            if isinstance(refreshed, str):
                return CheckResult(check, None, False, refreshed)
            return _compared(check, refreshed.value, f"read at step {step}", bound)
        kept = f"kept at step {fact.step}"
        if fact.origin is FactOrigin.CONTROL and not fact.may_change:
            return _compared(check, fact.value, kept, bound, painted=fact.painted)
        if self._fact_here(fact) and fact.epoch == self.epoch:
            return _compared(check, fact.value, kept, bound, painted=fact.painted)
        return CheckResult(
            check, None, False, "cites a fact that may have changed; read it again"
        )

    def _unverified(
        self,
        step: int,
        problem: str,
        claim: Finish,
        gaps: tuple[str, ...] = (),
        results: tuple[CheckResult, ...] = (),
    ) -> RunResult | None:
        """Report a claim that did not hold, and hand over once retries run out."""
        self.failures += 1
        if self.verification_epoch != self.epoch:
            self.verification_epoch = self.epoch
            self.verification_failures = 0
        self.verification_failures += 1
        self.notices.append(
            f"{problem}; the run is not complete. Correct the checks, keep "
            "working, or ask for a person"
        )
        # A passing claim stays on offer until the page changes or a newer
        # claim passes; a later failing claim does not change the page.
        if results and all(item.passed for item in results):
            self.uncovered = (claim, gaps, results, self.epoch)
        if max(self.failures, self.verification_failures) <= (
            self.profile.budgets.max_retries_per_step
        ):
            return None
        if self.uncovered is not None and self.uncovered[3] == self.epoch:
            claim, gaps, results, _ = self.uncovered
        return self._confirm(step, claim, gaps, results)

    def _confirm(
        self,
        step: int,
        claim: Finish,
        gaps: tuple[str, ...] = (),
        results: tuple[CheckResult, ...] = (),
    ) -> RunResult | None:
        """Ask a person to confirm a result the loop could not verify itself.

        The person is shown the claim's outputs and what could not be
        verified, in the live channel only. An approval completes the run
        with those outputs, marked as confirmed by a person. A person who
        takes the session and hands it back resumes the run. Anything else
        ends it handed off.
        """
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)
        handoff = self._intervene(
            step,
            Trigger.UNVERIFIED_RESULT,
            "the result could not be verified against the page",
            outputs=claim.outputs,
            unverified=gaps,
            ask=Ask.APPROVAL,
        )
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)
        if (
            handoff.outcome is HandoffOutcome.APPROVED
            and claim is not None
            and not handoff.changed
        ):
            self._note(handoff)
            return self._end(
                Ending.COMPLETED,
                step,
                "a person confirmed the result in the live session",
                claim.outputs,
                verification=Verification.PERSON,
                checks=results,
            )
        if handoff.outcome is HandoffOutcome.RESUMED or handoff.changed:
            self._note(handoff)
            return self._resume(step)
        return self._end(
            Ending.HANDED_OFF,
            step,
            f"a person was asked for: {Trigger.UNVERIFIED_RESULT.value}",
            checks=results,
        )

    # Reads for the run's checks pass through the full read gate.

    def _gated_read(
        self,
        step: int,
        read: Action,
        *,
        displayed: bool = False,
        committed: bool = False,
    ) -> ActionResult | str | RunResult:
        """Read one control for the run itself, through every check a read passes.

        A completion check, a fact cited by one, and a fact refreshed before
        an action uses it all read the page this way. The surface must take
        the target's form, the target must name what the observation showed,
        the gate must permit the read on this route with any record evidence
        the profile requires, and no restriction this run learned may deny
        it. A read declared or learned risky asks a person first, as any
        risky action does. Returns the surface's result, a reason the read was
        refused, or a ``RunResult`` when the run must stop.

        What the person approves is this read on this page, in this window,
        on this operation, captured before they are asked whether or not an
        observation is in hand. After approval all of it is compared again,
        and every check runs again. A read whose page, window, or operation
        changed is not performed: the approval is discarded, the page is
        looked at again through the gate, and the claim goes back to the
        model. The read is sent with the expectation captured before the
        question, never one built from wherever the page is afterwards.

        The execution budget is checked before the read, after approval, and
        before the read is sent, so an expired budget starts no further read.

        Record evidence on these reads must be something the page displays,
        because they establish which record a value belongs to.
        """
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)
        allowed = self._read_allowed(step, read)
        if isinstance(allowed, str):
            return allowed
        route, risk = allowed
        held = self._held_read(read, displayed, committed=committed)
        operation = self._operation(read)
        if risk is Risk.RISKY:
            handoff = self._intervene(
                step,
                Trigger.RISKY_ACTION,
                self.risky_because,
                action=read,
                route=route,
                ask=Ask.APPROVAL,
            )
            spent = self.budget.exhausted(starting_step=False)
            if spent is not None:
                return self._end(Ending.EXHAUSTED, step, spent.value)
            self._note(handoff)
            match handoff.outcome:
                case HandoffOutcome.APPROVED:
                    pass
                case HandoffOutcome.RESUMED:
                    # The person used the session; the run looks again and the
                    # read waits for a new decision.
                    resumed = self._resume(step)
                    return resumed or "a person took the session; the read did not run"
                case HandoffOutcome.REJECTED:
                    self.notices.append(f"the operator declined {read.kind.value}")
                    self._refused()
                    return "the read was not approved"
                case _:
                    return self._end(Ending.HANDED_OFF, step, "handoff timed out")
            if handoff.changed or self._read_moved(held, operation, read):
                return self._discard_approval(step)
            again = self._read_allowed(step, read)
            if isinstance(again, str):
                return again
            if again[0] != route:
                return self._discard_approval(step)
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)
        return self._send(step, read, held)

    def _read_allowed(self, step: int, read: Action) -> tuple[str, Risk] | str:
        """Return the route and risk a read runs at, or why it is refused."""
        unfit = self._unfit(read, looked=False)
        if unfit is not None:
            return unfit
        verdict = policy.evaluate(read, self.profile, self.surface.location())
        risk = self._restricted(step, read, verdict, "")
        if risk is None or not isinstance(verdict, policy.Allowed):
            reason = verdict.reason.value if isinstance(verdict, policy.Denied) else ""
            return f"the read was refused: {reason or 'a restriction denies it'}"
        return verdict.route, risk

    def _held_read(
        self, read: Action, displayed: bool, *, committed: bool = False
    ) -> Expectation:
        """Capture the page, window, and control a read is decided about now."""
        held = self._expect(read) or Expectation(
            PageState(self.surface.location(), page=self._active_page()),
            windows=tuple(sorted(self.windows)) if self.windows else None,
        )
        page = held.page_state
        if not page.page:
            held = dataclasses.replace(
                held, page_state=dataclasses.replace(page, page=self._active_page())
            )
        return dataclasses.replace(
            held, displayed=displayed, displayed_evidence=True, committed=committed
        )

    def _active_page(self) -> str:
        return next((page.page_id for page in self.surface.pages() if page.active), "")

    def _read_moved(
        self, held: Expectation, operation: Operation, read: Action
    ) -> bool:
        """Report whether a read's page, window, or operation changed since ``held``.

        A new observation id is not a change; only where the surface is, which
        window is active, and which control the read resolves to are compared.
        """
        page = held.page_state
        if self.surface.location() != page.location:
            return True
        active = self._active_page()
        if page.page and active and active != page.page:
            return True
        return self._operation(read) != operation

    def _discard_approval(self, step: int) -> RunResult | str:
        """Drop an approval given for a page that is no longer the one shown."""
        self._forget()
        self.notices.append(
            "the page changed while a person approved a read, so the approval "
            "was discarded and the read did not run"
        )
        refreshed = self._refresh(step, ObservationProvenance.POST_INTERVENTION)
        if refreshed is not None:
            return refreshed
        return "the page changed during approval; the read did not run"

    # Hand off when the run cannot supply required record evidence.

    def _can_prove_record(self, action: Action) -> bool:
        """Report whether this run could attach record evidence to ``action``.

        Evidence is an element read through the structured tool. A pixel
        target cannot carry it, and neither can any target in a run whose
        profile does not grant the structured tool.
        """
        if isinstance(action.target, ScreenTarget):
            return False
        return ObservationMode.STRUCTURED in self.profile.perception.allowed_modes

    def _evidence_handoff(self, step: int, action: Action) -> RunResult | None:
        """Ask a person to perform a step the run cannot prove the record of.

        The action is never performed by the run. A person who takes the
        session and hands it back has performed the step, or chosen not to,
        and the run looks again. An approval performs nothing, because an
        approval is not the evidence the operator required.

        The gate asks about evidence last, so everything the profile says
        about this action has already passed. A restriction this run learned
        can still deny the operation, and then nobody is asked to perform it.
        """
        learned = policy.learned_limit(
            self.restrictions,
            self._operation(action),
            frozenset(self.lost),
            frozenset(self.missing),
        )
        if learned is Limit.DENY:
            self.notices.append(
                "a restriction this run holds denies that operation; it was "
                "not handed to a person"
            )
            return None
        handoff = self._intervene(
            step,
            Trigger.RECORD_EVIDENCE_REQUIRED,
            f"{action.kind.value} here must name its record, and this run "
            "cannot supply that evidence; please perform the step",
            action=action,
        )
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)
        if handoff.outcome is HandoffOutcome.RESUMED:
            self._note(handoff)
            self.notices.append(
                "a person had the session for a step that must name its record; "
                "look again to see what they did"
            )
            return self._resume(step)
        if handoff.outcome is HandoffOutcome.APPROVED:
            self._note(handoff)
            self.notices.append(
                "an approval does not perform a step that must name its record; "
                "the run did not perform it"
            )
            return None
        return self._end(
            Ending.HANDED_OFF,
            step,
            f"a person was asked for: {Trigger.RECORD_EVIDENCE_REQUIRED.value}",
        )

    # Store and refresh working-memory facts.

    def _keep(self, step: int, decision: Remember) -> RunResult | None:
        """Keep a fact the model chose, if the screen or the goal shows it.

        A fact with a source must equal what that control shows in the
        current structured observation. One the model marks as unchanging,
        such as an identifier inside a sentence, may instead appear in that
        text exactly once as whole words. A fact with no source must appear in
        the goal. Anything else would be the model's own words kept as if
        they were read, so it is refused. Record evidence must name one
        displayed control that shows its value and is tied to the source the
        way action evidence is, or the fact is not kept at all.
        """
        value = " ".join(decision.value.split())
        node: AxNode | None = None
        if decision.source is not None:
            structured = self.observations.get(ObservationMode.STRUCTURED)
            found = _named(structured, decision.source) if structured else []
            if structured is None or len(found) != 1 or found[0].secret:
                self.notices.append(
                    f"{decision.key} was not kept: its source does not name "
                    "exactly one readable control in the current observation"
                )
                return None
            node = found[0]
            shown = node.value if node.value is not None else node.name
            if not _shows(shown, value, whole=not decision.may_change):
                self.notices.append(
                    f"{decision.key} was not kept: its source shows another value; "
                    "keep exactly what one control on the current screen shows, "
                    "or an unchanging value it holds once as whole words"
                )
                return None
            if decision.record is not None:
                tied = _in_row(structured, node, decision.record)
                decision = dataclasses.replace(decision, record=tied)
                problem = _record_problem(structured, node, tied)
                if problem is not None:
                    self.notices.append(f"{decision.key} was not kept: {problem}")
                    return None
        elif decision.record is not None:
            self.notices.append(
                f"{decision.key} was not kept: record evidence needs the control "
                "the value came from"
            )
            return None
        elif not matching.contains(
            " ".join(self.goal.split()).casefold(), value.casefold()
        ):
            self.notices.append(
                f"{decision.key} was not kept: a fact without a source must "
                "appear in the goal"
            )
            return None
        if len(value) > MAX_FACT_VALUE:
            # A cut value would be a different fact (rule 11).
            self.notices.append(
                f"{decision.key} was not kept: it is longer than "
                f"{MAX_FACT_VALUE} characters; keep a shorter value it holds"
            )
            return None
        partial = False
        if decision.source is not None:
            reading = self._gated_read(
                step, Action(ActionKind.READ, decision.source, evidence=decision.record)
            )
            if isinstance(reading, RunResult):
                return reading
            if (
                isinstance(reading, str)
                or reading.outcome is not Outcome.OK
                or not _shows(
                    reading.extracted or "", value, whole=not decision.may_change
                )
            ):
                self.notices.append(
                    "the remembered value could not be verified by a live read"
                )
                return None
            partial = matching.spaced(reading.extracted or "") != value
            self._note_partial(decision.key, partial)
        self._store(
            Fact(
                key=decision.key,
                value=value,
                step=step,
                route=self.route,
                source=decision.source,
                may_change=decision.may_change and decision.source is not None,
                origin=FactOrigin.GOAL
                if decision.source is None
                else FactOrigin.CONTROL,
                location=self.surface.location(),
                epoch=self.epoch,
                record=decision.record,
                page=self.page_id,
                control=node.control if node is not None else "",
                displayed=node is not None and node.value is None,
                partial=partial,
            )
        )
        self.journal.record(
            Remembered(step=step, from_control=decision.source is not None)
        )
        return None

    def _note_partial(self, key: str, partial: bool) -> None:
        """Explain which surrounding text a reusable partial read needs."""
        if partial:
            self.notices.append(
                f"{key} was kept from inside a longer text; it can be checked "
                "there. Returning or reusing it requires unique surrounding "
                "text made only of website labels and named inputs or facts. "
                "Otherwise keep it from a control that shows it alone"
            )

    def _keep_reading(self, step: int, action: Action, extracted: str | None) -> None:
        """Keep what a read returned, so it outlives the turn that read it.

        The read's record evidence was checked by the surface as it read, so
        it is kept as the fact's record, when its source is a control the
        page displays rather than a field.
        """
        value = matching.spaced(extracted or "")
        if not value or len(value) > MAX_FACT_VALUE:
            # A cut reading would be a different fact, so none is kept.
            return
        source = action.target
        node = self._node_for(source)
        record = action.evidence
        structured = self.observations.get(ObservationMode.STRUCTURED)
        if record is not None:
            sources = _named(structured, record.source) if structured else []
            if len(sources) != 1 or sources[0].value is not None:
                record = None
        self._store(
            Fact(
                key=f"read_at_step_{step}",
                value=value,
                step=step,
                route=self.route,
                source=source if isinstance(source, (AxLocator, DomLocator)) else None,
                origin=FactOrigin.READ,
                location=self.surface.location(),
                epoch=self.epoch,
                record=record,
                page=self.page_id,
                control=node.control if node is not None else "",
                displayed=node is not None and node.value is None,
                painted=isinstance(source, ScreenTarget),
            )
        )

    def _store(self, fact: Fact) -> None:
        self.memory.pop(fact.key, None)
        self.memory[fact.key] = fact
        if self.trace is not None:
            self.trace.kept(fact)
        while len(self.memory) > MAX_FACTS:
            self.memory.pop(next(iter(self.memory)))

    def _fact_here(self, fact: Fact) -> bool:
        """Report whether ``fact`` was kept at this location in this window."""
        if fact.location != self.surface.location():
            return False
        return not fact.page or not self.page_id or fact.page == self.page_id

    def _reread(self, step: int, fact: Fact) -> Fact | str | RunResult:
        """Read a fact's control again, only where it is still the same record.

        The route template is not enough: two members' pages share one. The
        fact must have been kept at this location and in this window. A fact
        kept with record evidence is read with that evidence, and the surface
        refuses the read when the record on the page is another one. A fact
        kept without it is read again only while nothing the run did could
        have changed the page, and only from the same element. Whatever the
        model called the fact, or said about whether it changes, plays no
        part. A new value replaces the old one and keeps its source, record,
        and place. Returns the fact as it now stands, a reason the refresh
        was refused, or a ``RunResult`` when the run must stop.
        """
        if fact.source is None:
            return f"{fact.key} has no control to read again"
        if not self._fact_here(fact):
            return (
                f"{fact.key} was kept on another page, and reading the same "
                "control here could read another record; find it again on its "
                "own record, or ask a person"
            )
        if fact.record is None:
            node = self._node_for(fact.source)
            if fact.epoch != self.epoch or node is None or node.control != fact.control:
                return (
                    f"{fact.key} is not tied to a record and the page has changed "
                    f"since step {fact.step}; keep it again with record evidence, "
                    "or ask a person"
                )
        read = Action(ActionKind.READ, fact.source, evidence=fact.record)
        reading = self._gated_read(step, read)
        if isinstance(reading, RunResult):
            return reading
        if isinstance(reading, str):
            return f"{fact.key} could not be read again: {reading}"
        shown = matching.spaced(reading.extracted or "")
        if reading.outcome is not Outcome.OK or not shown:
            detail = reading.detail or reading.outcome.value
            return f"{fact.key} could not be read again from its record: {detail}"
        if fact.partial:
            # A value kept from inside a longer text keeps its meaning: the
            # text must still hold it once, and the whole text never
            # replaces it (rule 11).
            if not matching.contains(shown, fact.value, once=True):
                return (
                    f"{fact.key} is no longer shown once in its control's text; "
                    "keep it again from a control that shows it"
                )
            value = fact.value
        elif len(shown) > MAX_FACT_VALUE:
            return f"{fact.key} is now longer than a fact may be, so it is not kept"
        else:
            value = shown
        updated = dataclasses.replace(fact, value=value, step=step, epoch=self.epoch)
        self._store(updated)
        return updated

    def _recall(self, step: int, action: Action, key: str) -> Action | RunResult | None:
        """Check a remembered value again before an action uses it.

        A fact the model kept as unchanging is used as kept, the value it
        showed at that step. One that can change is read again from its
        source through ``_reread``, and a refresh it refuses sends the
        proposal back. A different value replaces the fact and the proposal
        is sent back, so the next decision sees it.
        """
        fact = self.memory.get(key)
        if fact is None:
            self.notices.append(f"no fact named {key} is in working memory")
            return None
        if fact.may_change and fact.source is None and fact.origin is FactOrigin.READ:
            if not (self._fact_here(fact) and fact.epoch == self.epoch):
                self.notices.append(
                    f"{key} was read from a screenshot and the page may have "
                    "changed since; read it again before using it"
                )
                return None
        elif fact.may_change and fact.source is not None:
            refreshed = self._reread(step, fact)
            if isinstance(refreshed, RunResult):
                return refreshed
            if isinstance(refreshed, str):
                self.notices.append(refreshed)
                return None
            if refreshed.value != fact.value:
                self.notices.append(
                    f"{key} changed since step {fact.step}; decide again with "
                    "the value now in working memory"
                )
                return None
        return dataclasses.replace(action, value=fact.value)

    # Settle the goal's requirements before the first decision.

    def _interpret(self) -> RunResult | None:
        """Settle the task's requirements from the goal, or stop the run.

        The decider reads the goal alone, before any page is seen. The loop
        checks what it returns: every identifier must appear in the goal and
        every reference must resolve. A refused reading goes back with the
        reason, within the retry allowance. A reading the decider calls
        ambiguous goes to a person, whose approval settles it and whose
        refusal sends it back with their note. The settled task is held for
        the whole run.
        """
        notices: list[str] = []
        for _ in range(self.profile.budgets.max_retries_per_step + 1):
            try:
                task = self.decider.interpret(self.goal, tuple(notices))
            except InvalidDecisionError as error:
                self.journal.record(Corrected(step=0))
                notices.append(f"the task was not usable: {error}")
                continue
            problem = _task_problem(task, self.goal) or _undeclared(task, self.outputs)
            if problem is not None:
                self.journal.record(Corrected(step=0))
                notices.append(f"the task was refused: {problem}")
                continue
            if not task.question:
                self.task = task
                self.journal.record(
                    Interpreted(records=len(task.records), outputs=len(task.outputs))
                )
                return None
            handoff = self._intervene(
                0,
                Trigger.AMBIGUOUS_TASK,
                f"{task.question} The run read the goal as: {_task_text(task)}",
                ask=Ask.APPROVAL,
            )
            spent = self.budget.exhausted(starting_step=False)
            if spent is not None:
                return self._end(Ending.EXHAUSTED, 0, spent.value)
            if handoff.outcome is HandoffOutcome.APPROVED:
                self.task = dataclasses.replace(task, question="")
                self.journal.record(
                    Interpreted(
                        records=len(task.records),
                        outputs=len(task.outputs),
                        confirmed=True,
                    )
                )
                return None
            if handoff.outcome is HandoffOutcome.TIMED_OUT:
                return self._end(
                    Ending.HANDED_OFF,
                    0,
                    f"a person was asked for: {Trigger.AMBIGUOUS_TASK.value}",
                )
            notices.append("a person did not confirm that reading of the goal")
            if handoff.operator_note is not None:
                notices.append(f"operator: {handoff.operator_note}")
        self._intervene(
            0, Trigger.AMBIGUOUS_TASK, "the goal could not be settled as a task"
        )
        return self._end(
            Ending.HANDED_OFF,
            0,
            f"a person was asked for: {Trigger.AMBIGUOUS_TASK.value}",
        )

    # Give new windows to a person.

    def _adopt_windows(self, *, everything: bool) -> None:
        """Take the windows open now as known, and the active one as the run's.

        At the start only the active window is known, so any other already
        open goes to a person. After a person hands the session back, every
        window open then is theirs to have left, and is not handed over again
        for being open.
        """
        pages = self.surface.pages()
        active = next((page.page_id for page in pages if page.active), "")
        if everything:
            self.windows = {page.page_id for page in pages}
            self.handed_dialogs |= {page.dialog for page in pages if page.dialog}
        else:
            self.windows = {active} if active else set()
        self.page_id = active

    def _new_windows(self) -> tuple[PageInfo, ...]:
        """Return the windows a person must see before the run sends more input.

        Those are windows the run does not know, the window that became active
        when the run's own closed, and a known window with a dialog waiting,
        which stops every page of the session. A known window with nothing
        waiting is not returned: it was handed over once already.
        """
        pages = self.surface.pages()
        if not pages:
            return ()
        new = tuple(page for page in pages if page.page_id not in self.windows)
        if new:
            return new
        active = tuple(page for page in pages if page.active)
        if self.page_id and active and active[0].page_id != self.page_id:
            return active
        return tuple(page for page in pages if not page.active and page.dialog)

    def _hand_window(
        self,
        step: int,
        opened: tuple[PageInfo, ...],
        after: tuple[Action, Outcome] | None = None,
    ) -> RunResult | None:
        """Give the session to a person for a window the run does not operate.

        Nothing is switched to and nothing more is sent. The request carries
        every window's id, route, and waiting dialog, which the adapter
        reports from metadata even when no page body can be read. When the
        person hands back, the run forgets what it saw, takes the windows then
        open as known, and looks again at the active page. Handing back is
        not an approval: a risky action still asks.
        """
        stuck = [
            page
            for page in opened
            if page.page_id in self.windows and page.dialog in self.handed_dialogs
        ]
        if stuck and len(stuck) == len(opened):
            return self._end(
                Ending.HANDED_OFF,
                step,
                "a dialog in another window still stops this session",
            )
        pages = self.surface.pages()
        self._forget()
        waiting = [
            f"{page.page_id} has {page.dialog} waiting"
            for page in opened
            if page.dialog
        ]
        ids = ", ".join(page.page_id for page in opened)
        reason = (
            f"{ids} needs a person. The run does not operate new windows; "
            "handle it, close it if it is done, and hand the session back"
        )
        if waiting:
            reason += f"; {', '.join(waiting)}"
        handoff = self._intervene(step, Trigger.NEW_WINDOW, reason, pages=pages)
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)
        if handoff.outcome not in {HandoffOutcome.APPROVED, HandoffOutcome.RESUMED}:
            return self._end(
                Ending.HANDED_OFF,
                step,
                f"a person was asked for: {Trigger.NEW_WINDOW.value}",
            )
        self._note(handoff)
        self.notices.append(_interrupted(ids, after))
        return self._resume(step)

    def _invalid_decision(
        self, step: int, error: InvalidDecisionError
    ) -> RunResult | None:
        self.failures += 1
        self.journal.record(Corrected(step=step, tool=error.tool))
        self.notices.append(
            f"the proposal was invalid: {error}; correct it before acting"
        )
        if self.failures > self.profile.budgets.max_retries_per_step:
            return self._hand_over(
                step, Trigger.NO_PROGRESS, "repeated invalid model proposals"
            )
        return None

    def _transcript(self, location: str) -> Transcript:
        limit = self.profile.perception.max_alternate_observations_per_step
        back = self.control.hand_back
        returned = (
            HumanReturn(
                back.intervention,
                back.segment.taken,
                summary(back.segment.steps),
                tuple(
                    check.requirement or check.output or "state"
                    for check in self.human_verified
                ),
            )
            if back is not None
            else None
        )
        return Transcript(
            goal=self.goal,
            location=location,
            allowed=self.profile.actions,
            permitted_modes=self.profile.perception.allowed_modes,
            observations=tuple(self.observations.values()),
            attempted_modes=tuple(self.attempted),
            history=tuple(self.history),
            notices=tuple(self.notices),
            alternates_remaining=max(0, limit - self.alternates_used),
            steps_remaining=max(0, self.profile.budgets.max_steps - self.budget.steps),
            seconds_remaining=self.budget.remaining_seconds(),
            dialog=self.pending,
            record_bound=policy.record_bound(self.profile, location),
            effect_rules=self.profile.effects,
            restrictions=tuple(self.restrictions),
            secret_names=tuple(self.profile.secrets),
            capabilities=self.surface.capabilities(),
            memory=tuple(self.memory.values()),
            task=self.task,
            human_return=returned,
            inputs=self.inputs,
        )

    def _requested_look(
        self, step: int, request: ObservationRequest, location: str
    ) -> RunResult | None:
        """Take an observation the model asked for, if the profile grants it."""
        alternate = bool(self.attempted)
        looked = self._look(step, request, location, alternate=alternate)
        if isinstance(looked, RunResult):
            return looked
        return None

    def _look(
        self,
        step: int,
        request: ObservationRequest,
        location: str,
        *,
        alternate: bool,
    ) -> Observation | RunResult | None:
        first = self._single_look(step, request, location, alternate=alternate)
        if not isinstance(first, Observation):
            return first
        ended: RunResult | None = None

        def read(part: ObservationRequest) -> Observation | None:
            nonlocal ended
            spent = self.budget.exhausted(starting_step=False)
            if spent is not None:
                ended = self._end(Ending.EXHAUSTED, step, spent.value)
                return None
            result = self._single_look(
                step, part, self.surface.location(), alternate=False
            )
            if isinstance(result, RunResult):
                ended = result
                return None
            return result

        whole = coverage.gather(first, request, read)
        if ended is not None:
            return ended
        if whole.usable:
            self._remember(whole)
        return whole

    def _single_look(
        self,
        step: int,
        request: ObservationRequest,
        location: str,
        *,
        alternate: bool,
    ) -> Observation | RunResult | None:
        """Gate one observation, then take it.

        Returns the observation, a ``RunResult`` when the run must stop, or
        None when the gate refused and the next decision should try something
        else. Nothing reads the page before the verdict.

        When looking is declared risky the gate runs twice. The page can move
        while a person is deciding, and a verdict given for the screen the run
        used to be on is not a verdict for the screen it is on now.
        """
        verdict = self._approve_observation(step, request, location, alternate)
        if isinstance(verdict, RunResult) or verdict is None:
            return verdict

        if request.mode not in self.attempted:
            self.attempted.append(request.mode)
        if alternate:
            self.alternates_used += 1
        observation = self.surface.observe(request)
        self.journal.record(
            Observed(
                step=step,
                mode=observation.mode,
                status=observation.status,
                provenance=request.provenance,
                alternate=alternate,
            )
        )
        self.history.append(
            Turn(step=step, observation=request, status=observation.status)
        )
        opened = self._new_windows()
        if opened:
            return self._hand_window(step, opened)
        self.pending = observation.dialog
        self.pending_at = observation.location
        if observation.usable:
            return observation
        if observation.dialog is not None:
            self.notices.append(
                f"{observation.dialog.dialog_id} is open and the page behind it "
                "cannot be read; accept it, dismiss it, or ask for a person"
            )
        else:
            self.notices.append(
                f"the {request.mode.value} tool returned {observation.status.value}"
                + (
                    "; request an unscoped look to inspect the available controls"
                    if request.scope is not None
                    else ""
                )
            )
        return observation

    def _approve_observation(
        self,
        step: int,
        request: ObservationRequest,
        location: str,
        alternate: bool,
    ) -> RunResult | bool | None:
        """Settle permission to look, and keep it settled until the capture.

        Returns True when the adapter may read the page, None when the gate
        refused and the next decision should try something else, and a
        ``RunResult`` when the run must stop. Nothing between the last check
        here and the capture reads page content.
        """
        if not self._gate_observation(step, request, location, alternate):
            return None
        risk = self.profile.actions[ActionKind.OBSERVE]
        if risk is not Risk.RISKY:
            return True

        cleared = self._approve_looking(step)
        if cleared is not True:
            return cleared
        current = self.surface.location()
        if current != location:
            self._forget()
            self.notices.append("the surface moved during approval; look again")
            return None
        if not self._gate_observation(step, request, current, alternate):
            return None
        return True

    def _gate_observation(
        self,
        step: int,
        request: ObservationRequest,
        location: str,
        alternate: bool,
    ) -> bool:
        """Ask the gate about one observation and record a refusal."""
        verdict = policy.evaluate_observation(
            request,
            self.profile,
            location,
            held=alternate,
            alternates_used=self.alternates_used,
        )
        if isinstance(verdict, policy.Denied):
            self.journal.record(
                Refused(step=step, kind=ActionKind.OBSERVE, reason=verdict.reason)
            )
            self.notices.append(f"policy refused observe: {verdict.reason.value}")
            return False
        return True

    def _approve_looking(self, step: int) -> RunResult | bool | None:
        """Escalate an observation the operator declared risky.

        Returns True when the look may proceed, None when it may not happen
        now, and a ``RunResult`` when the run ends. Only an approval lets it
        proceed. A person who took the session and handed it back has not
        approved anything: the screen they left is a new screen, and looking
        at it is still declared risky, so the next look asks again. Anything
        else ends the run, because an operator who will not approve looking
        at the screen has taken the run.
        """
        handoff = self._intervene(
            step, Trigger.RISKY_ACTION, "observe is declared risky", ask=Ask.APPROVAL
        )
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)
        if handoff.outcome is HandoffOutcome.RESUMED:
            self._reset_progress()
            self._note(handoff)
            self._forget()
            self._handed_back()
            self.notices.append(
                "a person had the session; looking is declared risky, so the "
                "next look asks for approval again"
            )
            return None
        if handoff.outcome is HandoffOutcome.APPROVED:
            self._note(handoff)
            if handoff.changed:
                self._forget()
                self._handed_back()
                self.notices.append(
                    "the session changed while the approval was pending; look again"
                )
                return None
            return True
        return self._end(Ending.HANDED_OFF, step, "operator kept the session")

    def _propose(
        self, step: int, action: Action, location: str, rationale: str = ""
    ) -> RunResult | None:
        """Check one action against the profile, escalate it if risky, run it.

        The operator's verdict comes first and nothing below can loosen it.
        Then the run's own restrictions: any restriction bound to this
        operation, or to this effect on this screen, makes it need a person.
        A proposal the model flags is bound here, before it runs, so the
        flag holds for every later attempt at the same operation whatever
        that attempt is called.
        """
        unfit = self._unfit(action)
        if unfit is not None:
            self.notices.append(unfit)
            return None

        verdict = policy.evaluate(action, self.profile, location)
        risk = self._restricted(step, action, verdict, rationale)
        if (
            isinstance(verdict, policy.Denied)
            and verdict.reason is policy.Denial.RECORD_EVIDENCE_MISSING
            and not self._can_prove_record(action)
        ):
            return self._evidence_handoff(step, action)
        if (
            risk is None
            or not isinstance(verdict, policy.Allowed)
            or not self._selected(step, action)
        ):
            return None
        action_route = verdict.route

        self.journal.record(Decided(step=step, kind=action.kind))
        navigating = action.kind is ActionKind.NAVIGATE
        spent = self.budget.exhausted(starting_step=False, navigating=navigating)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)

        # Only input can repeat a change; a risky read or look changes
        # nothing, so it neither counts as sent nor is refused as a repeat.
        operation = self._operation(action)
        repeat = _change_key(operation) if action.kind not in policy.LOOKS else ""
        already_sent = repeat in self.sent or any(
            (held.business or operation.business) and policy.covers(held, operation)
            for held in self.sent_operations.values()
        )
        if risk is Risk.RISKY and already_sent:
            # The same change was already delivered in this run. Sending it
            # again would make it twice; only a person may decide that.
            self.notices.append(
                "this change was already sent in this run, and sending it again "
                "would repeat it; finish with what it did, or ask for a person"
            )
            self._refuse(step, action, policy.Denial.REPEATED_CHANGE)
            return None
        if risk is Risk.RISKY:
            cleared, proceed = self._approve_acting(step, action, action_route)
            if cleared is not None:
                return cleared
            if not proceed:
                return None

        spent = self.budget.exhausted(starting_step=False, navigating=navigating)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)
        if self.control.halted():
            # The operator stopped or took the run after this was decided.
            # It was not sent, and a decision made before a pause is never
            # sent after one.
            self.control.discard()
            self.journal.record(Superseded(step=step))
            return None
        ended = self._perform(step, action, action_route, navigating)
        if risk is Risk.RISKY and repeat and step in self.performed:
            self.sent.add(repeat)
            self.sent_operations[repeat] = self.performed[step]
        return ended

    def _restricted(
        self, step: int, action: Action, verdict: policy.Verdict, rationale: str
    ) -> Risk | None:
        """Combine the operator's verdict with what this run has learned.

        Returns the risk to act at, or None when the action is refused. The
        operator's verdict can only be made stricter here.
        """
        operation = self._operation(action)
        if isinstance(verdict, policy.Denied):
            if verdict.reason is policy.Denial.EFFECT_DENIED:
                self._bind(step, operation, action.effect, Limit.DENY)
            self._refuse(step, action, verdict.reason)
            return None
        binding_limit = operations.limit(self.profile, operation)
        if binding_limit is not None and operation.business:
            self._bind(step, operation, action.effect, binding_limit)
        if binding_limit is Limit.DENY:
            self._refuse(step, action, policy.Denial.EFFECT_DENIED)
            return None
        learned = policy.learned_limit(
            self.restrictions,
            operation,
            frozenset(self.lost),
            frozenset(self.missing),
        )
        # An unknown input receives the route limit for this proposal. That
        # uncertainty is not new evidence about identified operations.
        if learned is None and (
            binding_limit is Limit.RISKY or self._unplaced(operation)
        ):
            learned = Limit.RISKY
        if learned is Limit.DENY:
            self._refuse(step, action, policy.Denial.EFFECT_DENIED)
            return None
        # A form submission is judged by the observed control, not by the
        # effect the model named it.
        submission = policy.submission_limit(self.profile, operation)
        if submission is Limit.DENY:
            self._refuse(step, action, policy.Denial.SUBMISSION_DENIED)
            return None
        if submission is Limit.RISKY:
            learned = Limit.RISKY
        if self.profile.effect_limit(action.kind, action.effect) is Limit.RISKY:
            self._bind(step, operation, action.effect, Limit.RISKY)
        if action.flag_risky:
            self._bind(
                step,
                operation,
                action.effect,
                Limit.RISKY,
                source=Source.PROPOSAL,
                reason=rationale,
            )
        self.risky_because = _risk_source(
            action,
            verdict.risk,
            flagged=action.flag_risky,
            submission=submission is Limit.RISKY,
            learned=learned is Limit.RISKY,
            effect=self.profile.effect_limit(action.kind, action.effect) is Limit.RISKY,
        )
        if learned is Limit.RISKY or action.flag_risky:
            return Risk.RISKY
        return verdict.risk

    def _unplaced(self, operation: Operation) -> bool:
        """Report whether a restriction here might apply to an unresolved target.

        A locator that did not resolve to exactly one observed control has no
        identity to compare. On a screen with restrictions of the same action
        type, it could be a restricted control named another way, so it goes
        to a person rather than running as unrestricted.
        """
        if operation.resolved:
            return False
        return any(
            held.operation.route == operation.route
            and held.operation.kind is operation.kind
            for held in self.restrictions
        )

    def _refuse(self, step: int, action: Action, denial: policy.Denial) -> None:
        self.journal.record(Refused(step=step, kind=action.kind, reason=denial))
        self.notices.append(f"policy refused {action.kind.value}: {denial.value}")
        self._refused()

    def _refused(self) -> None:
        """Count one refusal toward the no-progress handoff (rule 17)."""
        self.refusals += 1
        self.refused_total += 1

    def _selected(self, step: int, action: Action) -> bool:
        if operations.selections_ready(
            self.profile, self._operation(action), self._node_for(action.target)
        ):
            return True
        self.notices.append(
            "choose an item in every required picker, then observe again; "
            "typing a search query does not select its result"
        )
        self._refuse(step, action, policy.Denial.SELECTION_UNCONFIRMED)
        return False

    def _operation(self, action: Action) -> Operation:
        """Describe what ``action`` would physically do on this screen.

        The identity comes from the observed control the target resolves to,
        so an accessibility locator and a DOM locator for one button describe
        one operation. The model's locator is only a way to find that control.
        """
        node = self._node_for(action.target)
        physical = Operation.of(
            action,
            self.route,
            node=node,
            painted=self._painted(action.target),
        )
        mode = (
            ObservationMode.VISUAL
            if isinstance(action.target, ScreenTarget) and node is None
            else ObservationMode.STRUCTURED
        )
        binding = operation_binding(
            self.surface, action, self.observations.get(mode), node
        )
        return (
            physical
            if binding is None
            else dataclasses.replace(
                physical,
                binding=binding.name,
                business=binding.business,
            )
        )

    def _node_for(self, target: Target | None) -> AxNode | None:
        """Return the one observed control ``target`` names, if there is one.

        A screenshot point names the observed control the surface finds under
        it, so a screenshot click on page content is judged, restricted, and
        checked at input as that control (rule 6). A painted canvas, a point
        on nothing, and keys sent to the focus name none.
        """
        if target is None or isinstance(target, VisualAnchor):
            return None
        structured = self.observations.get(ObservationMode.STRUCTURED)
        if structured is None:
            return None
        if isinstance(target, ScreenTarget):
            if target.point is None:
                return None
            key = (target.capture_id, target.point.x, target.point.y)
            if key not in self.hits:
                self.hits[key] = control_at(self.surface, target)
            control = self.hits[key]
            found = [
                node for node in structured.nodes if control and node.control == control
            ]
            return found[0] if len(found) == 1 else None
        found = _named(structured, target)
        return found[0] if len(found) == 1 else None

    def _painted(self, target: Target | None) -> str:
        """Return a digest of the crop a painted target was cut from."""
        if not isinstance(target, VisualAnchor):
            return ""
        for observation in self.observations.values():
            for region in observation.regions:
                if region.anchor_id == target.anchor_id and region.image:
                    return hashlib.sha256(region.image).hexdigest()[:16]
        return ""

    def _bind(
        self,
        step: int,
        operation: Operation,
        effect: str | None,
        limit: Limit,
        *,
        source: Source = Source.OPERATOR,
        reason: str = "",
    ) -> None:
        """Hold a restriction on ``operation`` for the rest of the run.

        A restriction is only ever added. An identical one already held is
        not added twice, and a weaker one never replaces a stronger one,
        because both stay in the list and the strictest applies.
        """
        restriction = Restriction(operation, effect, limit, source, step, reason)
        held = [
            (item.operation, item.effect, item.limit, item.source)
            for item in self.restrictions
        ]
        if (operation, effect, limit, source) in held:
            return
        self.restrictions.append(restriction)
        self.journal.record(
            Flagged(
                step=self.budget.steps,
                action_step=step,
                kind=operation.kind,
                route=operation.route,
                source=source,
            )
        )

    def _flag(self, finding: RiskFinding) -> None:
        """Bind a restriction to an operation this run already performed.

        The step must name an action that reached the surface in this run.
        Nothing is performed again; a restriction is recorded, and the next
        proposal of that operation, under any label, needs a person.
        """
        operation = self.performed.get(finding.step)
        if operation is None:
            self.notices.append(
                f"step {finding.step} performed no action in this run; "
                "nothing was flagged"
            )
            return
        if finding.step in self.flagged:
            # Keep each finding once. Repeating one does not advance the task.
            self.notices.append(
                f"step {finding.step} is already flagged; nothing more is "
                "needed for it, so go on with the task"
            )
            self._refused()
            return
        self.flagged.add(finding.step)
        self._bind(
            finding.step,
            operation,
            finding.effect,
            Limit.RISKY,
            source=Source.FINDING,
            reason=finding.reason,
        )
        self.notices.append(
            f"the finding is kept: from now on, the operation from step "
            f"{finding.step} asks a person before it runs again. What it already "
            "did is unchanged, and nobody is asked now"
        )

    def _unfit(self, action: Action, *, looked: bool = True) -> str | None:
        """Return why this action cannot be tried against what the run holds.

        A dialog answer needs a dialog in hand, and nothing else may be tried
        while one is: the page behind it does not answer. Everything else
        needs an observation of the page, a painted target from the current
        picture, and record evidence the observation actually showed. The
        run's own reads pass ``looked`` false: a check reads the page itself
        and needs no observation, except to judge record evidence.
        """
        supported = self.surface.capabilities().get(action.kind, frozenset())
        form = target_form(action.target)
        if form not in supported:
            offered = ", ".join(sorted(item.value for item in supported)) or "nothing"
            return (
                f"this surface does not perform {action.kind.value} with a "
                f"{form.value} target; it takes {offered}"
            )
        if action.kind in DIALOG_ACTIONS:
            if self.pending is None:
                return "no dialog is known to be waiting; look before answering one"
            return None
        if self.pending is not None:
            return (
                f"{self.pending.dialog_id} is waiting and the page behind it cannot "
                "be operated; accept it, dismiss it, or ask for a person"
            )
        if looked and not self.observations:
            return "look at the surface before proposing an action"
        if not self._anchor_is_current(action):
            return "that visual anchor is not in the current observation"
        return self._evidence_problem(action)

    def _row_evidence(self, action: Action) -> Action:
        """Scope an action's row evidence to the target's row, when that is unique.

        See ``_in_row``. Anything else is left for the checks to judge.
        """
        structured = self.observations.get(ObservationMode.STRUCTURED)
        if action.evidence is None or structured is None:
            return action
        targets = _named(structured, action.target)
        if len(targets) != 1:
            return action
        return dataclasses.replace(
            action, evidence=_in_row(structured, targets[0], action.evidence)
        )

    def _evidence_problem(self, action: Action) -> str | None:
        """Check record evidence against the structured observation it came from.

        The model chooses the evidence, and the choice is checked here rather
        than believed. The source must name exactly one observed control that
        is not withheld, and that control must show exactly the value
        claimed. For an element target, the relation must hold in the
        observation too: a shared row or smallest shared element that holds
        one control like the target and one value like the source, or an
        explicit reference from the target to the source. A painted target's
        canvas cannot be told apart in the structured observation, so its
        relation is checked by the surface alone.

        None of this proves the record at the moment of acting. The surface
        reads the evidence again for that.
        """
        evidence = action.evidence
        if evidence is None:
            return None
        structured = self.observations.get(ObservationMode.STRUCTURED)
        if structured is None:
            return (
                "record evidence is checked against a structured observation of "
                "this screen; look with the structured tool or ask for a person"
            )
        sources = _named(structured, evidence.source)
        if len(sources) != 1:
            return (
                f"the record evidence names {len(sources)} observed controls, not "
                "one; name the control that shows the identifier by its ref from "
                "the current observation"
            )
        source = sources[0]
        shown = source.value if source.value is not None else source.name
        if source.secret or shown != evidence.displayed:
            return "the record evidence is not a value the observation showed"
        if isinstance(action.target, (VisualAnchor, ScreenTarget)):
            return None
        targets = _named(structured, action.target)
        if len(targets) != 1:
            return (
                f"the target matches {len(targets)} observed controls; name the "
                "specific control by its ref or exact row/container scope. "
                "Record evidence does not disambiguate an unscoped locator"
            )
        return unbound(structured, evidence.relation, targets[0], source)

    def _approve_acting(
        self, step: int, action: Action, route: str
    ) -> tuple[RunResult | None, bool]:
        """Ask a human about a risky action. Returns (finished, may proceed)."""
        handoff = self._intervene(
            step,
            Trigger.RISKY_ACTION,
            self.risky_because,
            action=action,
            route=route,
            ask=Ask.APPROVAL,
        )
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value), False
        match handoff.outcome:
            case HandoffOutcome.APPROVED:
                self._note(handoff)
                return None, self._still_approved(action, handoff)
            case HandoffOutcome.RESUMED:
                self._note(handoff)
                return self._resume(step), False
            case HandoffOutcome.REJECTED:
                self.notices.append(f"the operator declined {action.kind.value}")
                self._note(handoff)
                self._refused()
                if handoff.changed:
                    self._forget()
                    self._handed_back()
                    self.notices.append(
                        "the session changed while the approval was pending; look again"
                    )
                return None, False
            case _:
                # The loader accepts only ABORT for escalation.on_timeout. When
                # it accepts more, decide the run's fate here.
                return self._end(Ending.HANDED_OFF, step, "handoff timed out"), False

    def _still_approved(self, action: Action, handoff: Handoff) -> bool:
        """Report whether an approval still covers ``action`` as proposed.

        An approval is for one proposal on the screen it was made against.
        It lapses if a person used the session while it waited or the page
        moved, and the gate is asked again, because an approval never
        stands in for the operator's policy. The surface still checks the
        target and any record evidence when the action is sent.
        """
        held = self._held_location()
        location = self.surface.location()
        if handoff.changed or (held is not None and location != held):
            self._forget()
            self._handed_back()
            self.notices.append(
                "the session changed while the approval was pending; look again"
            )
            return False
        verdict = policy.evaluate(action, self.profile, location)
        if not isinstance(verdict, policy.Allowed):
            self._refuse(self.budget.steps, action, verdict.reason)
            return False
        return True

    def _trace_action(
        self,
        step: int,
        action: Action,
        result: ActionResult,
        location: str,
        operation: Operation | None = None,
    ) -> None:
        if self.trace is not None:
            self.trace.action(
                step,
                action,
                result,
                location,
                self.observations.get(ObservationMode.STRUCTURED),
                operation,
            )

    def _send(
        self, step: int, action: Action, expect: Expectation | None
    ) -> ActionResult:
        """Keep an uncertain trace event even when the adapter loses the session."""
        state = expect.page_state if expect else PageState(self.surface.location())
        operation = self._operation(action)
        try:
            result = self.surface.act(action, expect=expect)
        except SurfaceError:
            self._trace_action(
                step,
                action,
                ActionResult(Outcome.SURFACE_ERROR, state),
                state.location,
                operation,
            )
            raise
        self._trace_action(step, action, result, state.location, operation)
        return result

    def _perform(
        self, step: int, action: Action, route: str, navigating: bool
    ) -> RunResult | None:
        """Act, record what happened, and look at what it left behind."""
        operation = self._operation(action)
        fingerprint = self._progress_state()
        before = self.surface.location()
        result = self._send(step, action, self._expect(action))
        self._count_change(action, result, before)
        self.performed[step] = operation
        if navigating and result.outcome is Outcome.OK:
            self.budget.charge_navigation()
        source, secret_name = describe_value(action)
        self.journal.record(
            Acted(
                step=step,
                kind=action.kind,
                route=route,
                outcome=result.outcome,
                flagged=action.flag_risky,
                value_source=source,
                secret_name=secret_name,
                target_kind=describe_target(action),
                side_effects=result.side_effects,
            )
        )
        self.history.append(
            Turn(
                step=step,
                action=action,
                outcome=result.outcome,
                extracted=result.extracted,
                detail=result.detail,
                side_effects=result.side_effects,
            )
        )
        if action.kind is ActionKind.READ and result.outcome is Outcome.OK:
            self._keep_reading(step, action, result.extracted)
        opened = self._new_windows()
        if opened:
            return self._hand_window(step, opened, (action, result.outcome))
        self._forget()
        if result.dialog is not None:
            self.pending = result.dialog
            self.pending_at = result.page_state.location
            if action.kind not in DIALOG_ACTIONS:
                self.notices.append(
                    f"{action.kind.value} opened {result.dialog.dialog_id}; what it "
                    "started has not finished and waits on the answer"
                )
        landed = policy.route_for(self.profile, result.page_state.location)
        if landed is None:
            return self._blocked()
        self.route = landed
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)

        if result.outcome in {Outcome.UNCERTAIN, Outcome.HANDOFF}:
            trigger = (
                Trigger.DELIVERY_UNCERTAIN
                if result.outcome is Outcome.UNCERTAIN
                else Trigger.AUTHENTICATION_REQUIRED
            )
            return self._hand_over(step, trigger, result.detail or result.outcome.value)
        if result.outcome is Outcome.OK:
            self._reset_progress()
        else:
            self.notices.append(
                f"{action.kind.value} did not resolve: {_failure_detail(result)}"
            )
            self.failures += 1
        refreshed = self._refresh(step, ObservationProvenance.POST_ACTION)
        if refreshed is not None:
            return refreshed
        repeated = self._cycle(action, operation, fingerprint)
        self._cycle_notice(repeated)
        if self.failures <= self.profile.budgets.max_retries_per_step:
            return None
        reason = "repeated actions did not make progress"
        return self._hand_over(step, Trigger.NO_PROGRESS, reason)

    def _cycle_notice(self, repeated: int) -> None:
        if repeated:
            self.failures = max(self.failures, repeated)
            self.notices.append("the same action returned to an already visited state")

    def _progress_state(self) -> tuple[object, ...]:
        """Describe what an action could change, in memory only.

        A screenshot counts by its digest and a control by whether it is in
        view, so a scroll or a click on a painted screen that changes the
        picture is progress, while a return to an earlier picture is not.
        Plain text is left out: a page may print the time or a request
        number there on every load, and a loop between two pages would then
        never repeat a state. Controls, their values, headings, cells, and
        status messages are what an action changes.
        """
        return (
            self.surface.location(),
            tuple(
                (
                    item.mode,
                    tuple(
                        (node.identity, node.value, node.context, node.in_view)
                        for node in item.nodes
                        if node.role not in _PROSE or node.value is not None
                    ),
                    hashlib.sha256(item.image).hexdigest() if item.image else None,
                )
                for item in self.observations.values()
            ),
            (self.pending.kind, self.pending.message) if self.pending else None,
        )

    def _cycle(
        self, action: Action, operation: Operation, before: tuple[object, ...]
    ) -> int:
        # Screen input shares one operation target, so where it lands and how
        # the mouse moved tell two painted actions apart. The capture id does
        # not, because every new look issues one.
        point = action.target.point if isinstance(action.target, ScreenTarget) else None
        key = (
            action.kind,
            operation.target,
            action.value,
            action.destination,
            point,
            action.mouse,
            before,
            self._progress_state(),
        )
        self.cycles[key] = self.cycles.get(key, 0) + 1
        while len(self.cycles) > 128:
            self.cycles.pop(next(iter(self.cycles)))
        return self.cycles[key] - 1

    def _count_change(self, action: Action, result: ActionResult, before: str) -> None:
        """Advance the epoch when ``action`` may have changed what the page shows."""
        if (
            action.kind not in _UNCHANGING
            or result.page_state.location != before
            or result.dialog is not None
        ):
            self.epoch += 1

    def _ask_human(
        self, step: int, trigger: Trigger, detail: str, location: str
    ) -> RunResult | None:
        """Hand the session to a person, because the model asked for one.

        The request is honoured as it stands, even when only one tool has run
        against this screen. The model is the one that can tell whether
        another look would help: it may spend its alternate allowance by
        asking for the other tool, and if it asks for a person instead, then
        running a tool first would be this loop overruling the decision it
        just asked for, and billing the run for the privilege.

        What a person cannot be asked to settle is policy. An action the
        operator declared risky still escalates on its own, whatever the model
        proposes.
        """
        del location
        self.journal.record(Decided(step=step, kind=None))
        return self._hand_over(step, trigger, detail)

    def _hand_over(self, step: int, trigger: Trigger, reason: str) -> RunResult | None:
        """Offer the live session to a person and act on their answer."""
        handoff = self._intervene(step, trigger, reason)
        spent = self.budget.exhausted(starting_step=False)
        if spent is not None:
            return self._end(Ending.EXHAUSTED, step, spent.value)
        if handoff.outcome not in {HandoffOutcome.APPROVED, HandoffOutcome.RESUMED}:
            # The reason went to the operator channel live. The ending names
            # the trigger instead, because the reason may quote the screen.
            return self._end(
                Ending.HANDED_OFF, step, f"a person was asked for: {trigger.value}"
            )
        self._note(handoff)
        return self._resume(step)

    def _resume(self, step: int) -> RunResult | None:
        """Take the session back: forget the old screen and look again."""
        # A person may have changed anything while they held the session,
        # including opening or closing windows, which are theirs to leave.
        self.epoch += 1
        self._forget()
        self._reset_progress()
        self.cycles.clear()
        self._adopt_windows(everything=True)
        self._handed_back()
        location = self.surface.location()
        if policy.route_for(self.profile, location) is None:
            return self._blocked()
        ended = self._refresh(step, ObservationProvenance.POST_INTERVENTION)
        if ended is not None or not self.human_checks:
            return ended
        # The checks belong to the request this hand-back answers. A later,
        # unrelated hand-back must not read them again.
        checks, self.human_checks = self.human_checks, ()
        ended = self._after(step, checks)
        self.human_verified = self.checked_after
        return ended

    def _refresh(
        self, step: int, provenance: ObservationProvenance
    ) -> RunResult | None:
        """Take one automatic observation, gated like any other.

        Exactly one tool runs, the first the operator declared, and the run
        never reaches for the other one on its own afterwards. An automatic
        look that walked down the list would spend the model's alternate
        allowance on a decision the model never made, and would put a
        screenshot in front of it that nobody asked for.

        A tool that cannot read the screen leaves the run with no observation
        in hand. That is reported as a notice, the next decision sees it, and
        the model asks for whichever tool it wants.
        """
        location = self.surface.location()
        mode = self.profile.perception.allowed_modes[0]
        request = ObservationRequest(mode=mode, provenance=provenance)
        looked = self._look(step, request, location, alternate=False)
        if isinstance(looked, RunResult):
            return looked
        del looked
        if self.pending is None and not self.observations:
            self.notices.append("no observation of this screen is in hand; look again")
        return None

    def _intervene(
        self,
        step: int,
        trigger: Trigger,
        reason: str,
        *,
        action: Action | None = None,
        route: str | None = None,
        pages: tuple[PageInfo, ...] | None = None,
        outputs: Mapping[str, str] = _NO_OUTPUTS,
        unverified: tuple[str, ...] = (),
        ask: Ask = Ask.PERSON,
    ) -> Handoff:
        """Ask a person through the run's control, which stops the clock.

        An approval request offers approve once, reject, and a takeover. A
        request for help offers a takeover and a resume, never an approval,
        because approving would not supply what the run is missing.
        """
        request = InterventionRequest(
            trigger=trigger,
            goal=self.goal,
            profile_id=self.profile.profile_id,
            step=step,
            route=route if route is not None else self.route,
            reason=reason,
            timeout_s=self.profile.escalation.handoff_timeout_s,
            action=action,
            observed_modes=tuple(self.attempted),
            pages=self.surface.pages() if pages is None else pages,
            outputs=outputs,
            unverified=unverified,
            ask=ask,
            mode=Mode.DISCOVERY,
        )
        segment_count = len(self.control.segments)
        handoff = self.control.intervene(request)
        segments = self.control.segments[segment_count:]
        if self.trace is not None and (
            trigger is Trigger.DELIVERY_UNCERTAIN or _acted(handoff, segments)
        ):
            # A note with nothing done is advice, not a step of the workflow.
            # Whether an uncertain step went through is always the person's
            # answer, so that one is kept even when they changed nothing.
            self.trace.handoff(request, handoff, segments)
            self._label_person(handoff, segments)
        self.journal.record(
            Escalated(
                step=step,
                trigger=trigger,
                outcome=handoff.outcome,
                intervention=handoff.intervention,
            )
        )
        if handoff.outcome is HandoffOutcome.TERMINATED:
            raise _Terminated
        return handoff

    def _expect(self, action: Action) -> Expectation | None:
        """Describe what must still hold when this action reaches the surface.

        The page the decision was made against, and the context path the
        control sat under when the run saw it. For a dialog answer, the dialog
        the decision was about. The surface checks these immediately before it
        acts, which is the only moment where checking means anything:
        everything this loop knows is already one round trip old.
        """
        windows = tuple(sorted(self.windows)) if self.windows else None
        if action.kind in DIALOG_ACTIONS:
            if self.pending is None:
                return None
            return Expectation(
                page_state=PageState(self.pending_at),
                dialog=self.pending.dialog_id,
                windows=windows,
            )
        page = self._page()
        if page is None:
            return None
        node = self._node_for(action.target)
        operation = self._operation(action)
        return Expectation(
            page_state=page,
            context=self._context(action.target),
            control=node.control if node is not None else "",
            strict=any(
                held.operation.route == self.route for held in self.restrictions
            ),
            windows=windows,
            submits_as=node.submits_as if node is not None else None,
            enter_as=node.enter_as if node is not None else None,
            binding=operation.binding,
            business=operation.business,
            observed_control=node,
        )

    def _context(self, target: Target | None) -> tuple[str, ...]:
        """Return the context path of the one observed control ``target`` names."""
        if target is None or isinstance(target, (VisualAnchor, ScreenTarget)):
            return ()
        found = [
            node
            for observation in self.observations.values()
            for node in observation.nodes
            if names_node(target, node)
        ]
        if len(found) != 1:
            return ()
        return found[0].context

    def _anchor_is_current(self, action: Action) -> bool:
        """Report whether a visual target names a region this run just saw."""
        if isinstance(action.target, ScreenTarget):
            return any(
                item.mode is ObservationMode.VISUAL
                and item.observation_id == action.target.capture_id
                for item in self.observations.values()
            )
        if not isinstance(action.target, VisualAnchor):
            return True
        anchors = {
            region.anchor_id
            for observation in self.observations.values()
            for region in observation.regions
        }
        return action.target.anchor_id in anchors

    def _remember(self, observation: Observation) -> None:
        if self.trace is not None:
            self.trace.look(observation)
        held = self._page()
        if held is not None and held.differs_from(observation.page_state):
            self.observations.clear()
        self.observations[observation.mode] = observation
        if observation.mode is ObservationMode.STRUCTURED:
            self._account(observation)

    def _account(self, observation: Observation) -> None:
        """Note restricted controls that left this screen with no successor.

        A restricted control still present, by its id or by the same name, is
        accounted for. One that is gone with nothing of its name in its place
        may have been drawn again under another name, so it is marked lost,
        and a control of its kind that appears after it is treated as it.

        A restricted submission is accounted for when some control on this
        screen performs a submission with the same description, whatever its
        element ids now are. That is how a restriction reconnects after a
        reload or a redraw, without anyone clicking the button again. One that
        no control performs any more is marked missing, and until it is seen
        again every submission in its frame is treated as it.
        """
        controls = {node.control for node in observation.nodes}
        names = {describe(node) for node in observation.nodes}
        shown = {
            description
            for node in observation.nodes
            for description in (node.submits_as, node.enter_as)
            if description
        }
        for held in self.restrictions:
            operation = held.operation
            if operation.route != self.route:
                continue
            if operation.submission_as in shown:
                self.missing.discard(operation.submission_as)
            elif operation.submission_as:
                self.missing.add(operation.submission_as)
            if not operation.control:
                continue
            if operation.control in controls:
                self.lost.discard(operation.control)
            elif operation.target not in names:
                self.lost.add(operation.control)

    def _page(self) -> PageState | None:
        for observation in self.observations.values():
            if observation.page_state.signature:
                return observation.page_state
        for observation in self.observations.values():
            return observation.page_state
        return None

    def _held_location(self) -> str | None:
        page = self._page()
        if page is not None:
            return page.location
        return self.pending_at if self.pending is not None else None

    def _forget(self) -> None:
        self.observations.clear()
        self.pending = None

    def _reset_progress(self) -> None:
        """Clear what only an unresolved action was accumulating.

        Two boundaries reset this and nothing else does: an action that
        resolved, and a person handing the session back. A new model
        response, a new request, a new observation id, and an observation that
        came back unusable all end nothing, because none of them are progress.
        Resetting on a failed look is what would let a run take one tool
        forever.
        """
        self.failures = 0
        self.refusals = 0
        self.alternates_used = 0
        self.attempted.clear()

    def _note(self, handoff: Handoff) -> None:
        if handoff.operator_note is not None:
            self.notices.append(f"operator: {handoff.operator_note}")

    def _hold(self) -> RunResult | None:
        """Wait at a boundary while the operator holds the run."""
        count = len(self.control.segments)
        request = InterventionRequest(
            Trigger.NO_PROGRESS,
            self.goal,
            self.profile.profile_id,
            self.budget.steps,
            self.route,
            "the operator paused the run",
            self.profile.escalation.handoff_timeout_s,
        )
        handoff = self.control.hold()
        segments = self.control.segments[count:]
        # A pause where nobody acted and the page did not move is no step of
        # the workflow. Recorded, it would make every replay stop for a person.
        if self.trace is not None and _acted(handoff, segments):
            self.trace.handoff(request, handoff, segments)
            self._label_person(handoff, segments)
        if handoff.outcome is HandoffOutcome.TERMINATED:
            return self._terminated()
        if handoff.outcome is not HandoffOutcome.RESUMED:
            return self._end(Ending.HANDED_OFF, self.budget.steps, "handoff timed out")
        self._note(handoff)
        return self._resume(self.budget.steps)

    def _handed_back(self) -> None:
        """Tell the next decision what a person did and what was interrupted.

        Facts read from a control may no longer hold after someone else used
        the session, so each one is read again before a proposal uses it.
        """
        back = self.control.hand_back
        if back is None:
            return
        self._unsettle()
        done = summary(back.segment.steps)
        if done:
            self.notices.append(
                f"while {back.intervention} was open a person: " + "; ".join(done)
            )
        interrupted = explain(back.interrupted, back.dispatched)
        if interrupted:
            self.notices.append(
                f"{interrupted}; decide again from the screen as it is now"
            )
        self.notices.extend(back.notices)

    def _unsettle(self) -> None:
        """Treat what the run read earlier as something a person may have changed."""
        self.epoch += 1
        for key, fact in list(self.memory.items()):
            if fact.source is not None and not fact.may_change:
                self.memory[key] = dataclasses.replace(fact, may_change=True)

    def _terminated(self) -> RunResult:
        """End the run for the operator, saying what was in flight."""
        return self._end(Ending.TERMINATED, self.budget.steps, self._terminal_detail())

    def _terminal_detail(self) -> str:
        """Describe any in-flight operation left by termination."""
        detail = "the operator terminated the run"
        dispatched = self.control.interrupted()
        if dispatched is not None:
            detail += (
                f"; the {_what(dispatched.kind)} sent at step {dispatched.step} "
                f"came back {_outcome(dispatched.outcome)} and nothing was undone"
            )
        return detail

    def _blocked(self) -> RunResult:
        return self._end(
            Ending.BLOCKED, self.budget.steps, "surface left the permitted routes"
        )

    def _fail(self, stage: str, detail: str) -> RunResult:
        self.journal.record(Failed(step=self.budget.steps, stage=stage))
        return self._end(Ending.FAILED, self.budget.steps, detail)

    def _end(
        self,
        ending: Ending,
        steps: int,
        detail: str,
        outputs: Mapping[str, str] = _NO_OUTPUTS,
        *,
        verification: Verification | None = None,
        checks: tuple[CheckResult, ...] = (),
        outcome: str = "",
    ) -> RunResult:
        if ending is not Ending.COMPLETED and self.control.terminated():
            # A termination arrived while an operation was settling, and
            # whatever that operation led to is not why the run stopped.
            ending, detail = Ending.TERMINATED, self._terminal_detail()
        self.control.finish(ending.value)
        if ending is not Ending.COMPLETED:
            self.journal.record(
                FailureEvidence(
                    steps,
                    "",
                    ending.value,
                    ("goal_coverage",),
                    snapshot(self.observations.get(ObservationMode.STRUCTURED)),
                )
            )
        self.journal.record(RunEnded(ending=ending.value, steps=steps, detail=detail))
        return RunResult(
            ending=ending,
            steps=steps,
            detail=detail,
            outputs=outputs,
            restrictions=tuple(self.restrictions),
            verification=verification,
            checks=checks,
            task=self.task,
            segments=self.control.segments,
            outcome=outcome,
        )


def _unsupported(
    claim: Finish, goal: str, identifiers: tuple[str, ...] = ()
) -> str | None:
    """Return why a finish claim cannot be checked as written, or None.

    Every output needs a result check expecting exactly its value. A record
    check must expect a value the goal names, because it is there to show
    the page is about the record the goal asked for.

    Examples
    --------
    >>> from computeruse.decider import ResultCheck
    >>> cell = AxLocator("cell", "$4.00")
    >>> _unsupported(Finish({"balance": "$4.00"}), "read it")
    'the claim carries no checks'
    >>> check = ResultCheck(CheckKind.RESULT, cell, "$4.00", output="balance")
    >>> _unsupported(Finish({"balance": "$4.00"}, checks=(check,)), "read it") is None
    True
    """
    if not claim.checks:
        return "the claim carries no checks"
    for check in claim.checks:
        if check.kind is CheckKind.RESULT:
            if check.output not in claim.outputs:
                return f"a result check names {check.output!r}, which is not an output"
            if check.expected != claim.outputs[check.output]:
                return f"the result check for {check.output!r} expects another value"
        elif check.kind is CheckKind.RECORD and not _in_goal(check.expected, goal):
            return (
                f"a record check expects {check.expected!r}, which the goal does not "
                f"name; expect the identifier itself{_naming(identifiers)} on a "
                "control that shows it"
            )
        if check.record is not None and not _in_goal(check.record.value, goal):
            return (
                f"a result check's record evidence value {check.record.value!r} is "
                f"a value the goal does not name; give the identifier"
                f"{_naming(identifiers)} as the value, and any other text the "
                "control shows before or after it as prefix or suffix"
            )
    supported = {c.output for c in claim.checks if c.kind is CheckKind.RESULT}
    missing = [name for name in claim.outputs if name not in supported]
    if missing:
        named = ", ".join(repr(name) for name in missing)
        return (
            f"no result check supports {named}; report only outputs a control "
            "on this page or a fact you kept can show"
        )
    return None


def _naming(identifiers: tuple[str, ...]) -> str:
    named = ", ".join(repr(value) for value in identifiers if value)
    return f" ({named})" if named else ""


def _judged_label(
    profile: Profile, event: ManualEvent, label: str | None, learned: Limit | None
) -> bool | None:
    """Judge a person's labelled click as the gate judges the run's own.

    Returns whether it needs an approval on every run, or None when it may
    not be automated: no valid label, a gate refusal such as a denied effect,
    or a learned deny.

    Examples
    --------
    >>> from pathlib import Path
    >>> from computeruse.manual import ManualKind
    >>> from computeruse.profile import load_profile
    >>> profile = load_profile(Path("examples/profile.yaml"))
    >>> click = ManualEvent(1, 1.0, ManualKind.CLICK, True, route="/members",
    ...     target=AxLocator("button", "Go"), location=profile.scope.base_url
    ...     + "/members")
    >>> _judged_label(profile, click, "open_member", None)
    False
    >>> _judged_label(profile, click, "submit_payment", None)
    True
    >>> _judged_label(profile, click, "close_account", None) is None
    True
    >>> _judged_label(profile, click, "open_member", Limit.DENY) is None
    True
    """
    if label is None or not is_effect_name(label) or event.target is None:
        return None
    action = Action(ActionKind.CLICK, event.target, effect=label)
    verdict = policy.evaluate(action, profile, event.location)
    if not isinstance(verdict, policy.Allowed) or learned is Limit.DENY:
        return None
    # A form submission is judged by the control, whatever the label says,
    # for a person's click as for the run's own (rule 6).
    submission = (
        policy.submission_limit(profile, event.operation)
        if event.operation is not None
        else None
    )
    if submission is Limit.DENY:
        return None
    return (
        verdict.risk is Risk.RISKY
        or learned is Limit.RISKY
        or submission is Limit.RISKY
        or profile.effect_limit(ActionKind.CLICK, label) is Limit.RISKY
    )


def _acted(handoff: Handoff, segments: Sequence[ManualSegment]) -> bool:
    """Report whether a person did anything while they held the run.

    A person acted when they used the session, when the page moved while
    they held it, or when the session could not record what they did, which
    a gap says. Only then is their time a step a capability must keep.
    """
    return handoff.changed or any(item.steps or item.gaps for item in segments)


def _split_identifier(
    evidence: RecordEvidence, identifiers: tuple[str, ...]
) -> RecordEvidence:
    """Keep the goal's identifier as the value and the rest of the text around it.

    A control can show an identifier with other text, such as
    "12345 · Jordan Smith". Evidence that quotes the whole text is split here
    when the text holds exactly one of the task's identifiers as a whole word,
    so the claim says what the control shows and which record it names. The
    prefix and suffix must leave the identifier whole, as ``separates`` says.

    Examples
    --------
    >>> from computeruse.actions import Relation
    >>> source = AxLocator("cell", "12345 · Jordan Smith")
    >>> split = _split_identifier(
    ...     RecordEvidence(source, "12345 · Jordan Smith", Relation.ROW), ("12345",))
    >>> split.value, split.prefix, split.suffix
    ('12345', '', ' · Jordan Smith')
    >>> _split_identifier(
    ...     RecordEvidence(source, "123456", Relation.ROW), ("12345",)).value
    '123456'
    """
    text = evidence.value
    if text in identifiers:
        return evidence
    found = [
        (value, at)
        for value in identifiers
        if value
        for at in word_positions(text, value)
    ]
    if len(found) != 1:
        return evidence
    value, at = found[0]
    prefix = evidence.prefix + text[:at]
    suffix = text[at + len(value) :] + evidence.suffix
    if not separates(prefix, suffix):
        return evidence
    return dataclasses.replace(evidence, value=value, prefix=prefix, suffix=suffix)


def _with_identifiers(decision: Decision, identifiers: tuple[str, ...]) -> Decision:
    """Split composite record evidence in one decision, before anything checks it."""
    if not identifiers:
        return decision
    match decision:
        case Propose(action=action) if action.evidence is not None:
            evidence = _split_identifier(action.evidence, identifiers)
            return dataclasses.replace(
                decision, action=dataclasses.replace(action, evidence=evidence)
            )
        case Remember(record=record) if record is not None:
            return dataclasses.replace(
                decision, record=_split_identifier(record, identifiers)
            )
        case Finish(checks=checks):
            return dataclasses.replace(
                decision,
                checks=tuple(
                    check
                    if check.record is None
                    else dataclasses.replace(
                        check, record=_split_identifier(check.record, identifiers)
                    )
                    for check in checks
                ),
            )
    return decision


def _compared(
    check: ResultCheck,
    seen: str,
    note: str = "",
    bound: str = "",
    *,
    painted: bool = False,
) -> CheckResult:
    """Compare what a check's target shows with what the check expects.

    A line read from a screenshot compares as painted text, whether the
    check points at it or cites a fact kept from it.
    """
    shown = matching.spaced(seen)
    expected = matching.spaced(check.expected)
    if painted or isinstance(check.target, ScreenTarget):
        return _painted_compared(check, shown, expected, note, bound)
    # A record check shows which record the page is about, so ``matching``
    # finds its identifier as whole words, as in "12345 · Jordan Smith",
    # never as part of a longer identifier. A requirement fails when the
    # control also denies it.
    passed = matching.shows(shown, expected, check.match, purpose=check.kind)
    detail = note if passed else f"but the page shows {shown[:160]!r}"
    return CheckResult(check, shown, passed, detail, bound if passed else "")


def _painted_compared(
    check: ResultCheck, shown: str, expected: str, note: str, bound: str
) -> CheckResult:
    """Compare a line read from a screenshot with what the check expects.

    Recognition can drop a space or change a letter's case, so the value is
    found as whole words with spacing and case ignored, bounded by anything
    that is not a letter or digit. A record or a required state must be shown
    beside a label: a line that is only the value may be a painted field the
    run typed into, which says nothing about what the application holds.

    Examples
    --------
    >>> from computeruse.actions import Point
    >>> from computeruse.decider import Match
    >>> screen = ScreenTarget("c1", Point(1, 1))
    >>> status = ResultCheck(
    ...     CheckKind.RESULT, screen, "active", Match.CONTAINS, output="status"
    ... )
    >>> _painted_compared(status, "Status:active", "active", "", "").passed
    True
    >>> exact = dataclasses.replace(status, match=Match.EQUALS)
    >>> _painted_compared(exact, "Status: active", "active", "", "").passed
    True
    >>> _painted_compared(exact, "Status: inactive", "active", "", "").passed
    False
    >>> member = ResultCheck(CheckKind.RECORD, screen, "NM5", Match.CONTAINS)
    >>> _painted_compared(member, "NM5", "NM5", "", "").passed
    False
    >>> signed = ResultCheck(
    ...     CheckKind.REQUIREMENT, screen, "OP2", Match.CONTAINS, requirement="operator"
    ... )
    >>> _painted_compared(signed, "Signed on: OP2 Sam Lee", "OP2", "", "").passed
    True
    """
    joined, wanted = (
        "".join(shown.split()).casefold(),
        "".join(expected.split()).casefold(),
    )
    # ``matching`` reads painted text without case, with a colon as a word
    # break, and finds a value with spaces also with its spaces dropped. A
    # value without spaces is never found inside a longer word.
    passed = matching.shows(
        shown, expected, check.match, purpose=check.kind, painted=True
    )
    labelled = joined != wanted
    if check.kind in {CheckKind.RECORD, CheckKind.REQUIREMENT} and not labelled:
        return CheckResult(
            check,
            shown,
            False,
            "a line that is only the value may be a field; point at a line that "
            "labels it",
        )
    detail = note if passed else f"but the screen shows {shown[:160]!r}"
    return CheckResult(check, shown, passed, detail, bound if passed else "")


def _displayed_only(check: ResultCheck) -> bool:
    """Report whether a fact ``check`` cites must have come from displayed text.

    A kept fact cannot say whether anybody changed its field since, so a
    record or required state never rests on a fact kept from a field.
    """
    return check.kind in {CheckKind.RECORD, CheckKind.REQUIREMENT}


def _outcome_gaps(
    task: Task,
    outcome: str,
    results: tuple[CheckResult, ...],
    observation: Observation | None,
) -> list[str]:
    """Return what an outcome claim's passing checks leave unshown.

    Examples
    --------
    >>> from computeruse.decider import TaskRecord
    >>> task = Task((TaskRecord("member", "NM9"),))
    >>> state = ResultCheck(CheckKind.STATE, AxLocator("status", "0 found"), "0 found")
    >>> searched = ResultCheck(CheckKind.RECORD, AxLocator("searchbox", "Q"), "NM9")
    >>> missing = "record_not_found"
    >>> _outcome_gaps(task, missing, (CheckResult(state, "0 found", True),), None)
    ['member NM9 is not shown as searched for']
    >>> both = (CheckResult(state, "0 found", True), CheckResult(searched, "NM9", True))
    >>> _outcome_gaps(task, missing, both, None)
    []
    """
    passing = [item for item in results if item.passed]
    # The application refused the change, so the states it would have
    # reached are not shown. Who did the work, and where, still must be.
    context = tuple(item for item in task.requirements if item.context)
    gaps = _uncovered(Task(requirements=context), results)
    if not any(item.check.kind is CheckKind.STATE for item in passing):
        gaps.append("no state check shows the outcome")
    for record in task.records:
        if not any(
            item.check.kind is CheckKind.RECORD and item.check.expected == record.value
            for item in passing
        ):
            shown = f"{record.name} {record.value} is not shown as searched for"
            if any(isinstance(item.check.target, ScreenTarget) for item in results):
                # A painted field cannot prove this claim, so report the missing proof.
                shown += (
                    "; on a screenshot, point the record check at a line that "
                    "holds it among other words, such as the application's answer"
                )
            gaps.append(shown)
        if outcome != RECORD_NOT_FOUND:
            continue
        rows = [
            node
            for node in (observation.nodes if observation is not None else ())
            if node.row is not None
            and not node.secret
            and node.value is None
            and word_positions(node.name, record.value)
        ]
        if rows:
            gaps.append(f"a result row shows {record.name} {record.value}")
    return gaps


def _failure_detail(result: ActionResult) -> str:
    """Keep the adapter's explanation in live feedback, never in journal events."""
    return (
        f"{result.outcome.value} ({result.detail})"
        if result.detail
        else result.outcome.value
    )


def _risk_source(
    action: Action,
    declared: Risk,
    *,
    flagged: bool,
    submission: bool,
    learned: bool,
    effect: bool,
) -> str:
    """Say why an action needs approval, for the person asked to give it.

    Examples
    --------
    >>> from computeruse.actions import AxLocator
    >>> read = Action(ActionKind.READ, AxLocator("cell", "Balance"))
    >>> _risk_source(read, Risk.SAFE, flagged=False, submission=False,
    ...     learned=True, effect=False)
    'a restriction learned earlier in this run covers this read'
    """
    kind = action.kind.value
    if flagged:
        return f"the model flagged this {kind} as risky"
    if submission:
        return f"this {kind} submits a form, and the profile marks submissions risky"
    if effect:
        return f"the profile declares the effect {action.effect} risky"
    if learned:
        return f"a restriction learned earlier in this run covers this {kind}"
    if declared is Risk.RISKY:
        return f"{kind} is declared risky"
    return f"this {kind} needs a person's approval"


def _change_key(operation: Operation) -> str:
    """Name a change by its screen and control, so a new flow keeps the name.

    A form's own description can hold a token the application issues per
    flow, so the same Commit in a second flow would look like another form.
    The screen's route template and the resolved control's frame, tag, role,
    and name stay the same. An unresolved target has no name, and so no guard.

    Examples
    --------
    >>> from computeruse.actions import Action, AxLocator, AxNode
    >>> commit = AxNode("button", "Commit", tag="button", control="d1:c4")
    >>> click = Action(ActionKind.CLICK, AxLocator("button", "Commit"))
    >>> _change_key(Operation.of(click, "/orders/:id/confirm", node=commit))
    '/orders/:id/confirm||button|button|Commit'
    """
    if operation.business:
        return operation.business
    return f"{operation.route}|{operation.target}" if operation.resolved else ""


def _shows(shown: str, value: str, *, whole: bool) -> bool:
    """Report whether a control's text is ``value``, or holds it once when allowed.

    Examples
    --------
    >>> text = "Order O-2001 placed for C-1001."
    >>> _shows(text, "O-2001", whole=True), _shows(text, "O-2001", whole=False)
    (True, False)
    >>> _shows("O-2001 O-2001", "O-2001", whole=True)
    False
    """
    shown = matching.spaced(shown)
    if shown == value:
        return True
    return whole and matching.contains(shown, value, once=True)


_PROSE = frozenset({"generic", "paragraph", "text"})
"""Roles of plain text, which the cycle check leaves out of a page's state."""


def _painted_ties(
    task: Task, results: tuple[CheckResult, ...]
) -> tuple[CheckResult, ...]:
    """Tie the painted readings of one screen to the record it names once.

    A painted screen has no row or container, so the only proof of which
    record it is about is a line such as "Member number: NM5": a label that
    ends with a colon, then exactly a record's identifier, and no other line
    of the same picture starting with that label. Two such lines are a list,
    and a list is not one record. A passing record check on that line ties
    every other passing painted reading of the same capture to the record.
    Nothing is tied when the picture names two records this way.

    Examples
    --------
    >>> from computeruse.actions import Point
    >>> from computeruse.decider import Match, TaskOutput, TaskRecord
    >>> def at(y):
    ...     return ScreenTarget("c1", Point(10, y))
    >>> screen = ("Member number: NM5", "Status: active")
    >>> member = ResultCheck(CheckKind.RECORD, at(1), "NM5", Match.CONTAINS)
    >>> status = ResultCheck(CheckKind.RESULT, at(2), "active", output="status")
    >>> task = Task((TaskRecord("member", "NM5"),), (TaskOutput("status", "member"),))
    >>> read = (CheckResult(member, screen[0], True, screen=screen),
    ...         CheckResult(status, screen[1], True, screen=screen))
    >>> [item.bound for item in _painted_ties(task, read)]
    ['NM5', 'NM5']
    >>> listed = ("Member number: NM5", "Member number: NM6", "Status: active")
    >>> read = tuple(dataclasses.replace(item, screen=listed) for item in read)
    >>> [item.bound for item in _painted_ties(task, read)]
    ['', '']
    """
    named: dict[str, set[str]] = {}
    for item in results:
        check = item.check
        if not (
            item.passed
            and item.shown
            and check.kind is CheckKind.RECORD
            and isinstance(check.target, ScreenTarget)
        ):
            continue
        for record in task.records:
            label = matching.label_before(item.shown, record.value)
            if (
                matching.same_identifier(check.expected, record.value)
                and label is not None
                and label.endswith(":")
                and sum(_starts(line, label) for line in item.screen) == 1
            ):
                named.setdefault(check.target.capture_id, set()).add(record.value)
    tied: list[CheckResult] = []
    for item in results:
        target = item.check.target
        values = (
            named.get(target.capture_id, set())
            if isinstance(target, ScreenTarget)
            else set()
        )
        if item.passed and not item.bound and len(values) == 1:
            item = dataclasses.replace(item, bound=next(iter(values)))
        tied.append(item)
    return tuple(tied)


def _starts(line: str, label: str) -> bool:
    """Report whether a painted line starts with ``label``, spacing and case aside."""
    return (
        "".join(line.split()).casefold().startswith("".join(label.split()).casefold())
    )


def _uncovered(task: Task, results: tuple[CheckResult, ...]) -> list[str]:
    """Return each requirement of ``task`` the passing checks do not cover.

    A record belonging to no other is covered by a passing record check that
    expects its identifier, or by a passing reading tied to it. A record
    within another is covered only by a passing record check expecting its
    identifier whose record evidence is the other record's identifier, which
    is the page tying the two together rather than showing both somewhere.
    An output of a record is covered only by a passing result check for that
    output tied to that record. Two records that share a value, such as a
    branch, are told apart here by the record each reading was tied to, never
    by the value. A context requirement, such as who is signed on, is
    covered only by a passing check like any other: a goal naming the
    operator is not proof the application is signed on as that operator.

    Examples
    --------
    >>> from computeruse.decider import TaskOutput, TaskRecord
    >>> cell = AxLocator("cell", "$4.00")
    >>> balance = ResultCheck(CheckKind.RESULT, cell, "$4.00", output="balance")
    >>> member, owned = TaskRecord("member", "12345"), TaskOutput("balance", "member")
    >>> task = Task((member,), (owned,))
    >>> _uncovered(task, (CheckResult(balance, "$4.00", True),))
    ['balance is not tied to member 12345', 'member 12345 is not shown']
    >>> _uncovered(task, (CheckResult(balance, "$4.00", True, bound="12345"),))
    []
    """
    passing = [item for item in results if item.passed]
    records = {record.name: record for record in task.records}
    gaps = [
        f"required state {required.name} is not established"
        for required in uncovered_requirements(
            task, ((item.check, item.bound) for item in passing)
        )
    ]
    for output in task.outputs:
        if not output.of:
            continue
        owner = records[output.of]
        if not any(
            item.check.kind is CheckKind.RESULT
            and item.check.output == output.name
            and matching.same_identifier(item.bound, owner.value)
            for item in passing
        ):
            gaps.append(f"{output.name} is not tied to {owner.name} {owner.value}")
    for record in task.records:
        shown = [
            item
            for item in passing
            if item.check.kind is CheckKind.RECORD
            and matching.same_identifier(item.check.expected, record.value)
        ]
        if record.within:
            parent = records[record.within]
            if not any(
                matching.same_identifier(item.bound, parent.value) for item in shown
            ):
                gaps.append(
                    f"{record.name} {record.value} is not shown to belong to "
                    f"{parent.name} {parent.value}"
                )
        elif not shown and not any(
            matching.same_identifier(item.bound, record.value) for item in passing
        ):
            gaps.append(f"{record.name} {record.value} is not shown")
    return gaps


def _undeclared(task: Task, declared: tuple[str, ...]) -> str | None:
    """Return why the task's outputs differ from the declared ones, or None.

    Examples
    --------
    >>> from computeruse.decider import TaskOutput
    >>> _undeclared(Task(outputs=(TaskOutput("order total"),)), ("order_total",))
    'the task must name exactly the declared outputs: order_total'
    >>> _undeclared(Task(outputs=(TaskOutput("order_total"),)), ("order_total",))
    >>> _undeclared(Task(outputs=(TaskOutput("order total"),)), ()) is None
    True
    """
    if not declared or {output.name for output in task.outputs} == set(declared):
        return None
    return "the task must name exactly the declared outputs: " + ", ".join(
        sorted(declared)
    )


def _task_problem(task: Task, goal: str) -> str | None:
    """Return why ``task`` cannot be the reading of ``goal``, or None.

    Every identifier must appear in the goal as whole words, so the task is
    grounded in what the person asked for rather than in anything a page
    showed. Names must be unique, and every reference must name a record of
    the task without a cycle. In a task about records, every output must say
    which record it belongs to, because an output tied to nothing could be
    any record's.

    Examples
    --------
    >>> from computeruse.decider import TaskOutput, TaskRecord
    >>> goal = "Report the balance of account S-1001 for member 12345."
    >>> member, account = TaskRecord("member", "12345"), TaskRecord(
    ...     "account", "S-1001", within="member")
    >>> _task_problem(Task((member, account), (TaskOutput("balance", "account"),)),
    ...               goal) is None
    True
    >>> _task_problem(Task((TaskRecord("member", "99999"),)), goal)
    'member 99999 is not an identifier the goal names'
    >>> _task_problem(Task((member,), (TaskOutput("balance"),)), goal)
    'balance must name the record it belongs to'
    """
    names = [record.name for record in task.records]
    if len(set(names)) != len(names) or not all(names):
        return "each record needs a name of its own"
    outputs = [output.name for output in task.outputs]
    if len(set(outputs)) != len(outputs) or not all(outputs):
        return "each output needs a name of its own"
    values = [" ".join(record.value.split()).casefold() for record in task.records]
    if len(set(values)) != len(values):
        return "two records share one identifier"
    wanted = " ".join(goal.split()).casefold()
    records = {record.name: record for record in task.records}
    for record in task.records:
        if not record.value.strip() or not matching.contains(
            wanted, record.value.casefold()
        ):
            return f"{record.name} {record.value} is not an identifier the goal names"
        seen = {record.name}
        parent = record.within
        while parent:
            if parent not in records:
                return f"{record.name} is within {parent}, which is not a record"
            if parent in seen:
                return f"{record.name} is within itself"
            seen.add(parent)
            parent = records[parent].within
    for output in task.outputs:
        if output.of and output.of not in records:
            return f"{output.name} belongs to {output.of}, which is not a record"
        if records and not output.of:
            return f"{output.name} must name the record it belongs to"
    return _requirement_problem(task)


def _changed_requirements(task: Task, claim: Finish) -> bool:
    required = {item.name: item.expected for item in task.requirements}
    return any(
        required.get(check.requirement) != check.expected
        for check in claim.checks
        if check.kind is CheckKind.REQUIREMENT
    )


def _requirement_problem(task: Task) -> str | None:
    records = {record.name for record in task.records}
    requirements = [item.name for item in task.requirements]
    if len(set(requirements)) != len(requirements) or not all(requirements):
        return "each requirement needs a name of its own"
    for item in task.requirements:
        if not item.expected.strip():
            return "each requirement needs an expected state"
        if item.of and item.of not in records:
            return "a requirement names an unknown record"
    if task.changes and all(item.context for item in task.requirements):
        return "a task that changes data needs a required end state"
    if records and not task.outputs and not task.requirements and not task.question:
        return "a record-only task needs an output or a required end state"
    return None


def _task_text(task: Task) -> str:
    """Describe a task in words, for the person asked to confirm it."""
    records = [
        f"{record.name} {record.value}"
        + (f" within {record.within}" if record.within else "")
        for record in task.records
    ]
    outputs = [
        output.name + (f" of {output.of}" if output.of else "")
        for output in task.outputs
    ]
    required = ", ".join(item.name for item in task.requirements) or "none"
    return (
        f"records: {', '.join(records) or 'none'}; "
        f"outputs: {', '.join(outputs) or 'none'}; "
        f"required states: {required}"
    )


def _interrupted(ids: str, after: tuple[Action, Outcome] | None) -> str:
    """Tell the next decision what became of the step a new window cut into."""
    if after is None:
        return (
            f"{ids} opened between steps and a person handled it; look at the "
            "page before deciding"
        )
    action, outcome = after
    if outcome is Outcome.OK:
        return (
            f"the {action.kind.value} before {ids} opened was performed, and a "
            "person has handled the window since; the page may already show "
            "its effect, so look before doing it again"
        )
    if outcome is Outcome.UNCERTAIN:
        return (
            f"the {action.kind.value} before {ids} opened may or may not have "
            "taken effect; look at the page before doing it again"
        )
    return (
        f"the {action.kind.value} was not performed, because {ids} needed a "
        "person first; propose it again only if the page still needs it"
    )


def _in_row(
    observation: Observation, node: AxNode, record: RecordEvidence
) -> RecordEvidence:
    """Scope row evidence to the row of ``node`` when only that row resolves it.

    Evidence tied by row names a control in the target's own row. In a list
    filtered to one member, every row holds a cell with that member, so the
    unscoped locator names many controls. When exactly one of them sits in
    the target's row, the evidence is scoped to that row, and every check,
    including the surface's own reading before acting, sees one control.

    Examples
    --------
    >>> from computeruse.actions import ObservationStatus, Scope
    >>> row = lambda key: Scope(ScopeKind.ROW, key)
    >>> link = AxNode("link", "A-2", scope=row("A-2"), row="r2")
    >>> cells = (AxNode("cell", "M-1", scope=row("A-1"), row="r1"),
    ...          AxNode("cell", "M-1", scope=row("A-2"), row="r2"))
    >>> seen = Observation("o", ObservationMode.STRUCTURED,
    ...     ObservationStatus.COMPLETE, PageState("https://h/"),
    ...     nodes=(link, *cells))
    >>> tied = RecordEvidence(AxLocator("cell", "M-1"), "M-1", Relation.ROW)
    >>> _in_row(seen, link, tied).source.scope.name
    'A-2'
    """
    source = record.source
    if (
        record.relation is not Relation.ROW
        or node.row is None
        or node.scope is None
        or node.scope.kind is not ScopeKind.ROW
        or not isinstance(source, (AxLocator, DomLocator))
        or source.scope is not None
    ):
        return record
    found = _named(observation, source)
    if len(found) < 2 or sum(other.row == node.row for other in found) != 1:
        return record
    scoped = dataclasses.replace(source, scope=node.scope)
    named = _named(observation, scoped)
    if len(named) != 1 or named[0].row != node.row:
        return record
    return dataclasses.replace(record, source=scoped)


def _record_problem(
    observation: Observation, node: AxNode, record: RecordEvidence
) -> str | None:
    """Return why ``record`` does not tie ``node`` to a record, or None.

    The evidence must name one control the observation showed, not withheld
    and not a field, showing exactly the value claimed, and tied to ``node``
    the way action evidence is.
    """
    sources = _named(observation, record.source)
    if len(sources) != 1:
        return "the record evidence does not name exactly one observed control"
    source = sources[0]
    if source.secret or source.value is not None:
        return "a field shows what was typed, so it cannot name a record"
    if " ".join(source.name.split()) != " ".join(record.displayed.split()):
        return "the record evidence is not a value the observation showed"
    return unbound(observation, record.relation, node, source)


def _in_goal(value: str, goal: str) -> bool:
    """Report whether the goal names ``value``, ignoring case and spacing.

    Examples
    --------
    >>> _in_goal("S-1001", "Report the balance of savings account s-1001.")
    True
    >>> _in_goal("S-1002", "Report the balance of savings account S-1001.")
    False
    """
    wanted = " ".join(value.split()).casefold()
    return bool(wanted) and wanted in " ".join(goal.split()).casefold()


def _named(observation: Observation, target: Target | None) -> list[AxNode]:
    return [node for node in observation.nodes if names_node(target, node)]


def _what(kind: ActionKind | None) -> str:
    return kind.value if kind is not None else "observation"


def _outcome(outcome: Outcome | None) -> str:
    return outcome.value if outcome is not None else Outcome.UNCERTAIN.value
