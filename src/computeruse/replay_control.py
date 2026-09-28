"""Adapt the live control window and terminal to deterministic replay requests."""

from __future__ import annotations

from computeruse.control import Control
from computeruse.escalation import (
    Ask,
    HandoffOutcome,
    InterventionRequest,
    Mode,
    Trigger,
)
from computeruse.profile import Profile
from computeruse.replay import Answer, HelpAnswer, HelpRequest, Reason
from computeruse.replay import Ask as ReplayAsk


class ControlledOperator:
    """Use the same recorded session and ownership state as discovery."""

    def __init__(self, control: Control, profile: Profile) -> None:
        self.control = control
        self.profile = profile

    def request(self, request: HelpRequest) -> HelpAnswer:
        """Translate one replay request without storing or reusing its approval."""
        trigger = {
            Reason.NEW_WINDOW: Trigger.NEW_WINDOW,
            Reason.APPROVAL_REQUIRED: Trigger.RISKY_ACTION,
            Reason.APPROVAL_INVALIDATED: Trigger.RISKY_ACTION,
            Reason.RECORD_EVIDENCE_REQUIRED: Trigger.RECORD_EVIDENCE_REQUIRED,
            Reason.DELIVERY_UNCERTAIN: Trigger.DELIVERY_UNCERTAIN,
            Reason.TARGET_AMBIGUOUS: Trigger.AMBIGUOUS_TARGET,
            Reason.AMBIGUOUS_STATE: Trigger.AMBIGUOUS_TARGET,
            Reason.OBSERVATION_INCOMPLETE: Trigger.INSUFFICIENT_OBSERVATION,
            Reason.VERIFICATION_FAILED: Trigger.UNVERIFIED_RESULT,
            Reason.COMPLETION_CHECK_FAILED: Trigger.UNVERIFIED_RESULT,
        }.get(request.reason, Trigger.NO_PROGRESS)
        if request.ask is ReplayAsk.APPROVAL and request.reason is Reason.PLANNED_STEP:
            # The replay's own result confirmation asks whether a result the
            # page cannot fully show is correct.
            trigger = Trigger.UNVERIFIED_RESULT
        handoff = self.control.intervene(
            InterventionRequest(
                trigger=trigger,
                goal=request.capability,
                profile_id=self.profile.profile_id,
                step=request.step,
                route=request.route,
                reason=_reason(request),
                timeout_s=request.timeout_s,
                action=request.proposal,
                ask=Ask(request.ask.value),
                mode=Mode.REPLAY,
                capability=request.capability,
                unverified=request.expected,
            )
        )
        answer = {
            HandoffOutcome.APPROVED: Answer.APPROVED,
            HandoffOutcome.RESUMED: Answer.RESUMED,
            HandoffOutcome.REJECTED: Answer.REJECTED,
            HandoffOutcome.TERMINATED: Answer.TERMINATED,
            HandoffOutcome.TIMED_OUT: Answer.TIMED_OUT,
        }.get(handoff.outcome, Answer.TIMED_OUT)
        if handoff.changed and answer is Answer.APPROVED:
            answer = Answer.RESUMED
        return HelpAnswer(answer, request.intervention, handoff.intervention)


def _reason(request: HelpRequest) -> str:
    """Put a request's question first, and the step it is about on the next line."""
    if not request.question:
        stopped = f"Settle why the replay stopped ({request.reason.value})."
        return f"{stopped}\nStep {request.node}"
    *said, operation = request.question.split("\n")
    if not said:
        return f"{operation}\nStep {request.node}"
    # Put the step's operation and values after the instructions.
    return "\n".join([*said, f"Step {request.node}: {operation}"])
