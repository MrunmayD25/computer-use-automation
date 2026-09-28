"""Public replay rejects invalid adapter and invocation boundaries."""

from __future__ import annotations

import pytest
from fakes import FakeClock, ScriptedSeat
from replay_fakes import INPUTS, PROFILE, App, Clock, bank, transfer_capability

from computeruse.actions import ObservationMode, ObservationRequest
from computeruse.control import Control, ControlChanged
from computeruse.escalation import Command, Handoff, HandoffOutcome, Mode, State
from computeruse.profile import load_profile
from computeruse.replay import MemoryReplayLog, Reason, Status, replay


class _Approves:
    def request(self, intervention):
        return Handoff(HandoffOutcome.APPROVED, intervention=intervention.intervention)


class _Listening:
    def attach(self, control):
        del control

    def show(self, status):
        del status

    def listening(self):
        return True


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE)
    return load_profile(path)


def _run(app, profile, control, clock):
    return replay(
        transfer_capability(),
        INPUTS,
        profile=profile,
        surface=app,
        control=control,
        clock=clock,
        log=MemoryReplayLog(),
    )


@pytest.mark.rule(1, 14)
@pytest.mark.parametrize("declaration", ["missing", None, [], {"unknown": frozenset()}])
def test_replay_refuses_an_invalid_capabilities_declaration(
    profile, monkeypatch, declaration
):
    app = bank()
    if declaration == "missing":
        monkeypatch.delattr(App, "capabilities")
    else:
        monkeypatch.setattr(app, "capabilities", lambda: declaration)
    clock = Clock()
    control = Control(mode=Mode.REPLAY, clock=clock, escalator=_Approves())

    result = _run(app, profile, control, clock)

    assert result.reason is Reason.UNSUPPORTED_BY_SURFACE
    assert app.acted == []
    assert control.status().state is State.FAILED


@pytest.mark.rule(14)
def test_replay_refuses_a_discovery_control_before_observing(profile):
    app = bank()
    clock = Clock()
    control = Control(mode=Mode.DISCOVERY, clock=clock, escalator=_Approves())

    with pytest.raises(ValueError, check=lambda error: "replay control" in str(error)):
        _run(app, profile, control, clock)

    assert app.acted == []
    assert app.looks == 0
    assert control.status().state is State.READY


@pytest.mark.rule(5, 14)
def test_replay_cannot_reuse_a_completed_invocation(profile):
    clock = Clock()
    control = Control(mode=Mode.REPLAY, clock=clock, escalator=_Approves())
    first = bank()
    _run(first, profile, control, clock)
    assert first.acted_on("Submit transfer") == 1
    second = bank()

    with pytest.raises(ValueError, check=lambda error: "already ended" in str(error)):
        _run(second, profile, control, clock)

    assert second.acted == []
    assert second.looks == 0
    assert control.status().state is State.COMPLETED


@pytest.mark.rule(14)
@pytest.mark.parametrize(
    ("command", "held"),
    [(Command.STOP, State.PAUSED), (Command.TAKE_CONTROL, State.HUMAN_CONTROL)],
)
def test_public_replay_does_not_dispatch_while_the_operator_holds_it(
    profile, command, held
):
    app = bank()
    ownership = []
    stopped = []

    def end_while_held(seat, status):
        if status.state is not held:
            return False
        ownership.append(status.owner)
        assert app.acted_on("Submit transfer") == 0
        seat.send(Command.TERMINATE, status)
        return True

    seat = ScriptedSeat(
        FakeClock(), moves=[end_while_held], wait_s=1.0, location=app.location()
    )
    control = Control(
        mode=Mode.REPLAY,
        clock=seat.clock,
        seat=seat,
        channels=[_Listening()],
        worker=False,
    )
    seat.control = control

    def stop_on_transfer(current):
        if current.state == "transfer" and not stopped:
            stopped.append(seat.send(command, control.status()))

    app.on_observe = stop_on_transfer
    result = _run(app, profile, control, seat.clock)

    assert ownership
    assert result.status is Status.TERMINATED
    assert app.acted_on("Submit transfer") == 0


@pytest.mark.rule(5, 14)
def test_an_unhandled_failure_ends_the_invocation_and_cannot_resume_it(profile):
    app = bank()
    clock = Clock()
    control = Control(mode=Mode.REPLAY, clock=clock, escalator=_Approves())

    def crash(_app, _action):
        raise RuntimeError("adapter crashed")

    app.on_act = crash
    with pytest.raises(
        RuntimeError, check=lambda error: str(error) == "adapter crashed"
    ):
        _run(app, profile, control, clock)

    assert control.status().state is State.FAILED
    with pytest.raises(ControlChanged):
        control.guard(app).observe(ObservationRequest(ObservationMode.STRUCTURED))
    with pytest.raises(ValueError, check=lambda error: "already ended" in str(error)):
        _run(app, profile, control, clock)
