"""A check that compares a control's text is a read (rule 19).

Discovery gates every read it makes, including reads behind its checks. Replay
does the same for every text comparison in a branch, precondition,
postcondition, or result. The read grant and any denial apply. A risky read
needs a person's approval once per visit. Presence, absence, and route checks
only inspect the page.
"""

from __future__ import annotations

import dataclasses

import pytest
from replay_fakes import (
    PROFILE,
    App,
    Clock,
    Page,
    People,
    _dom,
    dd,
    go,
    transfer_capability,
)

from computeruse.capability import (
    AtRoute,
    CheckNode,
    Match,
    Present,
    ResultKind,
    ResultNode,
    Shows,
    check_profile,
    constant,
    validate,
)
from computeruse.escalation import Ask, HandoffOutcome
from computeruse.profile import ActionKind, Risk, load_profile
from computeruse.replay import MemoryReplayLog, Reason, Status, replay


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE)
    return load_profile(path)


def at_desk(condition):
    base = transfer_capability()
    target = _dom("state", "/desk", "state")
    return dataclasses.replace(
        base,
        application=dataclasses.replace(
            base.application, entry_route="/desk", markers=(AtRoute("/desk"),)
        ),
        inputs=(),
        outputs=(),
        variables=(),
        secrets=(),
        outcomes=(),
        targets=(target,),
        nodes=(
            CheckNode("branch", (go("done", condition),)),
            ResultNode("done", ResultKind.SUCCESS, "", (AtRoute("/desk"),)),
        ),
        entry="branch",
    )


def run(capability, profile, people):
    assert validate(capability) == ()
    assert check_profile(capability, profile) == ()
    app = App({"desk": Page("/desk", (dd("Status", "Approved", "state"),))}, "desk", {})
    clock = Clock()
    return replay(
        capability,
        {},
        profile=profile,
        surface=app,
        control=people.control(clock),
        log=MemoryReplayLog(),
        clock=clock,
    )


SHOWS = Shows("state", constant("Approved"), Match.EQUALS)


@pytest.mark.rule(19)
def test_a_branch_check_that_compares_text_needs_the_read_grant(profile) -> None:
    ungranted = dataclasses.replace(
        profile,
        actions={
            kind: risk
            for kind, risk in profile.actions.items()
            if kind is not ActionKind.READ
        },
    )
    result = run(at_desk(SHOWS), ungranted, People([]))
    assert result.status is Status.FAILED
    assert result.reason is Reason.POLICY_DENIED


@pytest.mark.rule(19)
def test_a_presence_check_only_looks(profile) -> None:
    ungranted = dataclasses.replace(
        profile,
        actions={
            kind: risk
            for kind, risk in profile.actions.items()
            if kind is not ActionKind.READ
        },
    )
    result = run(at_desk(Present("state")), ungranted, People([]))
    assert result.status is Status.SUCCEEDED


@pytest.mark.rule(19)
def test_a_risky_read_in_a_branch_asks_a_person_once(profile) -> None:
    risky = dataclasses.replace(
        profile, actions={**profile.actions, ActionKind.READ: Risk.RISKY}
    )
    people = People([HandoffOutcome.APPROVED])
    result = run(at_desk(SHOWS), risky, people)
    assert result.status is Status.SUCCEEDED
    assert [request.ask for request in people.requests] == [Ask.APPROVAL]


@pytest.mark.rule(19)
def test_a_declined_read_ends_the_replay_rather_than_choose_a_branch(profile) -> None:
    risky = dataclasses.replace(
        profile, actions={**profile.actions, ActionKind.READ: Risk.RISKY}
    )
    result = run(at_desk(SHOWS), risky, People([HandoffOutcome.REJECTED]))
    assert result.status is not Status.SUCCEEDED
    assert result.reason is Reason.APPROVAL_REJECTED
