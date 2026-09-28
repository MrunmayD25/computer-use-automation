"""Judge a finish claim by the page, regardless of text around an identifier."""

import dataclasses
import time

import pytest
from fakes import FakeClock, Screen, ScriptedDecider, ScriptedEscalator, ScriptedSurface

from computeruse.actions import (
    AxLocator,
    ObservationMode,
    ObservationProvenance,
    ObservationRequest,
    Point,
    RecordEvidence,
    Relation,
    ScreenTarget,
)
from computeruse.decider import (
    CheckKind,
    Finish,
    Match,
    Observe,
    ResultCheck,
    Task,
    TaskOutput,
    TaskRecord,
)
from computeruse.journal import MemoryJournal, Observed
from computeruse.loop import Ending, _unsupported, discover

STRUCTURED = ObservationRequest(ObservationMode.STRUCTURED)
VISUAL = ObservationRequest(ObservationMode.VISUAL)

# A member cell that shows the number with the member's name, as some sites do.
MEMBERS = """<!doctype html><html><body><h1>Members</h1>
<table><thead><tr><th>Member</th><th>Status</th></tr></thead><tbody>
<tr><td>12345 · Jordan Smith</td><td>Active</td></tr>
<tr><td>12346 · Riley Jones</td><td>Inactive</td></tr>
</tbody></table></body></html>"""


def claim(evidence_value: str) -> Finish:
    member = AxLocator("cell", "12345 · Jordan Smith")
    status = ResultCheck(
        CheckKind.RESULT,
        AxLocator("cell", "Active"),
        "Active",
        output="membership_status",
        record=RecordEvidence(member, evidence_value, Relation.ROW),
    )
    record = ResultCheck(CheckKind.RECORD, member, "12345", match=Match.EQUALS)
    return Finish({"membership_status": "Active"}, checks=(status, record))


def run(pages, finish: Finish):
    with pages({"/": MEMBERS}) as (surface, profile):
        journal = MemoryJournal()
        result = discover(
            "Return the membership status of member 12345.",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [Observe(STRUCTURED), finish],
                tasks=[
                    Task(
                        records=(TaskRecord("member", "12345"),),
                        outputs=(TaskOutput("membership_status", "member"),),
                    )
                ],
            ),
            escalator=ScriptedEscalator([]),
            journal=journal,
            clock=time.monotonic,
        )
        return result, journal


def test_evidence_quoting_the_whole_cell_is_split_around_the_goals_id(pages):
    # The model quotes the cell. The loop keeps 12345 as the value.
    result, journal = run(pages, claim("12345 · Jordan Smith"))
    assert result.ending is Ending.COMPLETED, result.detail
    assert result.outputs == {"membership_status": "Active"}
    assert all(item.passed for item in result.checks)
    # The loop judged the claim against its own observation, not an older one.
    assert any(
        isinstance(event, Observed)
        and event.provenance is ObservationProvenance.REFRESH
        for event in journal.events
    )


@pytest.mark.rule(7)
def test_evidence_naming_another_record_is_still_refused(pages):
    result, _ = run(pages, claim("12346 · Riley Jones"))
    assert result.ending is not Ending.COMPLETED


def test_a_refused_claim_says_how_to_correct_the_evidence():
    source = AxLocator("cell", "Member A-7")
    check = ResultCheck(
        CheckKind.RESULT,
        AxLocator("cell", "Active"),
        "Active",
        output="status",
        record=RecordEvidence(source, "Member A-7", Relation.ROW),
    )
    problem = _unsupported(
        Finish({"status": "Active"}, checks=(check,)), "Look up 12345.", ("12345",)
    )
    assert problem is not None
    assert "'12345'" in problem
    assert "prefix or suffix" in problem


# A search for a member the application does not hold.
EMPTY = """<!doctype html><html><body><h1>Members</h1>
<label>Search <input value="99999"></label>
<p role="status">0 members found</p>
<table><thead><tr><th>Member</th><th>Status</th></tr></thead><tbody>{rows}</tbody>
</table></body></html>"""


def not_found(pages, rows: str = "", outcome: str = "record_not_found"):
    searched = ResultCheck(CheckKind.RECORD, AxLocator("textbox", "Search"), "99999")
    nothing = ResultCheck(
        CheckKind.STATE, AxLocator("status", "0 members found"), "0 members found"
    )
    finish = Finish({}, checks=(nothing, searched), outcome=outcome)
    with pages({"/": EMPTY.format(rows=rows)}) as (surface, profile):
        return discover(
            "Return the membership status of member 99999.",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [Observe(STRUCTURED), finish],
                tasks=[
                    Task(
                        records=(TaskRecord("member", "99999"),),
                        outputs=(TaskOutput("membership_status", "member"),),
                    )
                ],
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
        )


def test_a_search_that_found_nothing_completes_as_record_not_found(pages):
    result = not_found(pages)
    assert result.ending is Ending.COMPLETED, result.detail
    assert result.outcome == "record_not_found"
    assert result.outputs == {}


@pytest.mark.rule(15)
def test_record_not_found_is_refused_while_a_result_row_shows_the_record(pages):
    row = "<tr><td>99999 · Casey Patel</td><td>Active</td></tr>"
    result = not_found(pages, rows=row)
    assert result.ending is not Ending.COMPLETED
    assert result.outcome == ""


def test_an_outcome_outside_the_closed_list_is_refused(pages):
    result = not_found(pages, outcome="member_closed")
    assert result.ending is not Ending.COMPLETED


def test_an_outcome_about_a_record_that_exists_may_show_its_row(pages):
    # The operator may not act on the member, and the member's row is there.
    row = "<tr><td>99999 · Casey Patel</td><td>Active</td></tr>"
    result = not_found(pages, rows=row, outcome="permission_denied")
    assert result.ending is Ending.COMPLETED, result.detail
    assert result.outcome == "permission_denied"


# A form that shows the member it is for, beside its refusal.
REFUSED = """<!doctype html><html><body><h1>Open account</h1>
<dl><dt>Member</dt><dd>99999 · inactive</dd></dl>
<p role="status">This member cannot open an account</p></body></html>"""


def refused(pages, outcome: str):
    shown = ResultCheck(
        CheckKind.RECORD,
        AxLocator("definition", "99999 · inactive"),
        "99999",
        Match.CONTAINS,
    )
    state = ResultCheck(
        CheckKind.STATE,
        AxLocator("status", "This member cannot open an account"),
        "This member cannot open an account",
    )
    finish = Finish({}, checks=(state, shown), outcome=outcome)
    with pages({"/": REFUSED}) as (surface, profile):
        return discover(
            "Return the membership status of member 99999.",
            profile,
            surface=surface,
            decider=ScriptedDecider(
                [Observe(STRUCTURED), finish],
                tasks=[
                    Task(
                        records=(TaskRecord("member", "99999"),),
                        outputs=(TaskOutput("membership_status", "member"),),
                    )
                ],
            ),
            escalator=ScriptedEscalator([]),
            journal=MemoryJournal(),
            clock=time.monotonic,
        )


@pytest.mark.rule(15)
def test_an_existing_record_is_shown_by_a_control_holding_its_identifier(pages):
    result = refused(pages, "record_ineligible")
    assert result.ending is Ending.COMPLETED, result.detail
    assert result.outcome == "record_ineligible"


def test_not_found_still_needs_a_control_holding_exactly_the_identifier(pages):
    result = refused(pages, "record_not_found")
    assert result.ending is not Ending.COMPLETED


# A painted screen after a search for a member the application does not hold.
# The screen draws the typed number in its search field, and the answer below.
PAINTED = Screen("https://sandbox.example.test/members")
MESSAGE = "No member with number 99999."


def painted_not_found(profile, record_line: str, record_capture: str = "obs-2"):
    state = ResultCheck(
        CheckKind.STATE, ScreenTarget("obs-2", Point(40, 750)), MESSAGE, Match.CONTAINS
    )
    searched = ResultCheck(
        CheckKind.RECORD, ScreenTarget(record_capture, Point(60, 200)), "99999"
    )
    decider = ScriptedDecider(
        [
            Observe(VISUAL),
            Observe(VISUAL),
            Finish({}, checks=(state, searched), outcome="record_not_found"),
        ],
        tasks=[
            Task(
                records=(TaskRecord("member", "99999"),),
                outputs=(TaskOutput("membership_status", "member"),),
            )
        ],
    )
    result = discover(
        "Return the membership status of member 99999.",
        profile,
        surface=ScriptedSurface([PAINTED], extracts=[MESSAGE, record_line]),
        decider=decider,
        escalator=ScriptedEscalator([]),
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    notices = [notice for seen in decider.seen for notice in seen.notices]
    return result, " ".join(notices)


@pytest.mark.rule(15)
def test_a_painted_answer_that_names_the_searched_record_proves_not_found(profile):
    result, notices = painted_not_found(profile, MESSAGE)
    assert result.ending is Ending.COMPLETED, notices
    assert result.outcome == "record_not_found"


@pytest.mark.rule(15)
def test_a_painted_line_that_is_only_the_identifier_is_not_proof(profile):
    """A line holding only the number may be the field the run typed it into."""
    result, notices = painted_not_found(profile, "99999")
    assert result.ending is not Ending.COMPLETED
    assert "may be a field" in notices
    # Found in canvas batch 51: the model kept pointing at the field, so the
    # notice says which painted line would show the record.
    assert "among other words, such as the application's answer" in notices


@pytest.mark.rule(15)
def test_a_painted_record_line_from_another_capture_is_not_proof(profile):
    """Both lines must come from the screen the run sees now, as one screen."""
    result, notices = painted_not_found(profile, MESSAGE, record_capture="obs-1")
    assert result.ending is not Ending.COMPLETED
    assert "not in the current observation" in notices


# A painted member screen. The label line names the member once, so every
# reading of the same picture is about that member.
MEMBER_SCREEN = ("Member number: 12345", "Name: Sam Lee", "Status: active")


def painted_lookup(profile, screen: tuple[str, ...]):
    record = ResultCheck(
        CheckKind.RECORD, ScreenTarget("obs-1", Point(60, 160)), "12345"
    )
    status = ResultCheck(
        CheckKind.RESULT,
        ScreenTarget("obs-1", Point(60, 230)),
        "active",
        Match.EQUALS,
        output="membership_status",
    )
    decider = ScriptedDecider(
        [
            Observe(VISUAL),
            Finish({"membership_status": "active"}, checks=(record, status)),
        ],
        tasks=[
            Task(
                records=(TaskRecord("member", "12345"),),
                outputs=(TaskOutput("membership_status", "member"),),
            )
        ],
    )
    escalator = ScriptedEscalator([])
    result = discover(
        "Return the membership status of member 12345.",
        profile,
        surface=ScriptedSurface(
            [PAINTED],
            extracts=["Member number: 12345", "Status: active"],
            painted=screen,
        ),
        decider=decider,
        escalator=escalator,
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    notices = [notice for seen in decider.seen for notice in seen.notices]
    notices.extend(gap for request in escalator.requests for gap in request.unverified)
    return result, " ".join(notices)


@pytest.mark.rule(15)
def test_a_painted_screen_that_names_its_member_once_ties_its_readings(profile):
    result, notices = painted_lookup(profile, MEMBER_SCREEN)
    assert result.ending is Ending.COMPLETED, notices
    assert result.outputs == {"membership_status": "active"}


@pytest.mark.rule(15)
def test_a_painted_list_of_members_ties_nothing(profile):
    """Two lines with the member label are a list, not one record."""
    listed = ("Member number: 12345", "Member number: 12346", "Status: active")
    result, notices = painted_lookup(profile, listed)
    assert result.ending is not Ending.COMPLETED
    assert "membership_status is not tied to member 12345" in notices


@pytest.mark.rule(14, 15, 16)
@pytest.mark.parametrize(
    ("answer", "changed", "completed"),
    [
        ("approved", False, True),
        ("approved", True, False),
        ("resumed", False, False),
        ("rejected", False, False),
        ("timed_out", False, False),
    ],
)
def test_a_passing_partial_finish_immediately_offers_its_missing_requirement(
    pages, monkeypatch, answer, changed, completed
):
    import dataclasses

    from computeruse.actions import Action
    from computeruse.control import Control
    from computeruse.decider import AskHuman, Propose, TaskRequirement
    from computeruse.escalation import Ask, Handoff, HandoffOutcome, Mode, Trigger
    from computeruse.loop import Verification
    from computeruse.profile import ActionKind

    markup = """<h1>Account opened</h1>
      <table><tr><th>Member</th><th>Account</th></tr>
      <tr><td>M-7</td><td>A-9</td></tr></table>
      <button onclick="document.body.textContent='Receipt lost'">New search</button>"""
    with pages({"/": markup}) as (surface, profile):
        member = AxLocator("cell", "M-7")
        account = AxLocator("cell", "A-9")
        checks = (
            ResultCheck(CheckKind.RECORD, member, "M-7"),
            ResultCheck(
                CheckKind.RESULT,
                account,
                "A-9",
                output="number",
                record=RecordEvidence(member, "M-7", Relation.ROW),
            ),
        )
        decider = ScriptedDecider(
            [
                Observe(STRUCTURED),
                Finish({"number": "A-9"}, checks=checks),
                AskHuman(Trigger.NO_PROGRESS, "operator requested more work"),
                Propose(Action(ActionKind.CLICK, AxLocator("button", "New search"))),
            ],
            tasks=[
                Task(
                    records=(TaskRecord("member", "M-7"),),
                    outputs=(TaskOutput("number", "member"),),
                    requirements=(TaskRequirement("product", "Reserve", "member"),),
                )
            ],
        )
        escalator = ScriptedEscalator([Handoff(HandoffOutcome(answer))])
        control = Control(
            mode=Mode.DISCOVERY, clock=time.monotonic, escalator=escalator
        )
        intervene = control.intervene

        def handoff(request, checks=None):
            return dataclasses.replace(intervene(request, checks), changed=changed)

        monkeypatch.setattr(control, "intervene", handoff)
        result = discover(
            "Open a Reserve account for member M-7 and return number.",
            profile,
            surface=surface,
            decider=decider,
            control=control,
            journal=MemoryJournal(),
            clock=time.monotonic,
        )
        request = escalator.requests[0]
        assert request.step == 2
        assert request.ask is Ask.APPROVAL
        assert request.outputs == {"number": "A-9"}
        assert request.unverified == ("required state product is not established",)
        assert surface._page.get_by_role("heading", name="Account opened").is_visible()
        assert (result.ending is Ending.COMPLETED) is completed
        if completed:
            assert len(decider.seen) == 2
            assert result.verification is Verification.PERSON
            assert result.outputs == {"number": "A-9"}
            assert len(result.checks) == 2
            assert all(check.passed for check in result.checks)
        else:
            assert result.verification is None


@pytest.mark.rule(10)
def test_a_fact_kept_from_a_painted_line_compares_as_that_line(profile):
    """Found in canvas batch 53: a read kept "Accountnumber:AC5" whole.

    The finish cited that fact for the value after the label, and the loop
    compared it as plain text, so "AC5" never equalled the line. A fact
    read from a screenshot now compares as the painted line it came from.
    """
    from computeruse.actions import Action
    from computeruse.decider import FactRef, Propose
    from computeruse.profile import ActionKind

    line = ScreenTarget("obs-1", Point(60, 160))
    number = ResultCheck(
        CheckKind.RESULT,
        FactRef("read_at_step_2"),
        "AC5",
        Match.EQUALS,
        output="account_number",
    )
    result = discover(
        "Return the new account number as one output named account_number.",
        profile,
        surface=ScriptedSurface([PAINTED], extracts=["Accountnumber:AC5"]),
        decider=ScriptedDecider(
            [
                Observe(VISUAL),
                Propose(Action(ActionKind.READ, line)),
                Finish({"account_number": "AC5"}, checks=(number,)),
            ],
            tasks=[Task(outputs=(TaskOutput("account_number"),))],
        ),
        escalator=ScriptedEscalator([]),
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert result.ending is Ending.COMPLETED, result.detail


@pytest.mark.rule(15)
def test_a_passing_record_tie_gap_is_offered_before_another_model_decision(profile):
    """No later claim can replace a result already awaiting confirmation."""
    from computeruse.decider import AskHuman
    from computeruse.escalation import Ask, Handoff, HandoffOutcome, Trigger

    record = ResultCheck(
        CheckKind.RECORD, ScreenTarget("obs-1", Point(60, 160)), "12345"
    )
    status = ResultCheck(
        CheckKind.RESULT,
        ScreenTarget("obs-1", Point(60, 230)),
        "active",
        Match.EQUALS,
        output="membership_status",
    )
    passing = Finish({"membership_status": "active"}, checks=(record, status))
    wrong = dataclasses.replace(status, expected="inactive")
    failing = Finish({"membership_status": "inactive"}, checks=(record, wrong))
    escalator = ScriptedEscalator([Handoff(HandoffOutcome.APPROVED)])
    result = discover(
        "Return the membership status of member 12345.",
        profile,
        surface=ScriptedSurface(
            [PAINTED],
            extracts=["Member number: 12345", "Status: active"] * 2,
            painted=("Member number: 12345", "Member number: 12346"),
        ),
        decider=ScriptedDecider(
            [
                Observe(VISUAL),
                passing,
                failing,
                AskHuman(Trigger.UNVERIFIED_RESULT, "please verify the member"),
            ],
            tasks=[
                Task(
                    records=(TaskRecord("member", "12345"),),
                    outputs=(TaskOutput("membership_status", "member"),),
                )
            ],
        ),
        escalator=escalator,
        journal=MemoryJournal(),
        clock=FakeClock(),
    )
    assert [request.ask for request in escalator.requests] == [Ask.APPROVAL]
    assert result.ending is Ending.COMPLETED
    assert result.outputs == {"membership_status": "active"}
