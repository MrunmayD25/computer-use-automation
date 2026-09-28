"""Confirming the website text a capability saves, and the list that remembers it."""

import dataclasses

import pytest
from replay_fakes import (
    BALANCE,
    INPUTS,
    MEMBER,
    PROFILE,
    Clock,
    bank,
    go,
    transfer_capability,
)

from computeruse.capability import (
    Bound,
    Match,
    Purpose,
    ResultKind,
    ResultNode,
    Shows,
    validate,
)
from computeruse.confirm import Decision, _record_inputs, compare, decide, judge
from computeruse.control import Control, Order
from computeruse.escalation import Command, Mode, Via
from computeruse.model import _labels
from computeruse.profile import ProfileError, load_profile
from computeruse.recorder import Candidate, Place
from computeruse.words import (
    Confirmed,
    Word,
    WordsError,
    add_words,
    load_words,
    words_path,
)

LABEL = Candidate("Search", frozenset({Place.CONTROL}))
CELL = Candidate("Jordan Smith", frozenset({Place.RECORD}))
MIXED = Candidate("Status", frozenset({Place.CONTROL, Place.RECORD}))


class Labels:
    """A classifier that calls the given texts labels and records what it saw."""

    def __init__(self, *labels: str) -> None:
        self.labels = frozenset(labels)
        self.seen: list[tuple[str, ...]] = []

    def classify_words(self, texts: tuple[str, ...]) -> frozenset[str]:
        self.seen.append(texts)
        return self.labels & frozenset(texts)


# The word list.


def test_a_site_has_one_list_beside_its_profile(tmp_path):
    profile = tmp_path / "classic.yaml"
    assert words_path(profile) == tmp_path / "classic.safe-text.yaml"
    assert load_words(words_path(profile)) == ()


def test_words_are_added_once_with_how_they_were_confirmed(tmp_path):
    listed = tmp_path / "site.safe-text.yaml"
    listed.write_text("words:\n  - Go\n")
    added = add_words(
        listed,
        (Word("Go", Confirmed.MODEL), Word("Status", Confirmed.COMPARISON)),
    )
    assert added == (Word("Status", Confirmed.COMPARISON),)
    # A word a person wrote keeps its own confirmation.
    assert load_words(listed) == (
        Word("Go", Confirmed.OPERATOR),
        Word("Status", Confirmed.COMPARISON),
    )


@pytest.mark.rule(8)
@pytest.mark.parametrize("confirmation", list(Confirmed))
def test_private_values_override_existing_and_new_word_confirmations(
    tmp_path, confirmation
):
    listed = tmp_path / "site.safe-text.yaml"
    private = "PRIVATE-WORD-7462"
    exposed = f"Receipt {private.lower()} ready"
    add_words(listed, (Word(exposed, confirmation), Word("Search", Confirmed.OPERATOR)))

    assert add_words(
        listed,
        (Word(exposed, confirmation), Word("Ready", Confirmed.MODEL)),
        excluded=(private,),
    ) == (Word("Ready", Confirmed.MODEL),)
    assert load_words(listed) == (
        Word("Search", Confirmed.OPERATOR),
        Word("Ready", Confirmed.MODEL),
    )
    assert private.casefold() not in listed.read_text().casefold()
    assert add_words(listed, (), excluded=("Ready",)) == ()
    assert load_words(listed) == (Word("Search", Confirmed.OPERATOR),)


@pytest.mark.parametrize(
    "body",
    [
        "words: Go\n",
        "words:\n  - Go\n  - Go\n",
        "words:\n  - {text: Go, confirmed_by: guessed}\n",
        "words: []\nextra: 1\n",
        "words:\n  - ''\n",
    ],
)
def test_a_malformed_list_confirms_nothing_rather_than_part(tmp_path, body):
    listed = tmp_path / "site.safe-text.yaml"
    listed.write_text(body)
    with pytest.raises(WordsError):
        load_words(listed)


# The profile's second test record.


def _profile(tmp_path, extra: str, environment: str = "sandbox"):
    path = tmp_path / "profile.yaml"
    path.write_text(
        PROFILE.replace("environment: sandbox", f"environment: {environment}") + extra
    )
    return path


def test_a_sandbox_profile_may_name_a_second_test_record(tmp_path):
    extra = "confirmation:\n  inputs:\n    member_id: '10002'\n"
    profile = load_profile(_profile(tmp_path, extra))
    assert dict(profile.confirmation) == {"member_id": "10002"}
    assert dict(profile.outcomes) == {}
    assert dict(load_profile(_profile(tmp_path, "")).confirmation) == {}


def test_a_sandbox_profile_names_the_inputs_that_give_each_outcome(tmp_path):
    extra = (
        "confirmation:\n  inputs:\n    member_id: '10002'\n"
        "  outcomes:\n"
        "    record_not_found:\n      member_id: '99999'\n"
        "    permission_denied:\n      operator_id: OP0001\n"
    )
    profile = load_profile(_profile(tmp_path, extra))
    assert {name: dict(case) for name, case in profile.outcomes.items()} == {
        "record_not_found": {"member_id": "99999"},
        "permission_denied": {"operator_id": "OP0001"},
    }


def test_an_outcome_the_product_does_not_know_is_refused(tmp_path):
    extra = (
        "confirmation:\n  inputs:\n    member_id: '10002'\n"
        "  outcomes:\n    member_on_holiday:\n      member_id: '99999'\n"
    )
    with pytest.raises(ProfileError, match="outcomes"):
        load_profile(_profile(tmp_path, extra))


@pytest.mark.parametrize(
    ("extra", "environment"),
    [
        ("confirmation:\n  inputs:\n    member_id: '10002'\n", "production"),
        ("confirmation:\n  inputs: {}\n", "sandbox"),
        ("confirmation:\n  inputs:\n    member_id: ''\n", "sandbox"),
        ("confirmation:\n  record:\n    member_id: '10002'\n", "sandbox"),
        ("confirmation:\n  missing:\n    member_id: '9'\n", "sandbox"),
        (
            "confirmation:\n  inputs:\n    member_id: '1'\n  missing: {}\n",
            "sandbox",
        ),
        (
            (
                "confirmation:\n  inputs:\n    member_id: '1'\n"
                "  missing:\n    member_id: '9'\n"
            ),
            "production",
        ),
    ],
)
def test_a_second_record_is_refused_outside_a_sandbox_or_malformed(
    tmp_path, extra, environment
):
    with pytest.raises(ProfileError):
        load_profile(_profile(tmp_path, extra, environment))


# Who confirms what.


def test_the_model_sees_only_interface_labels_and_never_a_records_text():
    classifier = Labels("Search", "Jordan Smith", "Status")
    decided = judge((LABEL, CELL, MIXED), classifier)
    assert classifier.seen == [("Search",)]
    assert decided.confirmed == (Word("Search", Confirmed.MODEL),)
    assert decided.unconfirmed == (CELL, MIXED)


def test_a_person_sees_only_what_is_left_and_nothing_is_saved_without_them():
    asked: list[tuple[Candidate, ...]] = []

    def person(left, note):
        asked.append(left)
        del note
        # A word nobody asked about is never confirmed through a person.
        return (Word("Status", Confirmed.PERSON), Word("Other", Confirmed.PERSON))

    decided = decide(
        (LABEL, CELL, MIXED),
        compared=None,
        classifier=Labels("Search"),
        veto=None,
        ask=person,
    )
    assert asked == [(CELL, MIXED)]
    assert Word("Status", Confirmed.PERSON) in decided.confirmed
    assert all(word.text != "Other" for word in decided.confirmed)
    assert decided.unconfirmed == (CELL,)
    assert not decided.complete
    alone = decide((CELL,), compared=None, classifier=None, veto=None, ask=None)
    assert alone.unconfirmed == (CELL,)


@pytest.mark.rule(8)
def test_a_failed_comparison_is_never_passed_to_the_model():
    classifier = Labels("Search")
    failed = decide(
        (LABEL,),
        compared=decide((LABEL,), compared=None, classifier=None, veto=None, ask=None),
        classifier=classifier,
        veto=None,
        ask=None,
    )
    assert classifier.seen == []
    assert failed.unconfirmed == (LABEL,)


class Keeps:
    """A model that keeps only the named texts, and records what it saw."""

    def __init__(self, *kept: str) -> None:
        self.kept = frozenset(kept)
        self.seen: list[tuple[str, ...]] = []

    def keep_words(self, texts: tuple[str, ...]) -> frozenset[str]:
        self.seen.append(texts)
        return self.kept & frozenset(texts)


def test_the_model_takes_data_away_from_what_a_comparison_confirmed():
    """Found in a virtualized run: two compared members were both Alex Morgan."""
    name = Candidate("Alex Morgan", frozenset({Place.RECORD}))
    compared = Decision(
        (
            Word("Search", Confirmed.COMPARISON),
            Word("Alex Morgan", Confirmed.COMPARISON),
        ),
        (),
    )
    veto = Keeps("Search")
    decided = decide(
        (LABEL, name), compared=compared, classifier=None, veto=veto, ask=None
    )
    assert veto.seen == [("Search", "Alex Morgan")]
    assert decided.confirmed == (Word("Search", Confirmed.COMPARISON),)
    assert decided.unconfirmed == (name,)
    assert not decided.complete


@pytest.mark.rule(8)
def test_the_model_never_confirms_what_a_comparison_did_not():
    compared = Decision((), (CELL,), "the replay for the second record ended failed")
    veto = Keeps(CELL.text)
    decided = decide((CELL,), compared=compared, classifier=None, veto=veto, ask=None)
    assert veto.seen == []
    assert decided.unconfirmed == (CELL,)


def test_the_model_answer_names_only_texts_it_was_asked_about():
    body = {
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "name": "classify_texts",
                "call_id": "c1",
                "arguments": (
                    '{"texts": [{"text": "Go", "verdict": "label"},'
                    ' {"text": "Alex", "verdict": "label"},'
                    ' {"text": "Next", "verdict": "label"},'
                    ' {"text": "Next", "verdict": "risky"}]}'
                ),
            }
        ],
    }
    assert _labels(body, ("Go", "Next", "Status")) == frozenset({"Go"})


# A comparison replays the draft for a second record.


def _lookup():
    """The transfer fixture cut to a read-only lookup a replay needs no help for."""
    capability = transfer_capability()
    kept = ("enter_member", "search", "read_balance", "not_found")
    nodes = [node for node in capability.nodes if node.node_id in kept]
    nodes = [
        dataclasses.replace(node, transitions=(go("found"),))
        if node.node_id == "read_balance"
        else node
        for node in nodes
    ]
    nodes.append(ResultNode("found", ResultKind.SUCCESS, "", (Bound(BALANCE),)))
    lookup = dataclasses.replace(
        capability, inputs=capability.inputs[:1], secrets=(), nodes=tuple(nodes)
    )
    assert validate(lookup) == ()
    return lookup


@pytest.mark.rule(8)
def test_a_second_record_that_replays_confirms_only_texts_it_met(tmp_path):
    """The lookup uses "Search" and never shows "Status" (R5).

    The comparison confirms text that replay saw unchanged on the second
    record's page. Text it never saw remains for a person to confirm.
    """
    profile = load_profile(_profile(tmp_path, ""))
    clock = Clock()
    decided = compare(
        _lookup(),
        (LABEL, MIXED),
        {"member_id": "10002"},
        {"member_id": "10001"},
        profile=profile,
        surface=bank("10002"),
        control=Control(mode=Mode.REPLAY, clock=clock),
        clock=clock,
        sleep=clock.advance,
    )
    assert [word.text for word in decided.confirmed] == ["Search"]
    assert {word.confirmed_by for word in decided.confirmed} == {Confirmed.COMPARISON}
    assert decided.unconfirmed == (MIXED,)
    assert "never met" in decided.note


def test_a_comparison_stops_before_a_change_and_a_repeated_record_confirms_nothing(
    tmp_path,
):
    profile = load_profile(_profile(tmp_path, ""))
    clock = Clock()
    # The transfer's payment needs approval, so the comparison stops before
    # it. The comparison confirms text it saw before that step without asking
    # anyone.
    stopped = compare(
        transfer_capability(),
        (LABEL,),
        {**INPUTS, "member_id": "10002"},
        INPUTS,
        profile=profile,
        surface=bank("10002"),
        control=Control(mode=Mode.REPLAY, clock=clock),
        clock=clock,
        sleep=clock.advance,
    )
    assert stopped.ended == "stopped"
    assert stopped.confirmed == (Word("Search", Confirmed.COMPARISON),)
    same = compare(
        _lookup(),
        (LABEL,),
        {"member_id": "10001"},
        {"member_id": "10001"},
        profile=profile,
        surface=bank(),
        control=Control(mode=Mode.REPLAY, clock=clock),
        clock=clock,
        sleep=clock.advance,
    )
    assert same.note == "the second record repeats an input that names a record"


class Approver:
    """A person on the run's channel who approves the one step asked of them."""

    def __init__(self) -> None:
        self.control = None
        self.asked: list[str] = []

    def attach(self, control) -> None:
        self.control = control

    def listening(self) -> bool:
        return True

    def show(self, status) -> None:
        offer = status.offer
        if offer is None or self.control is None or offer.intervention in self.asked:
            return
        self.asked.append(offer.intervention)
        self.control.submit(
            Order(
                Command.APPROVE,
                status.run,
                status.revision,
                offer.intervention,
                Via.PANEL,
            )
        )


def test_a_comparison_never_asks_a_person_to_approve_a_change(tmp_path):
    profile = load_profile(_profile(tmp_path, ""))
    clock = Clock()
    person = Approver()
    control = Control(mode=Mode.REPLAY, clock=clock, channels=[person])
    # Even with someone on the run's channel, the comparison stops before the
    # payment rather than asking them to approve it.
    decided = compare(
        transfer_capability(),
        (LABEL,),
        {**INPUTS, "member_id": "10002"},
        INPUTS,
        profile=profile,
        surface=bank("10002"),
        clock=clock,
        sleep=clock.advance,
        control=control,
    )
    assert person.asked == []
    assert decided.complete, decided.note
    assert decided.confirmed == (Word("Search", Confirmed.COMPARISON),)


def test_a_check_whose_text_differs_for_the_second_record_confirms_the_rest(tmp_path):
    """A draft proved a step by a link named by the first record's receipt.

    The second record's page has no such link. The step's other check holds,
    so the comparison goes on and confirms every other text, and leaves the
    receipt's text unconfirmed.
    """
    from computeruse.capability import (
        LocatorForm,
        Match,
        Present,
        Shows,
        StructuralTarget,
        constant,
    )

    lookup = _lookup()
    receipt = StructuralTarget(
        "receipt",
        "/members/:id",
        LocatorForm.ACCESSIBILITY,
        "link",
        constant("RCP00000001"),
        "",
        None,
        None,
        (),
        None,
    )
    search = next(node for node in lookup.nodes if node.node_id == "search")
    proved = dataclasses.replace(
        search,
        verify=(
            Shows("member_shown", MEMBER, Match.EQUALS),
            Present("receipt"),
        ),
    )
    lookup = dataclasses.replace(
        lookup,
        targets=(*lookup.targets, receipt),
        nodes=tuple(proved if node is search else node for node in lookup.nodes),
    )
    assert validate(lookup) == ()
    code = Candidate("RCP00000001", frozenset({Place.CONTROL}))
    profile = load_profile(_profile(tmp_path, ""))
    clock = Clock()
    decided = compare(
        lookup,
        (LABEL, code),
        {"member_id": "10002"},
        {"member_id": "10001"},
        profile=profile,
        surface=bank("10002"),
        control=Control(mode=Mode.REPLAY, clock=clock),
        clock=clock,
        sleep=clock.advance,
    )
    assert [word.text for word in decided.confirmed] == ["Search"]
    assert decided.unconfirmed == (code,)
    assert decided.note == "these read differently for the second record"


@pytest.mark.rule(8)
def test_a_second_record_must_be_another_record_not_another_option(tmp_path):
    """Changing only the amount leaves the same member, so nothing is confirmed.

    Text that repeats for the same customer says nothing about whether it is
    that customer's data (R5).
    """
    profile = load_profile(_profile(tmp_path, ""))
    clock = Clock()
    decided = compare(
        transfer_capability(),
        (LABEL,),
        {**INPUTS, "amount": "999.00"},
        INPUTS,
        profile=profile,
        surface=bank("10001"),
        control=Control(mode=Mode.REPLAY, clock=clock),
        clock=clock,
        sleep=clock.advance,
    )
    assert decided.confirmed == ()
    assert decided.note == "the second record repeats an input that names a record"


@pytest.mark.rule(8)
def test_a_record_check_names_the_record_when_no_step_links_one():
    """A painted screen has no row to link, so its record check names the record.

    Without this, a capability with no record link required every input to
    differ, and a second member under the same operator confirmed nothing.
    """
    lookup = _lookup()
    lookup = dataclasses.replace(
        lookup,
        inputs=(*lookup.inputs, dataclasses.replace(lookup.inputs[0], name="operator")),
    )
    assert _record_inputs(lookup) == set()

    def found_with(check):
        return dataclasses.replace(
            lookup,
            nodes=tuple(
                dataclasses.replace(node, checks=(*node.checks, check))
                if node.node_id == "found"
                else node
                for node in lookup.nodes
            ),
        )

    shown = Shows(lookup.targets[0].target_id, MEMBER, Match.CONTAINS)
    assert _record_inputs(found_with(shown)) == set()
    record = dataclasses.replace(shown, purpose=Purpose.RECORD)
    assert _record_inputs(found_with(record)) == {"member_id"}


@pytest.mark.rule(8)
def test_comparison_confirms_only_text_it_saw(tmp_path) -> None:
    # Graduated from tests/test_known_gaps.py (R5).
    from replay_fakes import PROFILE, App, Page, _ax, _dom, dd

    from computeruse.capability import (
        AtRoute,
        Field,
        Match,
        ResultKind,
        ResultNode,
        Shows,
        ValueType,
    )
    from computeruse.capability import constant as saved_constant

    path = tmp_path / "profile.yaml"
    path.write_text(PROFILE)
    profile = load_profile(path)

    def at_desk(targets, nodes, **changes):
        base = transfer_capability()
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
            targets=tuple(targets),
            nodes=tuple(nodes),
            entry=nodes[0].node_id,
            **changes,
        )

    targets = [
        _dom("state", "/desk", "state"),
        _ax("unused", "/desk", "button", "Private case label"),
    ]
    capability = at_desk(
        targets,
        [
            ResultNode(
                "done",
                ResultKind.SUCCESS,
                "",
                (Shows("state", saved_constant("Approved"), Match.EQUALS),),
            ),
        ],
    )
    capability = dataclasses.replace(
        capability, inputs=(Field("request", ValueType.TEXT, True, 1, 20, ()),)
    )
    app = App({"desk": Page("/desk", (dd("Status", "Approved", "state"),))}, "desk", {})
    candidate = Candidate("Private case label", frozenset({Place.CONTROL}))
    clock = Clock()
    decision = compare(
        capability,
        (candidate,),
        {"request": "second"},
        {"request": "first"},
        profile=profile,
        surface=app,
        control=Control(mode=Mode.REPLAY, clock=clock),
        clock=clock,
    )
    assert "Private case label" not in [word.text for word in decision.confirmed]


def test_how_a_comparison_ended_reaches_the_review():
    compared = Decision(
        (), (LABEL,), "the replay never met some of these", "stopped", "x"
    )
    decided = decide((LABEL,), compared=compared, classifier=None, veto=None, ask=None)
    assert (decided.ended, decided.why) == ("stopped", "x")
