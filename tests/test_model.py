"""Checks for the decider backed by the provider, with no network.

An ``httpx.MockTransport`` answers every request. These tests cover what the
project sends and what it refuses to act on when a response violates the
contract. They do not test a model's judgment.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from typing import Any

import httpx
import pytest
from fakes import grants

from computeruse.actions import (
    AxLocator,
    AxNode,
    DomAttribute,
    DomLocator,
    Observation,
    ObservationMode,
    ObservationStatus,
    PageState,
    PendingDialog,
    RecordEvidence,
    Relation,
    Scope,
    ScopeKind,
    ScreenTarget,
    SecretRef,
    VisualAnchor,
    VisualMeta,
)
from computeruse.capability import Field, ValueType
from computeruse.decider import (
    AskHuman,
    Finish,
    FlagRisk,
    InvalidDecisionError,
    ModelError,
    Observe,
    Propose,
    RiskFinding,
)
from computeruse.decider import Transcript as Script
from computeruse.escalation import Trigger
from computeruse.model import (
    MAX_OUTPUT_TOKENS,
    MODEL,
    REASONING,
    REQUEST_TIMEOUT_S,
    LunaDecider,
)
from computeruse.profile import ActionKind, Profile

PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)
HERE = "https://sandbox.example.test/members/12345"


@dataclasses.dataclass
class Exchange:
    """Records the one request the decider sent, for assertions after."""

    request: httpx.Request | None = None
    timeout: object = None

    @property
    def payload(self) -> dict[str, Any]:
        assert self.request is not None
        return json.loads(self.request.content)


def _transcript(profile: Profile, **edits: object) -> Script:
    fields: dict[str, Any] = {
        "goal": "Read the savings balance for member 12345",
        "location": HERE,
        "allowed": profile.actions,
        "permitted_modes": (ObservationMode.STRUCTURED, ObservationMode.VISUAL),
        "steps_remaining": 9,
        "seconds_remaining": 300.0,
    }
    fields.update(edits)
    return Script(**fields)


def test_postcondition_schema_cannot_confuse_state_checks_with_goal_requirements(
    profile,
):
    from computeruse.decider import Fact, Task, TaskRequirement
    from computeruse.model import tools

    transcript = _transcript(
        profile,
        task=Task(requirements=(TaskRequirement("operator", "OP0002"),)),
        memory=(Fact("account", "12345", 1, "/members/:id"),),
    )
    offered = {tool["name"]: tool for tool in tools(transcript)}
    for name in ("act", "ask_human"):
        check = offered[name]["parameters"]["properties"]["after"]["items"][
            "properties"
        ]
        assert check["kind"]["enum"] == ["state"]
        assert check["requirement"] == {"type": "null"}
    finish = offered["finish"]["parameters"]["properties"]["checks"]["items"][
        "properties"
    ]
    assert "fact" in finish["target_kind"]["enum"]
    assert "operator" in finish["requirement"]["enum"]


def _picture() -> Observation:
    return Observation(
        observation_id="obs-1",
        mode=ObservationMode.VISUAL,
        status=ObservationStatus.COMPLETE,
        page_state=PageState(HERE),
        image=PIXEL,
        visual=VisualMeta(1280, 800, 0, 0, masked=("password field",)),
    )


def _answer(name: str, arguments: dict[str, Any], **edits: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "name": name,
                "arguments": json.dumps(arguments),
            }
        ],
    }
    body.update(edits)
    return body


def _decider(body: object, exchange: Exchange | None = None) -> LunaDecider:
    """Return a decider whose provider answers with ``body`` and records the call."""
    seen = exchange or Exchange()

    def handle(request: httpx.Request) -> httpx.Response:
        seen.request = request
        seen.timeout = request.extensions.get("timeout")
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    return LunaDecider(api_key="not-a-real-key", client=client)


def _look() -> dict[str, Any]:
    return _answer("look", {"mode": "structured", "reason": "read the controls"})


def _act(**edits: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "action": "click",
        "target_kind": "accessibility",
        "role": "button",
        "name": "Open",
        "frame": [],
        "scope_kind": None,
        "scope_name": None,
        "tag": None,
        "attribute": None,
        "attribute_value": None,
        "anchor_id": None,
        "record_evidence": None,
        "effect": "open_member",
        "flag_risky": False,
        "value": None,
        "secret": None,
        "destination": None,
        "reason": "open the member",
    }
    arguments.update(edits)
    return _answer("act", arguments)


@pytest.mark.rule(8, 9)
@pytest.mark.parametrize("tool", ["act", "computer"])
def test_value_bearing_effects_are_refused_before_proposing_input(profile, tool):
    if tool == "act":
        body = _act(effect="choose_copper")
    else:
        body = _answer(
            "computer",
            {
                "capture_id": "obs-1",
                "command": {"kind": "click", "x": 10, "y": 10},
                "effect": "choose_copper",
                "flag_risky": False,
                "reason": "choose the requested option",
            },
        )
    transcript = _transcript(
        profile, inputs={"product": "Copper"}, observations=(_picture(),)
    )
    with pytest.raises(InvalidDecisionError) as caught:
        _decider(body).decide(transcript)
    assert "invocation value" in str(caught.value)


@pytest.mark.rule(1, 8)
def test_a_declared_effect_keeps_its_limit_even_when_it_matches_an_input(profile):
    from computeruse.profile import Limit

    transcript = _transcript(
        profile,
        inputs={"operation": "close_account"},
        effect_rules={ActionKind.CLICK: {"close_account": Limit.DENY}},
    )
    proposal = _decider(_act(effect="close_account", flag_risky=True)).decide(
        transcript
    )
    assert isinstance(proposal, Propose)
    assert proposal.action.effect == "close_account"
    assert proposal.action.flag_risky


def test_visual_mode_offers_screen_inputs_without_locators(profile):
    exchange = Exchange()
    command = _answer(
        "computer",
        {
            "capture_id": "obs-1",
            "command": {
                "kind": "click",
                "x": 100,
                "y": 120,
                "button": "left",
                "modifiers": [],
            },
            "effect": "focus",
            "flag_risky": False,
            "reason": "focus the field",
        },
    )
    decision = _decider(command, exchange).decide(
        _transcript(
            profile,
            permitted_modes=(ObservationMode.VISUAL,),
            observations=(_picture(),),
        )
    )
    assert isinstance(decision, Propose)
    assert isinstance(decision.action.target, ScreenTarget)
    assert decision.action.target.point is not None
    assert decision.action.target.point.x == 100
    offered = {tool["name"]: tool for tool in exchange.payload["tools"]}
    # Without the structured tool, act offers only actions on the session
    # itself, and takes no locator.
    session = offered["act"]["parameters"]["properties"]
    assert "target_kind" not in session
    assert set(session["action"]["enum"]) <= {
        "navigate",
        "scroll",
        "press_key",
        "accept_dialog",
        "dismiss_dialog",
    }
    commands = offered["computer"]["parameters"]["properties"]["command"]["anyOf"]
    kinds = {entry["properties"]["kind"]["enum"][0] for entry in commands}
    assert kinds <= {kind.value for kind in profile.actions}
    assert "type" in kinds
    assert "press_key" in kinds
    images = [
        item
        for item in exchange.payload["input"][0]["content"]
        if item["type"] == "input_image"
    ]
    assert images[0]["detail"] == "original"


@pytest.mark.parametrize(
    "edits",
    [
        {"capture_id": "old"},
        {"command": {"kind": "click", "x": "bad", "y": 0}},
        {"command": {"kind": "click", "x": True, "y": 0}},
        {"command": {"kind": "type", "text": False, "secret": None}},
        {"command": {"kind": "press_key", "keys": 42}},
        {"command": {"kind": "type", "text": "both", "secret": "login_password"}},
        {"command": {"kind": "type", "text": None, "secret": "not_declared"}},
    ],
)
def test_invalid_computer_proposals_can_be_corrected(profile, edits):
    arguments = {
        "capture_id": "obs-1",
        "command": {"kind": "click", "x": 1, "y": 2},
        "effect": "focus",
        "flag_risky": False,
        "reason": "focus",
    }
    arguments.update(edits)
    with pytest.raises(InvalidDecisionError):
        _decider(_answer("computer", arguments)).decide(
            _transcript(
                profile, observations=(_picture(),), secret_names=("login_password",)
            )
        )


def test_computer_secret_typing_keeps_only_the_reference(profile):
    command = _answer(
        "computer",
        {
            "capture_id": "obs-1",
            "command": {"kind": "type", "text": None, "secret": "login_password"},
            "effect": "authenticate",
            "flag_risky": False,
            "reason": "type credential",
        },
    )
    result = _decider(command).decide(
        _transcript(
            profile, observations=(_picture(),), secret_names=("login_password",)
        )
    )
    assert isinstance(result, Propose)
    assert result.action.value == SecretRef("login_password")


def test_the_request_names_the_configured_model_and_forbids_storage(
    profile: Profile,
) -> None:
    exchange = Exchange()

    _decider(_look(), exchange).decide(_transcript(profile))

    payload = exchange.payload
    assert payload["model"] == MODEL
    assert payload["store"] is False
    assert payload["parallel_tool_calls"] is False
    assert payload["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert payload["tool_choice"] == "required"


def test_every_request_names_its_reasoning_level(profile: Profile) -> None:
    calls = [
        (_look(), lambda decider: decider.decide(_transcript(profile))),
        (
            _task_body(
                {"name": "operator", "expected": "OP1", "of": None, "context": True}
            ),
            lambda decider: decider.interpret("As OP1, report member 12345's status."),
        ),
        (
            _answer("classify_texts", {"texts": []}),
            lambda decider: decider.classify_words(("Search",)),
        ),
        (
            _answer("screen_texts", {"texts": []}),
            lambda decider: decider.keep_words(("Search",)),
        ),
        (
            _answer("label_effect", {"effect": "search"}),
            lambda decider: decider.label_effect("button", "Search", "/members"),
        ),
        (
            _answer("fill_inputs", {"values": []}),
            lambda decider: decider.fill_inputs("Look up member 12345.", (MEMBER,)),
        ),
    ]
    for body, call in calls:
        exchange = Exchange()
        call(_decider(body, exchange))
        # No run depends on the provider's default, which is lower.
        assert exchange.payload["reasoning"] == REASONING == {"effort": "xhigh"}


MEMBER = Field("member_id", ValueType.TEXT, True, 1, 40, ())
DELIVERY = Field("delivery", ValueType.CHOICE, True, 1, 20, ("paper", "electronic"))


@pytest.mark.rule(9)
def test_filling_inputs_shows_the_model_the_goal_and_the_inputs_alone() -> None:
    exchange = Exchange()
    values = [
        {"name": "member_id", "value": "12345"},
        {"name": "delivery", "value": "paper"},
        {"name": "not_asked", "value": "x"},
    ]
    decider = _decider(_answer("fill_inputs", {"values": values}), exchange)

    filled = decider.fill_inputs("Look up member 12345.", (MEMBER, DELIVERY))

    assert filled == {"member_id": "12345", "delivery": "paper"}
    shown = json.loads(exchange.payload["input"][0]["content"][0]["text"])
    assert shown == {
        "request": "Look up member 12345.",
        "inputs": [
            {"name": "member_id", "type": "text", "choices": []},
            {"name": "delivery", "type": "choice", "choices": ["paper", "electronic"]},
        ],
    }
    assert [tool["name"] for tool in exchange.payload["tools"]] == ["fill_inputs"]


def test_an_input_answered_twice_or_a_failed_answer_fills_nothing() -> None:
    twice = [
        {"name": "member_id", "value": "12345"},
        {"name": "member_id", "value": "54321"},
    ]
    decider = _decider(_answer("fill_inputs", {"values": twice}))
    assert decider.fill_inputs("goal", (MEMBER, DELIVERY)) == {
        "member_id": None,
        "delivery": None,
    }
    wrong_tool = _decider(_answer("classify_texts", {"texts": []}))
    assert wrong_tool.fill_inputs("goal", (MEMBER,)) == {"member_id": None}


def test_the_model_can_only_take_compared_texts_away(profile: Profile) -> None:
    verdicts = [
        {"text": "Member search", "verdict": "website"},
        {"text": "Alex Morgan", "verdict": "data"},
        {"text": "Not asked", "verdict": "website"},
    ]
    body = _answer("screen_texts", {"texts": verdicts})
    asked = ("Member search", "Alex Morgan", "Status")
    # Text judged as data or omitted is not kept. Text never presented for a
    # decision is ignored.
    assert _decider(body).keep_words(asked) == frozenset({"Member search"})
    # A failed call keeps nothing, so every text goes to a person.
    failed = {"status": "failed", "output": []}
    assert _decider(failed).keep_words(asked) == frozenset()
    del profile


def test_the_request_carries_the_key_as_a_bearer_token(profile: Profile) -> None:
    exchange = Exchange()

    _decider(_look(), exchange).decide(_transcript(profile))

    assert exchange.request is not None
    assert exchange.request.headers["authorization"] == "Bearer not-a-real-key"
    assert str(exchange.request.url) == "https://api.openai.com/v1/responses"


def test_a_screenshot_is_sent_as_an_image_beside_the_state(profile: Profile) -> None:
    exchange = Exchange()

    _decider(_look(), exchange).decide(_transcript(profile, observations=(_picture(),)))

    content = exchange.payload["input"][0]["content"]
    assert exchange.payload["input"][0]["role"] == "user"
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_text"
    assert "the viewport" in content[1]["text"]
    assert content[2]["type"] == "input_image"
    assert content[2]["image_url"].startswith("data:image/png;base64,")
    assert base64.b64decode(content[2]["image_url"].split(",", 1)[1]) == PIXEL


def test_a_structured_observation_sends_no_image(profile: Profile) -> None:
    exchange = Exchange()
    observation = Observation(
        observation_id="obs-1",
        mode=ObservationMode.STRUCTURED,
        status=ObservationStatus.COMPLETE,
        page_state=PageState(HERE),
    )

    _decider(_look(), exchange).decide(
        _transcript(profile, observations=(observation,))
    )

    content = exchange.payload["input"][0]["content"]
    assert [item["type"] for item in content] == ["input_text"]


def test_only_the_permitted_tools_and_actions_are_offered(
    edited_profile,
) -> None:
    profile = edited_profile(
        actions=grants({"observe": "safe", "read": "safe", "click": "safe"}),
        perception={
            "allowed_modes": ["structured"],
            "max_alternate_observations_per_step": 1,
        },
    )
    exchange = Exchange()

    _decider(_look(), exchange).decide(
        _transcript(profile, permitted_modes=(ObservationMode.STRUCTURED,))
    )

    offered = {tool["name"]: tool for tool in exchange.payload["tools"]}
    assert offered["look"]["parameters"]["properties"]["mode"]["enum"] == ["structured"]
    # Looking is the look tool. It is never offered as an action.
    assert offered["act"]["parameters"]["properties"]["action"]["enum"] == [
        "click",
        "read",
    ]


def test_the_timeout_is_bounded_by_the_time_the_run_has_left(
    profile: Profile,
) -> None:
    exchange = Exchange()

    _decider(_look(), exchange).decide(_transcript(profile, seconds_remaining=12.0))

    assert exchange.timeout == {
        "connect": 12.0,
        "pool": 12.0,
        "read": 12.0,
        "write": 12.0,
    }


def test_the_timeout_never_exceeds_the_adapter_limit(profile: Profile) -> None:
    exchange = Exchange()

    _decider(_look(), exchange).decide(_transcript(profile, seconds_remaining=9000.0))

    assert exchange.timeout == {
        "connect": REQUEST_TIMEOUT_S,
        "pool": REQUEST_TIMEOUT_S,
        "read": REQUEST_TIMEOUT_S,
        "write": REQUEST_TIMEOUT_S,
    }


def test_a_look_becomes_an_observation_request(profile: Profile) -> None:
    decision = _decider(_look()).decide(_transcript(profile))

    assert isinstance(decision, Observe)
    assert decision.request.mode is ObservationMode.STRUCTURED


def test_an_accessibility_target_keeps_its_frame_and_scope(profile: Profile) -> None:
    body = _act(frame=["ledger"], scope_kind="row", scope_name="12345")

    decision = _decider(body).decide(_transcript(profile))

    assert isinstance(decision, Propose)
    assert decision.action.kind is ActionKind.CLICK
    target = decision.action.target
    assert isinstance(target, AxLocator)
    assert target.frame == ("ledger",)
    assert target.scope is not None
    assert (target.scope.kind, target.scope.name) == (ScopeKind.ROW, "12345")


def test_a_dom_target_is_built_from_one_exact_attribute(profile: Profile) -> None:
    body = _act(
        target_kind="dom", tag="span", attribute="id", attribute_value="posting-lock"
    )

    decision = _decider(body).decide(_transcript(profile))

    assert isinstance(decision, Propose)
    assert decision.action.target == DomLocator("span", DomAttribute.ID, "posting-lock")


def test_a_painted_target_names_a_region_from_the_screenshot(
    profile: Profile,
) -> None:
    body = _act(target_kind="visual", anchor_id="region-2", frame=["ledger"])

    decision = _decider(body).decide(_transcript(profile))

    assert isinstance(decision, Propose)
    assert decision.action.target == VisualAnchor("region-2", ("ledger",))


def test_a_secret_becomes_a_reference_and_never_a_value(profile: Profile) -> None:
    body = _act(action="type", secret="login_password", value="hunter2")

    decision = _decider(body).decide(_transcript(profile))

    assert isinstance(decision, Propose)
    assert decision.action.value == SecretRef("login_password")


def test_a_request_for_a_person_keeps_its_structured_reason(profile: Profile) -> None:
    body = _answer(
        "ask_human", {"reason": "missing_user_input", "detail": "no member id"}
    )

    decision = _decider(body).decide(_transcript(profile))

    assert decision == AskHuman(Trigger.MISSING_USER_INPUT, "no member id")


def test_finishing_reports_the_named_outputs(profile: Profile) -> None:
    body = _answer(
        "finish",
        {"outputs": [{"name": "balance", "value": "1,240.00"}], "reason": "read it"},
    )

    decision = _decider(body).decide(_transcript(profile))

    assert isinstance(decision, Finish)
    assert decision.outputs == {"balance": "1,240.00"}


@pytest.mark.parametrize(
    "body",
    [
        {"status": "incomplete", "output": []},
        {"status": "completed", "output": []},
        {"status": "completed"},
        "not an object",
    ],
    ids=["not completed", "no call", "no output list", "not a body"],
)
def test_a_response_without_one_completed_call_is_refused(
    profile: Profile, body: object
) -> None:
    with pytest.raises(ModelError):
        _decider(body).decide(_transcript(profile))


def test_two_tool_calls_are_refused(profile: Profile) -> None:
    body = _look()
    body["output"] = [*body["output"], *body["output"]]

    with pytest.raises(ModelError):
        _decider(body).decide(_transcript(profile))


def test_an_unoffered_tool_is_refused(profile: Profile) -> None:
    with pytest.raises(ModelError):
        _decider(_answer("run_javascript", {"code": "1"})).decide(_transcript(profile))


def test_arguments_that_are_not_json_are_refused(profile: Profile) -> None:
    body = _look()
    body["output"][0]["arguments"] = "{not json"

    with pytest.raises(ModelError):
        _decider(body).decide(_transcript(profile))


def test_arguments_that_are_not_an_object_are_refused(profile: Profile) -> None:
    body = _look()
    body["output"][0]["arguments"] = json.dumps(["structured"])

    with pytest.raises(ModelError):
        _decider(body).decide(_transcript(profile))


def test_an_observation_mode_the_profile_denies_is_refused(profile: Profile) -> None:
    body = _answer("look", {"mode": "visual", "reason": "look at it"})

    with pytest.raises(ModelError):
        _decider(body).decide(
            _transcript(profile, permitted_modes=(ObservationMode.STRUCTURED,))
        )


def test_an_unknown_observation_mode_is_refused(profile: Profile) -> None:
    body = _answer("look", {"mode": "xray", "reason": "look at it"})

    with pytest.raises(ModelError):
        _decider(body).decide(_transcript(profile))


def test_an_action_type_the_profile_denies_is_refused(edited_profile) -> None:
    profile = edited_profile(actions=grants({"observe": "safe", "read": "safe"}))

    with pytest.raises(ModelError):
        _decider(_act()).decide(_transcript(profile))


def test_an_action_that_is_not_well_formed_is_refused(profile: Profile) -> None:
    body = _act(action="navigate", target_kind=None, destination=None)

    with pytest.raises(ModelError):
        _decider(body).decide(_transcript(profile))


def test_a_target_without_a_role_is_refused(profile: Profile) -> None:
    with pytest.raises(ModelError):
        _decider(_act(role=None)).decide(_transcript(profile))


def test_an_unknown_targeting_method_is_refused(profile: Profile) -> None:
    with pytest.raises(ModelError):
        _decider(_act(target_kind="coordinates")).decide(_transcript(profile))


def test_an_unknown_reason_for_a_person_is_refused(profile: Profile) -> None:
    body = _answer("ask_human", {"reason": "bored", "detail": ""})

    with pytest.raises(ModelError):
        _decider(body).decide(_transcript(profile))


def test_outputs_that_are_not_a_list_are_refused(profile: Profile) -> None:
    body = _answer("finish", {"outputs": {"balance": "1.00"}, "reason": "done"})

    with pytest.raises(ModelError):
        _decider(body).decide(_transcript(profile))


def test_a_transport_failure_is_reported_as_a_model_error(profile: Profile) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    client = httpx.Client(transport=httpx.MockTransport(fail))
    decider = LunaDecider(api_key="not-a-real-key", client=client)

    with pytest.raises(ModelError):
        decider.decide(_transcript(profile))


def test_a_provider_error_status_is_reported_as_a_model_error(
    profile: Profile,
) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    decider = LunaDecider(api_key="not-a-real-key", client=client)

    with pytest.raises(ModelError):
        decider.decide(_transcript(profile))


def test_a_secret_field_is_described_without_its_value(profile: Profile) -> None:
    exchange = Exchange()
    observation = Observation(
        observation_id="obs-1",
        mode=ObservationMode.STRUCTURED,
        status=ObservationStatus.COMPLETE,
        page_state=PageState(HERE),
        nodes=(
            AxNode(
                role="textbox",
                name="Approver passcode",
                value="hunter2",
                tag="input",
                secret=True,
            ),
        ),
    )

    _decider(_look(), exchange).decide(
        _transcript(profile, observations=(observation,))
    )

    sent = json.dumps(exchange.payload["input"])
    assert "hunter2" not in sent
    assert "(withheld)" in sent


def test_the_key_appears_only_in_the_header(profile: Profile) -> None:
    exchange = Exchange()

    _decider(_look(), exchange).decide(_transcript(profile))

    assert "not-a-real-key" not in json.dumps(exchange.payload)


# Record evidence and dialogs.

EVIDENCE = {
    "source_kind": "dom",
    "role": None,
    "name": None,
    "tag": "dd",
    "attribute": "id",
    "attribute_value": "member-number",
    "value": "10001",
    "relation": "container",
}


def test_record_evidence_is_parsed_into_the_targets_frame(profile: Profile) -> None:
    body = _act(frame=["ledger"], record_evidence=EVIDENCE)

    decision = _decider(body).decide(_transcript(profile))

    assert isinstance(decision, Propose)
    assert decision.action.evidence == RecordEvidence(
        DomLocator("dd", DomAttribute.ID, "member-number", frame=("ledger",)),
        "10001",
        Relation.CONTAINER,
    )


@pytest.mark.rule(7, 17)
@pytest.mark.parametrize("use", ["action", "after", "finish", "remember"])
def test_record_evidence_accepts_the_observed_control_reference(profile, use):
    from computeruse.decider import Remember

    observed = dataclasses.replace(
        LEDGER,
        nodes=(
            AxNode("combobox", "", tag="input", slot="input|Record", control="record"),
            AxNode(
                "combobox", "", tag="input", slot="input|Product", control="product"
            ),
            *LEDGER.nodes,
        ),
    )
    evidence = {
        **EVIDENCE,
        "source_kind": "control",
        "ref": "record",
        "role": None,
        "tag": None,
        "attribute": None,
        "attribute_value": None,
    }
    check = {
        **EMPTY_TARGET,
        "kind": "result",
        "output": "balance",
        "expected": "$4,212.55",
        "target_kind": "control",
        "ref": "d1:c2",
        "record_evidence": evidence,
    }
    bodies = {
        "action": _act(record_evidence=evidence),
        "after": _act(after=[check]),
        "finish": _answer(
            "finish",
            {"outputs": [{"name": "balance", "value": "$4,212.55"}], "checks": [check]},
        ),
        "remember": _answer(
            "remember",
            {
                "key": "balance",
                "value": "$4,212.55",
                "ref": "d1:c2",
                "record_evidence": evidence,
            },
        ),
    }
    exchange = Exchange()
    decision = _decider(bodies[use], exchange).decide(
        _transcript(profile, observations=(observed,))
    )
    if isinstance(decision, Propose):
        record = (
            decision.after[0].record if use == "after" else decision.action.evidence
        )
    elif isinstance(decision, Finish):
        record = decision.checks[0].record
    else:
        assert isinstance(decision, Remember)
        record = decision.record
    assert record == RecordEvidence(
        DomLocator("input", DomAttribute.SLOT, "input|Record"),
        "10001",
        Relation.CONTAINER,
    )
    act_tool = next(tool for tool in exchange.payload["tools"] if tool["name"] == "act")
    schema = act_tool["parameters"]["properties"]["record_evidence"]["anyOf"][0]
    assert "control" in schema["properties"]["source_kind"]["enum"]
    assert "ref" in schema["required"]


@pytest.mark.rule(7)
@pytest.mark.parametrize("problem", ["missing", "stale", "ambiguous", "other_frame"])
def test_an_evidence_reference_cannot_guess_or_change_its_frame(profile, problem):
    source = AxNode("rowheader", "S-1001", tag="th", control="record")
    if problem == "other_frame":
        source = dataclasses.replace(source, frame=("other",))
    nodes = (source,)
    if problem == "ambiguous":
        nodes += (dataclasses.replace(source, control="another"),)
    observed = dataclasses.replace(LEDGER, nodes=nodes)
    evidence = {
        **EVIDENCE,
        "source_kind": "control",
        "ref": None
        if problem == "missing"
        else "stale"
        if problem == "stale"
        else "record",
        "role": None,
    }
    with pytest.raises(ModelError):
        _decider(_act(record_evidence=evidence)).decide(
            _transcript(profile, observations=(observed,))
        )


@pytest.mark.parametrize(
    "evidence",
    [
        {**EVIDENCE, "value": ""},
        {**EVIDENCE, "relation": "page"},
        {**EVIDENCE, "source_kind": "visual"},
        # The identifier would continue into its static label text.
        {**EVIDENCE, "value": "NM00000", "prefix": "Member: ", "suffix": "1"},
        {**EVIDENCE, "value": "000001", "prefix": "Member: NM", "suffix": ""},
        "member 10001",
    ],
)
def test_record_evidence_that_is_not_well_formed_is_refused(
    profile: Profile, evidence: object
) -> None:
    with pytest.raises(ModelError):
        _decider(_act(record_evidence=evidence)).decide(_transcript(profile))


def test_the_dialog_and_the_record_bound_actions_are_sent(profile: Profile) -> None:
    exchange = Exchange()
    dialog = PendingDialog("dialog-3", "confirm", "Post this entry?")
    blocked = Observation(
        observation_id="obs-4",
        mode=ObservationMode.STRUCTURED,
        status=ObservationStatus.UNAVAILABLE,
        page_state=PageState(HERE),
        dialog=dialog,
    )

    _decider(_look(), exchange).decide(
        _transcript(
            profile,
            dialog=dialog,
            observations=(blocked,),
            record_bound=(ActionKind.CLICK,),
        )
    )

    state = json.loads(exchange.payload["input"][0]["content"][0]["text"])
    expected = {"id": "dialog-3", "kind": "confirm", "message": "Post this entry?"}
    assert state["dialog_waiting"] == expected
    assert state["observations"][0]["dialog"] == expected
    assert state["actions_that_must_name_their_record"] == ["click"]


def test_every_control_reports_its_row(profile: Profile) -> None:
    exchange = Exchange()
    structure = Observation(
        observation_id="obs-5",
        mode=ObservationMode.STRUCTURED,
        status=ObservationStatus.COMPLETE,
        page_state=PageState(HERE),
        nodes=(AxNode("button", "Approve", row="el-1", ancestors=("el-1",)),),
    )

    _decider(_look(), exchange).decide(_transcript(profile, observations=(structure,)))

    state = json.loads(exchange.payload["input"][0]["content"][0]["text"])
    control = state["observations"][0]["controls"][0]
    assert control["row"] == "el-1"


def test_human_return_is_sent_even_after_transient_notices_are_consumed(profile):
    from computeruse.decider import HumanReturn

    exchange = Exchange()
    returned = HumanReturn("iv-1", True, ("clicked a canvas",))
    _decider(_look(), exchange).decide(
        _transcript(profile, human_return=returned, notices=())
    )
    state = json.loads(exchange.payload["input"][0]["content"][0]["text"])
    assert state["human_return"] == {
        "intervention": "iv-1",
        "took_control": True,
        "actions": ["clicked a canvas"],
        "verified": [],
    }


def test_effect_and_flag_are_parsed_and_the_effect_normalised(
    profile: Profile,
) -> None:
    body = _act(
        action="press_key",
        role="textbox",
        name="Amount",
        value="Enter",
        effect="Submit payment",
        flag_risky=True,
    )

    decision = _decider(body).decide(_transcript(profile))

    assert isinstance(decision, Propose)
    assert decision.action.kind is ActionKind.PRESS_KEY
    assert decision.action.target == AxLocator("textbox", "Amount")
    assert decision.action.effect == "submit_payment"
    assert decision.action.flag_risky is True


@pytest.mark.parametrize("flag", [False, None, "true"])
def test_anything_but_a_true_flag_adds_nothing(profile: Profile, flag) -> None:
    decision = _decider(_act(effect="open", flag_risky=flag)).decide(
        _transcript(profile)
    )

    assert isinstance(decision, Propose)
    assert decision.action.flag_risky is False


def test_a_finding_is_parsed_into_a_flag_decision(profile: Profile) -> None:
    body = _answer(
        "flag_risky", {"step": 7, "effect": "submit_payment", "reason": "it posted"}
    )

    decision = _decider(body).decide(_transcript(profile))

    assert decision == FlagRisk(RiskFinding(7, "submit_payment", "it posted"))


@pytest.mark.parametrize(
    "arguments",
    [
        {"step": 0, "effect": "submit_payment", "reason": ""},
        {"step": "7", "effect": "submit_payment", "reason": ""},
        {"step": True, "effect": "submit_payment", "reason": ""},
        {"step": 7, "effect": "", "reason": ""},
        {"step": 7, "effect": "post!", "reason": ""},
    ],
)
def test_a_finding_that_is_not_well_formed_is_refused(
    profile: Profile, arguments
) -> None:
    with pytest.raises(ModelError):
        _decider(_answer("flag_risky", arguments)).decide(_transcript(profile))


def test_the_flag_tool_and_the_act_fields_are_offered(profile: Profile) -> None:
    exchange = Exchange()

    _decider(_look(), exchange).decide(_transcript(profile))

    offered = {tool["name"]: tool for tool in exchange.payload["tools"]}
    assert set(offered["flag_risky"]["parameters"]["properties"]) == {
        "step",
        "effect",
        "reason",
    }
    act = offered["act"]["parameters"]["properties"]
    assert act["flag_risky"]["type"] == "boolean"
    assert act["effect"]["type"] == "string"


# Finish checks, working memory, windows, and the state older turns leave.

LEDGER = Observation(
    observation_id="obs-2",
    mode=ObservationMode.STRUCTURED,
    status=ObservationStatus.COMPLETE,
    page_state=PageState(HERE),
    nodes=(
        AxNode("rowheader", "S-1001", tag="th", control="d1:c1"),
        AxNode("cell", "$4,212.55", tag="td", control="d1:c2"),
        AxNode("heading", "Member 12345", tag="h1", control="d1:c3"),
    ),
)

EMPTY_TARGET = {
    "ref": None,
    "role": None,
    "name": None,
    "frame": [],
    "tag": None,
    "attribute": None,
    "attribute_value": None,
    "capture_id": None,
    "fact": None,
    "x": None,
    "y": None,
    "record_evidence": None,
}


def test_finish_carries_its_checks_by_ref(profile: Profile) -> None:
    from computeruse.decider import CheckKind, Match

    evidence = {
        "source_kind": "accessibility",
        "role": "rowheader",
        "name": "S-1001",
        "tag": None,
        "attribute": None,
        "attribute_value": None,
        "value": "S-1001",
        "relation": "row",
    }
    body = _answer(
        "finish",
        {
            "outputs": [{"name": "balance", "value": "$4,212.55"}],
            "checks": [
                {
                    **EMPTY_TARGET,
                    "kind": "result",
                    "output": "balance",
                    "expected": "$4,212.55",
                    "match": "equals",
                    "target_kind": "control",
                    "ref": "d1:c2",
                    "record_evidence": evidence,
                },
                {
                    **EMPTY_TARGET,
                    "kind": "record",
                    "output": None,
                    "expected": "12345",
                    "match": "contains",
                    "target_kind": "control",
                    "ref": "d1:c3",
                },
            ],
            "reason": "read",
        },
    )
    decision = _decider(body).decide(_transcript(profile, observations=(LEDGER,)))
    assert isinstance(decision, Finish)
    result, record = decision.checks
    assert result.kind is CheckKind.RESULT
    assert result.target == AxLocator("cell", "$4,212.55")
    assert result.record is not None
    assert result.record.relation is Relation.ROW
    assert record.kind is CheckKind.RECORD
    assert record.match is Match.CONTAINS


def test_remember_names_its_source_by_ref(profile: Profile) -> None:
    from computeruse.decider import Remember

    body = _answer(
        "remember",
        {"key": "account", "value": "S-1001", "may_change": False, "ref": "d1:c1"},
    )
    decision = _decider(body).decide(_transcript(profile, observations=(LEDGER,)))
    assert decision == Remember(
        "account", "S-1001", may_change=False, source=AxLocator("rowheader", "S-1001")
    )


def test_a_look_can_ask_for_the_next_window(profile: Profile) -> None:
    body = _answer(
        "look",
        {
            "mode": "structured",
            "reason": "more",
            "start": 200,
            "frame": None,
            "scope_kind": "region",
            "scope_name": "Savings",
        },
    )
    seen = Observation(
        "scoped",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        PageState(HERE),
        nodes=(AxNode("textbox", "Amount", scope=Scope(ScopeKind.REGION, "Savings")),),
    )
    decision = _decider(body).decide(_transcript(profile, observations=(seen,)))
    assert isinstance(decision, Observe)
    assert decision.request.start == 200
    assert decision.request.scope is not None
    assert decision.request.scope.name == "Savings"


@pytest.mark.rule(2, 17)
def test_a_heading_cannot_invent_an_observation_scope(profile):
    from computeruse.model import tools

    seen = Observation(
        "unscoped",
        ObservationMode.STRUCTURED,
        ObservationStatus.COMPLETE,
        PageState(HERE),
        nodes=(AxNode("button", "Send", context=("heading:Review",)),),
    )
    transcript = _transcript(profile, observations=(seen,))
    schema = next(tool for tool in tools(transcript) if tool["name"] == "look")
    assert schema["parameters"]["properties"]["scope_name"]["enum"] == [None]
    body = _answer(
        "look",
        dict(
            mode="structured", reason="inspect", scope_kind="form", scope_name="Review"
        ),
    )
    with pytest.raises(ModelError):
        _decider(body).decide(transcript)


def test_older_turns_and_memory_stay_in_the_state_without_a_secret(
    profile: Profile,
) -> None:
    from computeruse.actions import Action, Outcome
    from computeruse.decider import Fact, Turn
    from computeruse.model import MAX_HISTORY, _state

    field = AxLocator("textbox", "Password")
    turns = [
        Turn(
            1,
            Action(ActionKind.TYPE, field, SecretRef("login_password")),
            None,
            Outcome.OK,
        ),
        *[
            Turn(
                step,
                Action(ActionKind.READ, AxLocator("cell", "Balance")),
                outcome=Outcome.OK,
                extracted="$4.00",
            )
            for step in range(2, 2 + MAX_HISTORY + 3)
        ],
    ]
    fact = Fact("member", "12345", step=1, route="/members", may_change=False)
    state = json.loads(
        _state(_transcript(profile, history=tuple(turns), memory=(fact,)))
    )
    assert len(state["recent"]) == MAX_HISTORY
    assert (
        state["earlier"][0]
        == 'step 1: type textbox "Password" value secret login_password -> ok'
    )
    assert state["working_memory"][0]["value"] == "12345"
    assert state["recent"][-1]["target"] == 'cell "Balance"'


def test_a_value_from_memory_is_named_for_the_loop(profile: Profile) -> None:
    from computeruse.decider import Fact

    fact = Fact("member", "12345", step=1, route="/members", may_change=False)
    body = _act(
        action="type",
        role="textbox",
        name="Member number",
        value=None,
        value_from_fact="member",
    )
    decision = _decider(body).decide(_transcript(profile, memory=(fact,)))
    assert isinstance(decision, Propose)
    assert decision.fact == "member"
    assert decision.action.value == "12345"


def test_a_control_given_by_role_and_name_without_a_ref_is_resolved(
    profile: Profile,
) -> None:
    body = _act(target_kind="control", ref=None, role="button", name="Open")
    decision = _decider(body).decide(_transcript(profile))
    assert isinstance(decision, Propose)
    assert decision.action.target == AxLocator("button", "Open")
    with pytest.raises(InvalidDecisionError) as refused:
        _decider(_act(target_kind="control", ref=None, role=None)).decide(
            _transcript(profile)
        )
    assert "needs the ref" in str(refused.value)


def test_a_check_that_names_only_a_fact_is_a_fact_check(profile: Profile) -> None:
    from computeruse.decider import Fact, FactRef

    fact = Fact("branch", "Riverside", step=2, route="/members", may_change=False)
    check = {
        **EMPTY_TARGET,
        "kind": "result",
        "output": "branch",
        "expected": "Riverside",
        "match": "equals",
        "target_kind": "control",
        "fact": "branch",
    }
    body = _answer(
        "finish",
        {
            "outputs": [{"name": "branch", "value": "Riverside"}],
            "checks": [check],
            "reason": "",
        },
    )
    decision = _decider(body).decide(_transcript(profile, memory=(fact,)))
    assert isinstance(decision, Finish)
    assert decision.checks[0].target == FactRef("branch")


def test_a_malformed_check_is_named_with_its_reason(profile: Profile) -> None:
    bad = {
        **EMPTY_TARGET,
        "kind": "result",
        "output": "balance",
        "expected": "$1.00",
        "match": "equals",
        "target_kind": "accessibility",
        "name": "$1.00",
    }
    body = _answer(
        "finish",
        {
            "outputs": [{"name": "balance", "value": "$1.00"}],
            "checks": [bad],
            "reason": "",
        },
    )
    with pytest.raises(InvalidDecisionError) as refused:
        _decider(body).decide(_transcript(profile))
    assert str(refused.value).startswith("check 1: ")
    assert "must name a role" in str(refused.value)


def test_a_session_action_with_an_empty_target_takes_no_target(
    profile: Profile,
) -> None:
    """The schema asks for every field, so a scroll can arrive with a bare kind."""
    body = _act(
        action="scroll",
        target_kind="accessibility",
        role=None,
        name=None,
        value="down",
    )
    decision = _decider(body).decide(_transcript(profile))
    assert isinstance(decision, Propose)
    assert decision.action.target is None
    assert decision.action.value == "down"


def test_the_task_is_read_from_the_goal_alone(profile: Profile) -> None:
    from computeruse.decider import Task, TaskOutput, TaskRecord

    body = _answer(
        "declare_task",
        {
            "records": [
                {"name": "member", "value": "12345", "within": None},
                {"name": "account", "value": "S-1001", "within": "member"},
            ],
            "outputs": [{"name": "balance", "of": "account"}],
            "question": None,
            "requirements": [],
            "changes": False,
        },
    )
    exchange = Exchange()
    task = _decider(body, exchange).interpret(
        "Report the balance of savings account S-1001 for member 12345.",
        ("an earlier reading was refused",),
    )
    assert task == Task(
        (TaskRecord("member", "12345"), TaskRecord("account", "S-1001", "member")),
        (TaskOutput("balance", "account"),),
    )
    payload = exchange.payload
    assert [tool["name"] for tool in payload["tools"]] == ["declare_task"]
    content = payload["input"][0]["content"]
    assert [item["type"] for item in content] == ["input_text"]
    shown = json.loads(content[0]["text"])
    assert shown == {
        "request": "Report the balance of savings account S-1001 for member 12345.",
        "notices": ["an earlier reading was refused"],
    }
    del profile


def _task_body(requirement: dict[str, object]) -> dict[str, Any]:
    return _answer(
        "declare_task",
        {
            "records": [{"name": "member", "value": "12345", "within": None}],
            "outputs": [{"name": "status", "of": "member"}],
            "question": None,
            "requirements": [requirement],
            "changes": False,
        },
    )


def test_each_requirement_says_whether_it_is_context(profile: Profile) -> None:
    from computeruse.decider import TaskRequirement

    body = _task_body(
        {"name": "institution", "expected": "Acme", "of": None, "context": True}
    )
    task = _decider(body).interpret("As Acme staff, report member 12345's status.")
    assert task.requirements == (TaskRequirement("institution", "Acme", "", True),)
    # Context is never assumed when the provider leaves it unsaid.
    unsaid = _task_body({"name": "institution", "expected": "Acme", "of": None})
    with pytest.raises(InvalidDecisionError):
        _decider(unsaid).interpret("As Acme staff, report member 12345's status.")
    del profile


def test_a_task_that_is_not_a_task_is_sent_back(profile: Profile) -> None:
    body = _answer("declare_task", {"records": "member 12345", "outputs": []})
    with pytest.raises(InvalidDecisionError):
        _decider(body).interpret("Report member 12345's balance.")
    with pytest.raises(InvalidDecisionError):
        _decider(_look()).interpret("Report member 12345's balance.")
    del profile


def test_remember_carries_record_evidence(profile: Profile) -> None:
    from computeruse.decider import Remember

    body = _answer(
        "remember",
        {
            "key": "balance",
            "value": "$4,212.55",
            "may_change": True,
            "ref": "d1:c2",
            "record_evidence": {
                "source_kind": "accessibility",
                "role": "rowheader",
                "name": "S-1001",
                "tag": None,
                "attribute": None,
                "attribute_value": None,
                "value": "S-1001",
                "relation": "row",
            },
        },
    )
    decision = _decider(body).decide(_transcript(profile, observations=(LEDGER,)))
    assert isinstance(decision, Remember)
    assert decision.record is not None
    assert decision.record.value == "S-1001"
    assert decision.record.relation is Relation.ROW
    assert decision.source is not None
    assert decision.record.source.frame == decision.source.frame


def test_the_state_shows_the_settled_task(profile: Profile) -> None:
    from computeruse.decider import Task, TaskOutput, TaskRecord

    exchange = Exchange()
    task = Task((TaskRecord("member", "12345"),), (TaskOutput("balance", "member"),))
    _decider(_look(), exchange).decide(_transcript(profile, task=task))
    state = json.loads(exchange.payload["input"][0]["content"][0]["text"])
    assert state["task"] == {
        "records": [{"name": "member", "value": "12345", "within": None}],
        "outputs": [{"name": "balance", "of": "member"}],
        "requirements": [],
        "changes": False,
    }


def test_the_model_cannot_ask_for_a_reason_only_the_run_raises(
    profile: Profile,
) -> None:
    """Found live: the model asked for new_window itself, with no window open."""
    exchange = Exchange()
    _decider(_look(), exchange).decide(_transcript(profile))
    offered = {tool["name"]: tool for tool in exchange.payload["tools"]}
    reasons = offered["ask_human"]["parameters"]["properties"]["reason"]["enum"]
    assert "new_window" not in reasons
    assert "ambiguous_task" not in reasons
    body = _answer("ask_human", {"reason": "new_window", "detail": "open it"})
    with pytest.raises(InvalidDecisionError):
        _decider(body).decide(_transcript(profile))


def test_a_not_found_finish_carries_a_state_check_and_its_outcome(
    profile: Profile,
) -> None:
    from computeruse.decider import CheckKind

    exchange = Exchange()
    body = _answer(
        "finish",
        {
            "outputs": [],
            "checks": [
                {
                    **EMPTY_TARGET,
                    "kind": "state",
                    "output": None,
                    "expected": "No members match your search.",
                    "match": "equals",
                    "target_kind": "control",
                    "ref": "d1:c2",
                },
                {
                    **EMPTY_TARGET,
                    "kind": "record",
                    "output": None,
                    "expected": "12345",
                    "match": "equals",
                    "target_kind": "control",
                    "ref": "d1:c3",
                },
            ],
            "reason": "nothing matched",
            "outcome": "record_not_found",
        },
    )
    decision = _decider(body, exchange).decide(
        _transcript(profile, observations=(LEDGER,))
    )
    assert isinstance(decision, Finish)
    assert decision.outcome == "record_not_found"
    assert decision.outputs == {}
    assert [check.kind for check in decision.checks] == [
        CheckKind.STATE,
        CheckKind.RECORD,
    ]
    offered = {tool["name"]: tool for tool in exchange.payload["tools"]}
    finish = offered["finish"]["parameters"]["properties"]
    assert "state" in finish["checks"]["items"]["properties"]["kind"]["enum"]
    assert finish["outcome"]["enum"] == [
        "permission_denied",
        "record_ineligible",
        "record_not_found",
        "validation_failed",
        None,
    ]


def test_a_refused_answer_names_the_tool_it_called(profile: Profile) -> None:
    from computeruse.decider import InvalidDecisionError

    body = _answer("finish", {"outputs": "not a list", "checks": [], "reason": ""})
    with pytest.raises(InvalidDecisionError) as refused:
        _decider(body).decide(_transcript(profile))
    assert refused.value.tool == "finish"


class _Waits:
    """A clock that moves only when the client sleeps, and what it slept."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _flaky(
    statuses: list[int], body: object, headers: dict[str, str] | None = None
) -> tuple[LunaDecider, list[int], _Waits]:
    """A provider answering with each status in turn, then ``body``."""
    seen: list[int] = []
    waits = _Waits()

    def handle(request: httpx.Request) -> httpx.Response:
        del request
        status = statuses.pop(0) if statuses else 200
        seen.append(status)
        if status == 200:
            return httpx.Response(status, json=body)
        return httpx.Response(status, json={}, headers=headers or {})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    decider = LunaDecider(
        api_key="not-a-real-key", client=client, sleep=waits.sleep, clock=waits.clock
    )
    return decider, seen, waits


def test_a_transient_provider_failure_is_tried_again_after_a_wait(
    profile: Profile,
) -> None:
    decider, seen, waits = _flaky([503, 503], _look())
    assert isinstance(decider.decide(_transcript(profile)), Observe)
    assert seen == [503, 503, 200]
    assert waits.slept == [1.0, 2.0]


def test_a_rate_limit_waits_as_long_as_the_provider_asks(profile: Profile) -> None:
    decider, seen, waits = _flaky([429], _look(), {"x-ratelimit-reset-tokens": "6.5s"})
    assert isinstance(decider.decide(_transcript(profile)), Observe)
    assert seen == [429, 200]
    assert waits.slept == [6.5]


def test_retries_are_bounded_and_a_refusal_is_not_tried_again(
    profile: Profile,
) -> None:
    always, seen, _ = _flaky([503] * 10, _look())
    with pytest.raises(ModelError):
        always.decide(_transcript(profile))
    assert len(seen) == 4
    refused, seen, _ = _flaky([400], _look())
    with pytest.raises(ModelError):
        refused.decide(_transcript(profile))
    assert seen == [400]


def test_no_wait_outlasts_the_time_the_run_has_left(profile: Profile) -> None:
    # The provider asks for ten minutes. The run has two.
    decider, seen, waits = _flaky([429], _look(), {"retry-after": "600"})
    transcript = dataclasses.replace(_transcript(profile), seconds_remaining=120.0)
    with pytest.raises(ModelError):
        decider.decide(transcript)
    assert seen == [429]
    assert waits.slept == []


def test_an_account_without_credit_is_not_tried_again(profile: Profile) -> None:
    seen: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        del request
        seen.append(429)
        return httpx.Response(
            429, json={"error": {"type": "insufficient_quota", "code": "x"}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handle))
    decider = LunaDecider(api_key="not-a-real-key", client=client, sleep=_Waits().sleep)
    with pytest.raises(ModelError):
        decider.decide(_transcript(profile))
    assert seen == [429]
