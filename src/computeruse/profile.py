"""Load and validate the operator-supplied policy profile.

Missing declarations grant no permission. The loader does not infer access
through defaults, merges, or fallbacks. The operator assigns risk because
the target application determines which actions are irreversible.

Each action type requires an explicit ``any`` grant. Its ``effects`` entries
can only restrict access with ``risky`` or ``deny``. They cannot grant
``safe`` access because the model chooses effect names and could otherwise
authorize its own proposal.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote

import yaml
from yaml.nodes import MappingNode

from computeruse.urls import http_origin, parse_http_url

SCHEMA_VERSION = 8
MAX_ALTERNATE_OBSERVATIONS = 10
MAX_PROFILE_BYTES = 1024 * 1024


class Risk(StrEnum):
    """The operator's declared risk for an action."""

    SAFE = "safe"
    RISKY = "risky"


class ActionKind(StrEnum):
    """Every action the executor can emit against a surface."""

    OBSERVE = "observe"
    READ = "read"
    WAIT_FOR = "wait_for"
    ASSERT = "assert"
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    MOVE = "move"
    DRAG = "drag"
    WAIT = "wait"
    TYPE = "type"
    SELECT = "select"
    PRESS_KEY = "press_key"
    DISMISS_DIALOG = "dismiss_dialog"
    ACCEPT_DIALOG = "accept_dialog"


UNSUPPORTED_ACTIONS = frozenset({"upload", "delete", "switch_context"})
"""Unsupported action names that may appear in older profiles.

The loader identifies these names so operators know to remove them.
``switch_context`` previously selected a new window. New windows now require
human intervention, and model decisions cannot select one.
"""


TARGETED_ACTIONS = frozenset(
    {
        ActionKind.ASSERT,
        ActionKind.CLICK,
        ActionKind.DOUBLE_CLICK,
        ActionKind.MOVE,
        ActionKind.DRAG,
        ActionKind.WAIT,
        ActionKind.READ,
        ActionKind.SELECT,
        ActionKind.TYPE,
        ActionKind.WAIT_FOR,
    }
)
"""Actions that must name the control they operate."""

FOCUSABLE_ACTIONS = frozenset({ActionKind.PRESS_KEY, ActionKind.SCROLL})
"""Actions that can focus a named control before sending input."""

RECORD_ACTIONS = TARGETED_ACTIONS | FOCUSABLE_ACTIONS
"""Actions that can carry record evidence, because they can name a control."""


class Limit(StrEnum):
    """An effect restriction that can require approval or deny an action."""

    RISKY = "risky"
    DENY = "deny"


class Environment(StrEnum):
    """The declared environment type. Discovery requires a sandbox.

    This declaration does not isolate the application. It records the
    operator's assertion that data is synthetic and effects are simulated,
    allowing discovery to test operations.
    """

    SANDBOX = "sandbox"
    PRODUCTION = "production"


class ObservationMode(StrEnum):
    """A permitted method for observing the application.

    ``STRUCTURED`` reports controls, roles, names, supported interactions,
    frames, and rows. ``VISUAL`` returns a masked screenshot for mouse and
    keyboard input, including controls without useful markup.
    """

    STRUCTURED = "structured"
    VISUAL = "visual"


class TimeoutBehaviour(StrEnum):
    """What happens when no operator accepts a handoff in time."""

    ABORT = "abort"


class ProfileError(ValueError):
    """The profile is malformed, incomplete, or declares something unrecognized."""


class _PolicyLoader(yaml.SafeLoader):
    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        mapping: dict[Hashable, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ProfileError("profile mapping keys must be strings")
            if key in mapping:
                raise ProfileError("profile must not contain duplicate mapping keys")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclasses.dataclass(frozen=True, slots=True)
class Budgets:
    """Limits on execution time, steps, retries, and navigation."""

    max_steps: int
    max_wall_clock_s: int
    max_retries_per_step: int
    max_navigations: int


@dataclasses.dataclass(frozen=True, slots=True)
class Perception:
    """Permitted observation methods and their retry allowance.

    ``allowed_modes`` must be nonempty. Unlisted modes are forbidden.
    ``max_alternate_observations_per_step`` limits additional observations
    of one screen before the run must act or ask for help.
    """

    allowed_modes: tuple[ObservationMode, ...]
    max_alternate_observations_per_step: int


@dataclasses.dataclass(frozen=True, slots=True)
class Records:
    """Action types and routes that require record evidence.

    A listed action on a matching route must identify its record. The policy
    gate rejects missing evidence. Other actions may include evidence too.
    These templates grant no access, so a template without an allowed route
    never applies.

    The requirement covers every action of the listed type on that route.
    If clicks require evidence, even a click that opens a menu must supply it.
    """

    actions: tuple[ActionKind, ...]
    routes: tuple[str, ...]


RECORD_NOT_FOUND = "record_not_found"
"""The application searched for exactly the goal's record and found nothing."""

RECORD_INELIGIBLE = "record_ineligible"
"""The record exists, and the application refuses the task for its state."""

PERMISSION_DENIED = "permission_denied"
"""The application refuses the task for the operator who is signed on."""

VALIDATION_FAILED = "validation_failed"
"""The application rejects a value the goal supplied."""

OUTCOMES = frozenset(
    {RECORD_NOT_FOUND, RECORD_INELIGIBLE, PERMISSION_DENIED, VALIDATION_FAILED}
)
"""The business outcomes allowed in a completion claim.

Each describes an application refusal that answers the task. Replay returns
one only when a learned branch verifies it.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class Escalation:
    """The human-intervention timeout and required timeout response."""

    handoff_timeout_s: int
    on_timeout: TimeoutBehaviour


@dataclasses.dataclass(frozen=True, slots=True)
class OriginRules:
    """Routes explicitly granted on one additional browser origin."""

    origin: str
    allow_routes: tuple[str, ...]
    deny_routes: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class BrowserScope:
    """HTTP boundaries that only a browser-aware adapter can enforce."""

    base_url: str
    allow_routes: tuple[str, ...]
    deny_routes: tuple[str, ...]
    allow_new_windows: bool
    allow_downloads: bool
    origins: tuple[OriginRules, ...] = ()

    @property
    def origin(self) -> str:
        """Return the HTTP origin this browser is allowed to reach."""
        return http_origin(self.base_url)

    def contains(self, location: str) -> bool:
        """Check the origin without reading application content."""
        try:
            return http_origin(location) in {
                self.origin,
                *(rule.origin for rule in self.origins),
            }
        except ValueError:
            return False

    def route(self, location: str) -> str | None:
        """Return the allowed route template for this browser location."""
        if not self.contains(location):
            return None
        origin = http_origin(location)
        path = parse_http_url(location).path or "/"
        if origin == self.origin:
            return self.matched_route(path)
        rule = next(rule for rule in self.origins if rule.origin == origin)
        matched = _matched(path, rule.allow_routes, rule.deny_routes)
        return None if matched is None else origin + matched

    def permits_route(self, path: str) -> bool:
        """Report whether the path is reachable under the declared routes."""
        return self.matched_route(path) is not None

    def matched_route(self, path: str) -> str | None:
        """Return the allowed template matching ``path``, or None.

        Runs record this template to identify the screen without retaining
        record identifiers from the concrete path.
        """
        return _matched(path, self.allow_routes, self.deny_routes)


def _matched(
    path: str, allowed: tuple[str, ...], denied: tuple[str, ...]
) -> str | None:
    segments = _decoded_segments(path)
    if segments is None or any(_matches(route, segments) for route in denied):
        return None
    return next((route for route in allowed if _matches(route, segments)), None)


@dataclasses.dataclass(frozen=True, slots=True)
class Submissions:
    """Form submission rules with explicit route exceptions.

    ``any`` applies unless ``routes`` defines an exception for the allowed
    route template. ``controls`` lists exact submit-button names that change
    no data, such as Review beside Commit. Action and effect rules govern
    those controls. ``None`` also delegates to action and effect rules.
    """

    any: Limit | None
    routes: Mapping[str, Limit | None] = MappingProxyType({})
    controls: frozenset[str] = frozenset()

    def limit(self, route: str, control: str = "") -> Limit | None:
        """Return the rule for a submission on ``route`` by the button ``control``."""
        if control and control in self.controls:
            return None
        return self.routes.get(route, self.any)


@dataclasses.dataclass(frozen=True, slots=True)
class OperationBinding:
    """An operator's mapping from observed controls to one business operation.

    Context entries are exact structured context paths or unique painted
    lines. A target or role of ``*`` explicitly covers every control in that
    context. The limit only adds a restriction; it grants no permission.
    """

    name: str
    operation: str
    route: str
    frame: tuple[str, ...]
    mode: ObservationMode
    context: tuple[str, ...]
    kinds: tuple[ActionKind, ...]
    target: str
    role: str
    key: str
    submission: str
    limit: Limit | None


@dataclasses.dataclass(frozen=True, slots=True)
class PickerRequirement:
    """A required field's selected state, shared by every alias of an operation."""

    operation: str
    slot: str
    source: str
    property: str


@dataclasses.dataclass(frozen=True, slots=True)
class Profile:
    """A validated policy with one enforceable environment boundary."""

    version: int
    profile_id: str
    environment: Environment
    scope: BrowserScope
    actions: Mapping[ActionKind, Risk]
    effects: Mapping[ActionKind, Mapping[str, Limit]]
    perception: Perception
    records: Records
    budgets: Budgets
    secrets: Mapping[str, str]
    escalation: Escalation
    confirmation: Mapping[str, str] = MappingProxyType({})
    """Inputs for the second test record used to confirm website text.

    Comparison requires the same text to appear for this record. An empty
    mapping disables comparison-based confirmation. Only sandbox profiles
    may declare these test inputs.
    """
    submissions: Submissions | None = None
    """Form submission rules independent of the model's effect label.

    The browser identifies submissions from observed controls. ``None``
    delegates to action and effect rules, as in profile versions 4 and 5.
    Version 6 requires an explicit submission declaration.
    """
    outcomes: Mapping[str, Mapping[str, str]] = MappingProxyType({})
    """Test inputs for learning declared business outcomes.

    Keys belong to ``OUTCOMES``. Values replace discovery inputs, such as a
    missing member for ``record_not_found`` or an unauthorized operator for
    ``permission_denied``. When outcome checks run, each entry requests a
    short discovery of that case. An empty mapping learns no outcomes.
    Only sandbox profiles may declare these inputs.
    """
    operations: tuple[OperationBinding, ...] = ()
    pickers: tuple[PickerRequirement, ...] = ()

    def effect_limit(self, kind: ActionKind, effect: str | None) -> Limit | None:
        """Return the operator's exception for this effect, if any."""
        return self.effects.get(kind, {}).get(effect) if effect is not None else None

    def requires_record(self, kind: ActionKind, path: str) -> bool:
        """Report whether this action needs trusted record evidence."""
        if kind not in self.records.actions:
            return False
        if path.startswith(("http://", "https://")):
            path = parse_http_url(path).path
        segments = _decoded_segments(path)
        return segments is None or any(
            _matches(route, segments) for route in self.records.routes
        )


def _decoded_segments(path: str) -> tuple[str, ...] | None:
    try:
        decoded = unquote(path, errors="strict")
    except UnicodeError:
        return None
    # Encoded separators and repeated encodings depend on server decoding
    # rules. Reject them rather than authorize an ambiguous destination.
    if decoded.count("/") != path.count("/") or not _canonical_path(decoded):
        return None
    return _segments(decoded)


def _segments(path: str) -> tuple[str, ...]:
    return tuple(segment for segment in path.split("/") if segment)


def _canonical_path(path: str) -> bool:
    return (
        path.startswith("/")
        and "//" not in path
        and not any(character in path for character in "%?#\\")
        and not any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in path
        )
        and not any(segment in {".", ".."} for segment in _segments(path))
    )


def _matches(route: str, segments: Sequence[str]) -> bool:
    """Match a path template against path segments.

    A ``:name`` segment matches exactly one segment. A trailing ``**`` matches
    the remainder, including nothing. Matching is structural so that route
    policy stays readable to the operator who wrote it.

    Examples
    --------
    >>> _matches("/members/:id", ("members", "12345"))
    True
    >>> _matches("/members/:id", ("members", "12345", "savings"))
    False
    >>> _matches("/members/:id/savings/**", ("members", "12345", "savings"))
    True
    >>> _matches("/members", ("admin",))
    False
    """
    pattern = _segments(route)
    for index, expected in enumerate(pattern):
        if expected == "**":
            return True
        if index >= len(segments):
            return False
        if expected.startswith(":"):
            continue
        if expected != segments[index]:
            return False
    return len(pattern) == len(segments)


def load_profile(path: Path) -> Profile:
    """Read and validate a profile file.

    Parameters
    ----------
    path
        Location of the operator's YAML profile.

    Returns
    -------
    Profile
        The validated policy, with mappings frozen against later mutation.

    Raises
    ------
    ProfileError
        If the file cannot be parsed, a required declaration is missing, or a
        declared value is not recognized.
    """
    try:
        if not path.is_file():
            raise ProfileError("profile must point to a readable local file")
        with path.open("rb") as stream:
            contents = stream.read(MAX_PROFILE_BYTES + 1)
        if len(contents) > MAX_PROFILE_BYTES:
            raise ProfileError("profile exceeds the 1 MiB size limit")
        # SafeLoader prevents arbitrary objects. This loader also rejects merges
        # and duplicate keys.
        document = yaml.load(
            contents.decode("utf-8"),
            Loader=_PolicyLoader,  # noqa: S506
        )
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError) as error:
        raise ProfileError("profile could not be read as valid UTF-8 YAML") from error
    if not isinstance(document, dict):
        raise ProfileError("profile must be a mapping of policy declarations")

    version = _require(document, "version", int)
    if version not in {4, 5, 6, 7, SCHEMA_VERSION}:
        raise ProfileError(f"profile version must be 4, 5, 6, 7, or {SCHEMA_VERSION}")

    scope = _scope(document)

    try:
        environment = Environment(_require(document, "environment", str))
    except ValueError:
        raise ProfileError("environment must be sandbox or production") from None

    grants = _require(document, "actions", dict)
    confirmation, outcomes = _confirmation(document, environment)
    profile = Profile(
        version=version,
        profile_id=_require(document, "profile_id", str),
        environment=environment,
        scope=scope,
        actions=_actions(grants),
        effects=_effects(grants),
        perception=_perception(document),
        records=_records(document),
        budgets=_budgets(document),
        secrets=_secrets(document),
        escalation=_escalation(document),
        confirmation=confirmation,
        outcomes=outcomes,
        submissions=_submissions(document, scope),
        operations=_operations(document, scope),
        pickers=_pickers(document, scope),
    )
    _reject_unknown_keys(document)
    return profile


_TOP_LEVEL_KEYS = frozenset(
    {
        "version",
        "profile_id",
        "environment",
        "base_url",
        "origins",
        "allow_routes",
        "deny_routes",
        "allow_new_windows",
        "allow_downloads",
        "actions",
        "perception",
        "records",
        "budgets",
        "secrets",
        "escalation",
        "confirmation",
        "submissions",
        "operations",
        "pickers",
    }
)


def _scope(document: Mapping[str, Any]) -> BrowserScope:
    base_url = _require(document, "base_url", str)
    _check_base_url(base_url)
    return BrowserScope(
        base_url,
        _routes(document, "allow_routes", required_non_empty=True),
        _routes(document, "deny_routes", required_non_empty=False),
        _require(document, "allow_new_windows", bool),
        _require(document, "allow_downloads", bool),
        _origins(document, base_url),
    )


_SUBMISSIONS = {"risky": Limit.RISKY, "deny": Limit.DENY, "by_effect": None}


def _submissions(
    document: Mapping[str, Any], scope: BrowserScope
) -> Submissions | None:
    """Parse form submission rules, required from profile version 6.

    ``risky`` requires approval for every submission. ``deny`` rejects them.
    ``by_effect`` delegates to action and effect rules. The long form sets
    this rule in ``any``, with optional ``routes`` exceptions and ``controls``
    naming submit buttons that change no data.
    """
    if document["version"] < 6:
        if "submissions" in document:
            raise ProfileError("submissions requires a version 6 browser profile")
        return None
    if "submissions" not in document:
        raise ProfileError("profile must declare submissions")
    declared = document["submissions"]
    if isinstance(declared, str):
        return Submissions(_submission_rule(declared))
    if not isinstance(declared, dict):
        raise ProfileError("submissions must be a rule or a mapping")
    _reject_extra(declared, ("any", "routes", "controls"), "submissions")
    routes = declared.get("routes", {})
    if not isinstance(routes, dict):
        raise ProfileError("submissions.routes must map routes to rules")
    permitted = {
        *scope.allow_routes,
        *(route for rule in scope.origins for route in rule.allow_routes),
    }
    if any(route not in permitted for route in routes):
        raise ProfileError("submissions.routes must name allow routes")
    controls = declared.get("controls", [])
    if not isinstance(controls, list) or not all(
        isinstance(name, str) and name.strip() for name in controls
    ):
        raise ProfileError("submissions.controls must list button names")
    return Submissions(
        _submission_rule(_require(declared, "any", str)),
        MappingProxyType(
            {route: _submission_rule(rule) for route, rule in routes.items()}
        ),
        frozenset(controls),
    )


def _submission_rule(declared: object) -> Limit | None:
    if not isinstance(declared, str) or declared not in _SUBMISSIONS:
        raise ProfileError("submissions must be risky, deny, or by_effect")
    return _SUBMISSIONS[declared]


def _operations(
    document: Mapping[str, Any], scope: BrowserScope
) -> tuple[OperationBinding, ...]:
    if document["version"] < 7:
        if "operations" in document:
            raise ProfileError("operations requires a version 7 browser profile")
        return ()
    declared = _require(document, "operations", list)
    if len(declared) > 128:
        raise ProfileError("operations exceeds the binding limit")
    routes = {
        *scope.allow_routes,
        *(route for origin in scope.origins for route in origin.allow_routes),
    }
    result = tuple(_operation_binding(item, routes) for item in declared)
    if len({item.name for item in result}) != len(result):
        raise ProfileError("operation binding names must be unique")
    return result


def _pickers(
    document: Mapping[str, Any], scope: BrowserScope
) -> tuple[PickerRequirement, ...]:
    if document["version"] < 8:
        if "pickers" in document:
            raise ProfileError("pickers requires a version 8 browser profile")
        return ()
    bindings = _operations(document, scope)
    items = _require(document, "pickers", list)
    if len(items) > 64:
        raise ProfileError("picker requirements exceed the limit")
    result: list[PickerRequirement] = []
    fields = ("operation", "slot", "source", "property")
    for item in items:
        if not isinstance(item, dict):
            raise ProfileError("each picker requirement must be a mapping")
        _reject_extra(item, fields, "pickers")
        values = {key: _require(item, key, str) for key in fields}
        aliases = [rule for rule in bindings if rule.operation == values["operation"]]
        if not aliases or any(
            rule.mode is not ObservationMode.STRUCTURED for rule in aliases
        ):
            raise ProfileError(
                "picker requirements need declared structured operations"
            )
        if not values["slot"].strip() or values["source"] not in {
            "native",
            "aria",
            "host",
        }:
            raise ProfileError("picker needs a field slot and a known state source")
        if (values["source"] == "host" and not values["property"].isidentifier()) or (
            values["source"] != "host" and values["property"]
        ):
            raise ProfileError("only a custom picker names a boolean property")
        requirement = PickerRequirement(**values)
        if requirement in result:
            raise ProfileError("picker requirements must be unique")
        result.append(requirement)
    return tuple(result)


def _operation_binding(item: object, routes: set[str]) -> OperationBinding:
    allowed = {
        ActionKind.CLICK,
        ActionKind.PRESS_KEY,
        ActionKind.TYPE,
        ActionKind.SELECT,
    }
    if not isinstance(item, dict):
        raise ProfileError("each operation binding must be a mapping")
    _reject_extra(
        item,
        tuple(field.name for field in dataclasses.fields(OperationBinding)),
        "operations",
    )
    strings = {
        key: _require(item, key, str)
        for key in ("name", "operation", "route", "target", "role", "key", "submission")
    }
    if any(not is_effect_name(strings[key]) for key in ("name", "operation")):
        raise ProfileError("operation names must be unique identifiers")
    if (
        strings["route"] not in routes
        or not strings["target"].strip()
        or not strings["role"].strip()
    ):
        raise ProfileError("operation binding needs an allow route, target, and role")
    context = _require(item, "context", list)
    frame = _require(item, "frame", list)
    if not context or any(
        not isinstance(value, str) or not value.strip() for value in (*context, *frame)
    ):
        raise ProfileError("operation binding needs nonempty context and frame names")
    if len(context) != len(set(context)) or len(context) > 16 or len(frame) > 16:
        raise ProfileError("operation context must be unique and bounded")
    try:
        mode = ObservationMode(_require(item, "mode", str))
        kinds = tuple(ActionKind(value) for value in _require(item, "kinds", list))
    except (ValueError, TypeError):
        raise ProfileError(
            "operation binding names an unknown mode or action"
        ) from None
    if not kinds or not set(kinds) <= allowed or len(kinds) != len(set(kinds)):
        raise ProfileError("operation bindings cover only declared input kinds")
    if mode is ObservationMode.VISUAL and strings["role"] != "canvas":
        raise ProfileError("painted operation bindings require a canvas")
    if strings["key"] and ActionKind.PRESS_KEY not in kinds:
        raise ProfileError("an operation key requires press_key")
    if strings["submission"] not in {"any", "none", "native"}:
        raise ProfileError("operation submission must be any, none, or native")
    if mode is ObservationMode.VISUAL and strings["submission"] != "any":
        raise ProfileError("a painted binding cannot identify native submission")
    limit = _submission_rule(_require(item, "limit", str))
    return OperationBinding(
        **strings,
        frame=tuple(frame),
        mode=mode,
        context=tuple(context),
        kinds=kinds,
        limit=limit,
    )


def _origins(document: Mapping[str, Any], base_url: str) -> tuple[OriginRules, ...]:
    if document["version"] == 4:
        if "origins" in document:
            raise ProfileError("additional origins require profile version 5")
        return ()
    declared = _require(document, "origins", list)
    result: list[OriginRules] = []
    known = {http_origin(base_url)}
    for item in declared:
        if not isinstance(item, dict):
            raise ProfileError("each origin needs explicit route declarations")
        _reject_extra(item, ("origin", "allow_routes", "deny_routes"), "origins")
        origin = _require(item, "origin", str)
        try:
            canonical = http_origin(origin)
        except ValueError:
            raise ProfileError("origins must be valid HTTP addresses") from None
        if canonical != origin or origin in known:
            raise ProfileError("origins must be canonical and distinct")
        known.add(origin)
        result.append(
            OriginRules(
                origin,
                _routes(item, "allow_routes", required_non_empty=True),
                _routes(item, "deny_routes", required_non_empty=False),
            )
        )
    return tuple(result)


def _reject_unknown_keys(document: Mapping[str, Any]) -> None:
    unknown = sorted(set(document) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ProfileError("profile declares unknown settings")


def _require[T](document: Mapping[str, Any], key: str, kind: type[T]) -> T:
    if key not in document:
        raise ProfileError(f"profile must declare {key}")
    value = document[key]
    # bool is a subclass of int, so an int field must reject a YAML boolean.
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ProfileError(f"{key} must be {kind.__name__}")
    return value


def _check_base_url(base_url: str) -> None:
    try:
        parse_http_url(base_url)
    except ValueError as error:
        raise ProfileError(
            "base_url must be an HTTP(S) URL without embedded credentials"
        ) from error


def _routes(
    document: Mapping[str, Any], key: str, *, required_non_empty: bool
) -> tuple[str, ...]:
    value = _require(document, key, list)
    for route in value:
        if not isinstance(route, str) or not _canonical_path(route):
            raise ProfileError(
                f"{key} entries must be canonical paths beginning with '/'"
            )
        segments = _segments(route)
        for index, segment in enumerate(segments):
            if "*" in segment and (segment != "**" or index != len(segments) - 1):
                raise ProfileError(f"{key} permits only a trailing '**' wildcard")
            if segment.startswith(":") and not segment[1:].isidentifier():
                raise ProfileError(f"{key} placeholders must have a valid name")
    if required_non_empty and not value:
        raise ProfileError(f"{key} must list at least one reachable route")
    return tuple(value)


def _actions(declared: Mapping[str, Any]) -> Mapping[ActionKind, Risk]:
    """Read the explicit grant for every declared action type.

    Each type is a mapping with an ``any`` entry, safe or risky, and an
    optional ``effects`` mapping. A bare risk string is rejected rather than
    read as a grant, so a profile written for the previous schema cannot be
    loaded as a broader one.
    """
    if not declared:
        raise ProfileError("actions must declare at least one permitted action")
    actions: dict[ActionKind, Risk] = {}
    for name, grant in declared.items():
        if name in UNSUPPORTED_ACTIONS:
            raise ProfileError(
                f"action {name} is not supported by any adapter; remove it"
            )
        try:
            kind = ActionKind(name)
        except ValueError:
            raise ProfileError("actions declares unknown action type") from None
        if not isinstance(grant, dict) or "any" not in grant:
            raise ProfileError(
                f"action {kind.value} must grant any: safe or any: risky explicitly"
            )
        _reject_extra(grant, ("any", "effects"), f"action {kind.value}")
        try:
            actions[kind] = Risk(grant["any"])
        except ValueError:
            raise ProfileError(
                f"action {kind.value} must be declared safe or risky"
            ) from None
    return MappingProxyType(actions)


def _effects(
    declared: Mapping[str, Any],
) -> Mapping[ActionKind, Mapping[str, Limit]]:
    """Parse effect rules that can only restrict their action type."""
    effects: dict[ActionKind, Mapping[str, Limit]] = {}
    for name, grant in declared.items():
        kind = ActionKind(name)
        listed = grant.get("effects", {})
        if not isinstance(listed, dict):
            raise ProfileError(f"action {kind.value} effects must be a mapping")
        if listed and kind is ActionKind.OBSERVE:
            raise ProfileError("observe does not take effects")
        limits: dict[str, Limit] = {}
        for effect, limit in listed.items():
            if not is_effect_name(effect):
                raise ProfileError(
                    "effect names use lowercase letters, digits, and underscores"
                )
            try:
                limits[effect] = Limit(limit)
            except ValueError:
                raise ProfileError(
                    f"effect exceptions of {kind.value} must be risky or deny"
                ) from None
        if limits:
            effects[kind] = MappingProxyType(limits)
    return MappingProxyType(effects)


def is_effect_name(name: object) -> bool:
    """Report whether ``name`` is a well formed effect name.

    Examples
    --------
    >>> is_effect_name("submit_payment")
    True
    >>> is_effect_name("Submit payment")
    False
    >>> is_effect_name("any")
    False
    """
    if not isinstance(name, str) or not name or name == "any":
        return False
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
    return name[0].isalpha() and all(character in allowed for character in name)


def _perception(document: Mapping[str, Any]) -> Perception:
    declared = _require(document, "perception", dict)
    listed = _require(declared, "allowed_modes", list)
    modes: list[ObservationMode] = []
    for name in listed:
        try:
            mode = ObservationMode(name)
        except (ValueError, TypeError):
            raise ProfileError(
                "perception declares an unknown observation mode"
            ) from None
        if mode in modes:
            raise ProfileError("perception must not repeat an observation mode")
        modes.append(mode)
    if not modes:
        raise ProfileError("perception.allowed_modes must grant at least one mode")
    alternates = _require(declared, "max_alternate_observations_per_step", int)
    if alternates < 0 or alternates > MAX_ALTERNATE_OBSERVATIONS:
        raise ProfileError(
            "perception.max_alternate_observations_per_step must be between 0 and "
            f"{MAX_ALTERNATE_OBSERVATIONS}"
        )
    _reject_extra(
        declared,
        ("allowed_modes", "max_alternate_observations_per_step"),
        "perception",
    )
    return Perception(
        allowed_modes=tuple(modes),
        max_alternate_observations_per_step=alternates,
    )


def _records(document: Mapping[str, Any]) -> Records:
    """Parse action types and routes requiring record evidence.

    An explicit empty list disables the requirement. A missing section is
    rejected so omission cannot silently remove evidence checks.
    """
    declared = _require(document, "records", dict)
    kinds: list[ActionKind] = []
    for name in _require(declared, "actions", list):
        if name in UNSUPPORTED_ACTIONS:
            raise ProfileError(
                f"records lists {name}, which no adapter supports; remove it"
            )
        try:
            kind = ActionKind(name)
        except (ValueError, TypeError):
            raise ProfileError("records declares an unknown action type") from None
        if kind not in RECORD_ACTIONS:
            raise ProfileError("records may list only actions that name a control")
        if kind in kinds:
            raise ProfileError("records must not repeat an action type")
        kinds.append(kind)
    routes = _routes(declared, "routes", required_non_empty=False)
    _reject_extra(declared, ("actions", "routes"), "records")
    return Records(actions=tuple(kinds), routes=routes)


def _budgets(document: Mapping[str, Any]) -> Budgets:
    declared = _require(document, "budgets", dict)
    fields = tuple(field.name for field in dataclasses.fields(Budgets))
    values = {name: _require(declared, name, int) for name in fields}
    for name, value in values.items():
        if value <= 0:
            raise ProfileError(f"budgets.{name} must be greater than zero")
    _reject_extra(declared, fields, "budgets")
    return Budgets(**values)


def _secrets(document: Mapping[str, Any]) -> Mapping[str, str]:
    declared = _require(document, "secrets", dict)
    secrets: dict[str, str] = {}
    for name, source in declared.items():
        if not isinstance(source, dict) or set(source) != {"env"}:
            raise ProfileError("each secret must declare exactly one env source")
        variable = source["env"]
        if not isinstance(variable, str) or not variable:
            raise ProfileError("each secret must name an environment variable")
        secrets[str(name)] = variable
    return MappingProxyType(secrets)


def _confirmation(
    document: Mapping[str, Any], environment: Environment
) -> tuple[Mapping[str, str], Mapping[str, Mapping[str, str]]]:
    """Parse optional sandbox test inputs for comparison and outcome discovery.

    ``inputs`` identifies a second existing record for text comparison.
    Optional ``outcomes`` entries identify inputs for learning each outcome.
    Without this section, text requires human confirmation and no outcomes
    are learned. Test inputs grant no permission.
    """
    empty: Mapping[str, str] = MappingProxyType({})
    none: Mapping[str, Mapping[str, str]] = MappingProxyType({})
    if "confirmation" not in document:
        return empty, none
    if environment is not Environment.SANDBOX:
        raise ProfileError("only a sandbox profile may name a confirmation record")
    declared = document["confirmation"]
    if not isinstance(declared, dict) or not (
        {"inputs"} <= set(declared) <= {"inputs", "outcomes"}
    ):
        raise ProfileError("confirmation declares inputs, and may declare outcomes")
    cases = declared.get("outcomes", {})
    if not isinstance(cases, dict) or any(name not in OUTCOMES for name in cases):
        raise ProfileError(
            "confirmation outcomes must name outcomes from "
            + ", ".join(sorted(OUTCOMES))
        )
    outcomes = MappingProxyType(
        {name: _test_record(case, name) for name, case in cases.items()}
    )
    return _test_record(declared["inputs"], "inputs"), outcomes


def _test_record(declared: object, part: str) -> Mapping[str, str]:
    if not isinstance(declared, dict) or not declared:
        raise ProfileError(f"confirmation {part} must name at least one input")
    record: dict[str, str] = {}
    for name, value in declared.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ProfileError(f"each confirmation {part} entry needs an identifier")
        if not isinstance(value, str) or not value.strip():
            raise ProfileError(f"each confirmation {part} entry needs a text value")
        record[name] = value
    return MappingProxyType(record)


def _escalation(document: Mapping[str, Any]) -> Escalation:
    declared = _require(document, "escalation", dict)
    timeout = _require(declared, "handoff_timeout_s", int)
    if timeout <= 0:
        raise ProfileError("escalation.handoff_timeout_s must be greater than zero")
    try:
        behaviour = TimeoutBehaviour(_require(declared, "on_timeout", str))
    except ValueError:
        raise ProfileError("escalation.on_timeout must be abort") from None
    _reject_extra(declared, ("handoff_timeout_s", "on_timeout"), "escalation")
    return Escalation(handoff_timeout_s=timeout, on_timeout=behaviour)


def _reject_extra(
    declared: Mapping[str, Any], known: Sequence[str], section: str
) -> None:
    unknown = sorted(set(declared) - set(known))
    if unknown:
        raise ProfileError(f"{section} declares unknown settings")
