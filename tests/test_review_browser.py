"""General markup regressions from the website review."""

import dataclasses
import time

import pytest

from computeruse.actions import (
    Action,
    AxLocator,
    DomAttribute,
    DomLocator,
    ObservationMode,
    ObservationRequest,
    ObservationStatus,
    Outcome,
    RecordEvidence,
    Relation,
    Scope,
    ScopeKind,
)
from computeruse.coverage import gather
from computeruse.profile import ActionKind
from computeruse.retarget import complete

pytest_plugins = ("test_takeover",)
STRUCTURED = ObservationRequest(ObservationMode.STRUCTURED)


def test_native_options_expose_values_and_select_across_composite_labels(recorded):
    markup = """<label>Operator<select>
    <option value="OP1">Person One (OP1)</option>
    <option value="OP2">Person Two (OP2)</option>
    <option value="OP3" disabled>Unavailable (OP3)</option>
    </select></label>"""
    with recorded({"/": markup}) as (surface, _):
        seen = surface.observe(STRUCTURED)
        node = next(node for node in seen.nodes if node.role == "combobox")
        assert [(option.value, option.disabled) for option in node.options] == [
            ("OP1", False),
            ("OP2", False),
            ("OP3", True),
        ]
        target = AxLocator("combobox", "Operator")
        assert (
            surface.act(Action(ActionKind.SELECT, target, "OP2")).outcome is Outcome.OK
        )
        assert (
            surface.act(Action(ActionKind.READ, target)).extracted == "Person Two (OP2)"
        )
        assert (
            surface.act(Action(ActionKind.SELECT, target, "OP3")).outcome
            is Outcome.NOT_ACTIONABLE
        )
        surface._page.locator("select").evaluate(
            "(el, mark) => el.setAttribute(mark.name, mark.token)", surface._mark()
        )
        protected = surface.observe(STRUCTURED)
        hidden = next(node for node in protected.nodes if node.role == "combobox")
        assert hidden.secret
        assert not hidden.options
        assert hidden.value is None


def test_native_select_refuses_conflicting_label_and_value(recorded):
    markup = """<label>Operator<select>
    <option value="original">Original</option>
    <option value="OP2">Another label</option><option value="other">OP2</option>
    </select></label>"""
    with recorded({"/": markup}) as (surface, _):
        target = AxLocator("combobox", "Operator")
        assert (
            surface.act(Action(ActionKind.SELECT, target, "OP2")).outcome
            is Outcome.NOT_ACTIONABLE
        )
        assert surface.act(Action(ActionKind.READ, target)).extracted == "Original"


def test_dialog_cannot_block_outer_script_deadline(recorded):
    from playwright.sync_api import TimeoutError as DriverTimeout
    from test_takeover import bounded

    with bounded(10), recorded({"/": "<h1>Ready</h1>"}) as (surface, _):
        surface._limits = dataclasses.replace(surface._limits, operation_ms=150)
        started = time.monotonic()
        with pytest.raises(DriverTimeout):
            surface._ask(
                surface._page.main_frame,
                "() => { confirm('Hold'); return 'ok'; }",
                None,
            )
        assert time.monotonic() - started < 3
        assert surface._pending
        surface._pending["page-1"].dialog.dismiss()
        surface.idle(0.1)
        assert surface.observe(STRUCTURED).usable


def test_script_deadline_does_not_wait_for_driver_cancellation(recorded):
    import asyncio

    from playwright.sync_api import TimeoutError as DriverTimeout
    from test_takeover import bounded

    with bounded(10), recorded({"/": "<h1>Ready</h1>"}) as (surface, _):
        surface._limits = dataclasses.replace(surface._limits, operation_ms=50)
        released = asyncio.Event()
        finished = []

        async def waiting_for_dialog():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await released.wait()
            finished.append(True)

        started = time.monotonic()
        with pytest.raises(DriverTimeout):
            surface._bounded(surface._page.main_frame, waiting_for_dialog())
        assert time.monotonic() - started < 2
        released.set()
        surface.idle(0.1)
        assert finished == [True]


def test_expand_controls_do_not_replace_row_identity(recorded):
    markup = """<table><tbody>
    <tr><td><button>+</button></td><td><a href="#">10001</a></td>
    <td><button>Open</button></td></tr>
    <tr><td><button>+</button></td><td><a href="#">10002</a></td>
    <td><button>Open</button></td></tr>
    </tbody></table>"""
    with recorded({"/": markup}) as (surface, _):
        observed = surface.observe(STRUCTURED)
        buttons = [node for node in observed.nodes if node.name == "Open"]
        assert {node.scope for node in buttons} == {
            Scope(ScopeKind.ROW, "10001"),
            Scope(ScopeKind.ROW, "10002"),
        }
        result = surface.act(
            Action(
                ActionKind.CLICK,
                AxLocator("button", "Open", scope=Scope(ScopeKind.ROW, "10002")),
            )
        )
        assert result.outcome is Outcome.OK


def test_composite_record_label_preserves_exact_identity(recorded):
    markup = """<section><span id="member">Member: NM000001</span>
    <button onclick="this.textContent='Posted'">Post</button></section>"""
    source = DomLocator("span", DomAttribute.ID, "member")
    evidence = RecordEvidence(source, "NM000001", Relation.CONTAINER, "Member: ")
    with recorded({"/": markup}) as (surface, _):
        surface.observe(STRUCTURED)
        action = Action(
            ActionKind.CLICK, AxLocator("button", "Post"), evidence=evidence
        )
        surface._page.locator("#member").text_content()
        surface._page.evaluate(
            "document.getElementById('member').textContent='Member: NM0000010'"
        )
        assert surface.act(action).outcome is Outcome.STALE
        surface._page.evaluate(
            "document.getElementById('member').textContent='Member: NM000001'"
        )
        assert surface.act(action).outcome is Outcome.OK


def test_default_browser_limit_can_cover_a_large_document(recorded):
    markup = "".join(f"<button>Action {index}</button>" for index in range(571))
    with recorded({"/": markup}) as (surface, _):
        first = surface.observe(STRUCTURED)
        assert not complete(first)
        full = gather(first, STRUCTURED, surface.observe)
        assert complete(full)
        assert len(full.nodes) == 571


def test_grid_slot_follows_its_header_when_columns_and_records_change(recorded):
    def grid(member, state, state_index, member_index):
        return f"""<div role="grid">
        <div role="row">
        <div role="columnheader" aria-colindex="{member_index}">Record</div>
        <div role="columnheader" aria-colindex="{state_index}">State</div></div>
        <div role="row" aria-label="{member}">
        <div role="gridcell" aria-colindex="{member_index}">{member}</div>
        <div role="gridcell" aria-colindex="{state_index}">{state}</div></div></div>"""

    with recorded({"/": grid("10001", "Active", 2, 1)}) as (surface, _):
        for member, expected, markup in (
            ("10001", "Active", None),
            ("10002", "Inactive", grid("10002", "Inactive", 1, 2)),
        ):
            if markup:
                surface._page.set_content(markup)
            seen = surface.observe(STRUCTURED)
            state = next(node for node in seen.nodes if node.name == expected)
            assert state.slot == "div|State"
            target = DomLocator(
                "div", DomAttribute.SLOT, state.slot, scope=Scope(ScopeKind.ROW, member)
            )
            source = AxLocator("gridcell", member)
            read = surface.act(
                Action(
                    ActionKind.READ,
                    target,
                    evidence=RecordEvidence(source, member, Relation.ROW),
                )
            )
            assert read.outcome is Outcome.OK
            assert read.extracted == expected


def test_record_boundaries_remain_distinct_across_observation_pages(recorded):
    from computeruse.loop import _record_problem

    rows = "".join(
        f"<tr><td>{index:05}</td><td>Active</td><td><button>Open</button></td></tr>"
        for index in range(180)
    )
    with recorded({"/": f"<table>{rows}</table>"}) as (surface, _):
        first = surface.observe(STRUCTURED)
        assert not complete(first)
        full = gather(first, STRUCTURED, surface.observe)
        assert complete(full)
        buttons = [node for node in full.nodes if node.name == "Open"]
        assert len({node.row for node in buttons}) == 180
        for button in buttons:
            assert button.scope is not None
            source = AxLocator("cell", button.scope.name, scope=button.scope)
            record = RecordEvidence(source, button.scope.name, Relation.ROW)
            assert _record_problem(full, button, record) is None
        later = surface.observe(STRUCTURED)
        assert full.nodes[0].ancestors == later.nodes[0].ancestors


def test_a_record_specific_id_is_not_selected_for_reuse(recorded):
    from computeruse.recorder import LocatorForm, TargetSample
    from computeruse.recording import target_sample

    with recorded({"/": '<button id="open-NM000001">Open member</button>'}) as (
        surface,
        _,
    ):
        observation = surface.observe(STRUCTURED)
        target = target_sample(
            DomLocator("button", DomAttribute.ID, "open-NM000001"),
            "/**",
            observation,
            inputs=("NM000001",),
            safe_text=frozenset({"Open member"}),
        )
        assert isinstance(target, TargetSample)
        assert target.form is LocatorForm.ACCESSIBILITY
        assert target.name == "Open member"
        other = dataclasses.replace(
            observation.nodes[0], attributes=(("id", "open-NM000002"),)
        )
        assert other.name == target.name


def test_inline_context_value_can_use_its_label_without_saving_a_persons_name(recorded):
    from computeruse.recorder import LocatorForm, TargetSample
    from computeruse.recording import target_sample

    markup = "<span>Operator</span><strong>Alex Example (OP0002)</strong>"
    with recorded({"/": markup}) as (surface, _):
        seen = surface.observe(STRUCTURED)
        sample = target_sample(
            AxLocator("generic", "Alex Example (OP0002)"),
            "/**",
            seen,
            inputs=("OP0002",),
            safe_text=frozenset({"strong|Operator"}),
        )
        assert isinstance(sample, TargetSample)
        assert sample.form is LocatorForm.DOM
        assert sample.attribute is DomAttribute.SLOT
        assert sample.value == "strong|Operator"
        surface._page.locator("strong").evaluate(
            "el => el.textContent='Jordan (OP0003)'"
        )
        read = surface.act(
            Action(
                ActionKind.READ, DomLocator("strong", DomAttribute.SLOT, sample.value)
            )
        )
        assert read.extracted == "Jordan (OP0003)"


def test_a_dialog_opening_during_collection_leaves_the_frame_unread(
    recorded, monkeypatch
):
    from computeruse.browser import BrowserSurface

    original = BrowserSurface._bounded

    def bounded(self, frame, work):
        result = original(self, frame, work)
        if getattr(work, "__name__", "") == "collect":
            # As if a confirm opened while the collector script ran.
            self._pending["page-late"] = None
        return result

    with recorded({"/": "<h1>Ready</h1>"}) as (surface, _):
        monkeypatch.setattr(BrowserSurface, "_bounded", bounded)
        try:
            seen = surface.observe(STRUCTURED)
        finally:
            monkeypatch.undo()
            surface._pending.pop("page-late", None)
        assert seen.status is not ObservationStatus.COMPLETE
        assert "a dialog interrupted the observation" in seen.notes


def test_an_observation_waits_a_bounded_time_for_the_document_to_load(
    recorded, monkeypatch
):
    from playwright.sync_api import Page

    waits = []
    original = Page.wait_for_load_state

    def wait(self, state=None, timeout=None):
        waits.append((state, timeout))
        return original(self, state, timeout=timeout)

    with recorded({"/": "<h1>Ready</h1>"}) as (surface, _):
        monkeypatch.setattr(Page, "wait_for_load_state", wait)
        surface.observe(STRUCTURED)
        monkeypatch.undo()
    # Child frames can lag behind the main document, so the full load is
    # awaited, and only for a bounded time.
    assert any(state == "load" and timeout for state, timeout in waits)


def test_the_private_driver_hooks_the_script_deadline_uses_still_exist(recorded):
    # The outer deadline reaches below Playwright's public API. A driver
    # upgrade that moves these names must fail here, not at a live dialog.
    from playwright._impl._sync_base import mapping

    with recorded({"/": "<h1>Ready</h1>"}) as (surface, _):
        frame = surface._page.main_frame
        assert callable(mapping.to_impl)
        assert callable(frame._sync)
        assert callable(frame._impl_obj.wait_for_function)


CONTEXT_PAGE = """<!doctype html><html><body><h1>Workstation</h1>
<label>Operator <select id="operator"><option>OP0001</option>
  <option selected>OP0002</option></select></label>
<label>Status <select id="status"><option selected>Pending</option>
  <option>Approved</option></select></label>
</body></html>"""


def test_a_field_the_application_set_can_prove_a_required_state(recorded):
    from computeruse.actions import Expectation

    with recorded({"/": CONTEXT_PAGE}) as (surface, _):
        seen = surface.observe(STRUCTURED)
        committed = Expectation(seen.page_state, committed=True)
        operator = DomLocator("select", DomAttribute.ID, "operator")
        read = surface.act(Action(ActionKind.READ, operator), expect=committed)
    assert read.outcome is Outcome.OK
    assert read.extracted == "OP0002"


def test_a_field_this_run_changed_cannot_prove_a_required_state(recorded):
    from computeruse.actions import Expectation

    with recorded({"/": CONTEXT_PAGE}) as (surface, _):
        status = DomLocator("select", DomAttribute.ID, "status")
        # Selecting "Approved" without saving must not complete an approval.
        surface.act(Action(ActionKind.SELECT, status, "Approved"))
        seen = surface.observe(STRUCTURED)
        committed = Expectation(seen.page_state, committed=True)
        changed = surface.act(Action(ActionKind.READ, status), expect=committed)
        operator = DomLocator("select", DomAttribute.ID, "operator")
        untouched = surface.act(Action(ActionKind.READ, operator), expect=committed)
        # After a reload the application renders the field again.
        surface._page.reload()
        seen = surface.observe(STRUCTURED)
        committed = Expectation(seen.page_state, committed=True)
        reloaded = surface.act(Action(ActionKind.READ, status), expect=committed)
    assert changed.outcome is Outcome.NOT_ACTIONABLE
    assert untouched.outcome is Outcome.OK
    assert reloaded.outcome is Outcome.OK
    assert reloaded.extracted == "Pending"
